"""
Niche Document Extraction API  (v1.0)
Reads an Indian invoice (photo or PDF) and returns clean structured JSON.

Environment variables (set these in Render > Environment):
  ANTHROPIC_API_KEY       your key from console.anthropic.com   (required)
  RAPIDAPI_PROXY_SECRET   the secret RapidAPI sends you          (required)
  ANTHROPIC_MODEL         optional, default claude-sonnet-5-5
                          (use claude-haiku-5-5 for a cheaper, faster option)
"""

import base64
import hmac
import io
import logging
import os
import time
from typing import Any, Optional

from anthropic import APIError, AsyncAnthropic
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, Field, ValidationError

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("parser")

PROXY_SECRET = os.getenv("RAPIDAPI_PROXY_SECRET", "")
MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")
MAX_UPLOAD_BYTES = 15 * 1024 * 1024   # reject anything bigger than 15 MB
MAX_IMAGE_EDGE = 1800                 # shrink big phone photos before sending

app = FastAPI(
    title="Niche Document Extraction API",
    version="1.0.0",
    description="Upload an invoice photo or PDF, get structured JSON back.",
)

# --------------------------------------------------------------------------
# Response models
# --------------------------------------------------------------------------


class Party(BaseModel):
    name: Optional[str] = None
    gstin: Optional[str] = Field(None, description="15-character GSTIN if printed")
    pan: Optional[str] = None
    address: Optional[str] = None


class LineItem(BaseModel):
    description: Optional[str] = None
    hsn_sac: Optional[str] = Field(None, description="HSN or SAC code if printed")
    quantity: Optional[float] = None
    unit: Optional[str] = None
    unit_price: Optional[float] = None
    tax_rate_percent: Optional[float] = None
    amount: Optional[float] = Field(None, description="Line amount as printed")


class Totals(BaseModel):
    subtotal: Optional[float] = None
    discount: Optional[float] = None
    cgst: Optional[float] = None
    sgst: Optional[float] = None
    igst: Optional[float] = None
    tax_total: Optional[float] = None
    round_off: Optional[float] = None
    grand_total: Optional[float] = None
    currency: str = "INR"


class Invoice(BaseModel):
    document_is_invoice: bool = Field(
        True, description="False if the document is not an invoice or bill"
    )
    invoice_number: Optional[str] = None
    invoice_date: Optional[str] = Field(None, description="YYYY-MM-DD")
    due_date: Optional[str] = Field(None, description="YYYY-MM-DD")
    vendor: Party = Party()
    buyer: Party = Party()
    line_items: list[LineItem] = []
    totals: Totals = Totals()
    notes: Optional[str] = Field(
        None, description="Anything unreadable, ambiguous or worth flagging"
    )


class APIResponse(BaseModel):
    status: str
    processing_time_ms: float
    data: Invoice


# --------------------------------------------------------------------------
# Build a clean JSON schema for Claude's tool call
# --------------------------------------------------------------------------


def _inline_refs(node: Any, defs: dict) -> Any:
    """Resolve $ref pointers and drop noisy keys so the schema is simple."""
    if isinstance(node, dict):
        if "$ref" in node:
            return _inline_refs(defs[node["$ref"].split("/")[-1]], defs)
        return {
            k: _inline_refs(v, defs)
            for k, v in node.items()
            if k not in ("title", "default", "$defs")
        }
    if isinstance(node, list):
        return [_inline_refs(i, defs) for i in node]
    return node


_raw_schema = Invoice.model_json_schema()
TOOL_SCHEMA = _inline_refs(_raw_schema, _raw_schema.get("$defs", {}))

SYSTEM_PROMPT = """You extract data from Indian business invoices, bills and receipts.
Rules:
- Copy values exactly as printed. Never invent or guess a value. Use null when a field is missing or unreadable.
- Numbers must be plain numbers (no commas, no currency symbols).
- Dates must be YYYY-MM-DD. Indian dates are usually DD/MM/YYYY.
- Vendor is the seller who issued the invoice. Buyer is the customer (Bill To).
- GSTIN is 15 characters. Only fill it if you can read it clearly.
- CGST, SGST, IGST are amounts in rupees, not percentages.
- If text is in Hindi or another language, still extract the fields.
- If the document is not an invoice or bill, set document_is_invoice to false.
- Put any doubts (blurry digits, cut-off edges, mismatched totals) in notes."""

# --------------------------------------------------------------------------
# Security: only RapidAPI may call this server
# --------------------------------------------------------------------------


def verify_rapidapi_proxy(
    x_rapidapi_proxy_secret: Optional[str] = Header(default=None),
) -> bool:
    if not PROXY_SECRET:
        raise HTTPException(500, "Server is missing RAPIDAPI_PROXY_SECRET.")
    given = (x_rapidapi_proxy_secret or "").encode()
    if not hmac.compare_digest(given, PROXY_SECRET.encode()):
        raise HTTPException(403, "Direct access prohibited. Use the RapidAPI gateway.")
    return True


# --------------------------------------------------------------------------
# File preparation
# --------------------------------------------------------------------------


def build_document_block(content: bytes) -> dict:
    """Turn the uploaded bytes into a content block Claude can read."""
    if content[:5] == b"%PDF-":
        if len(content) > 10 * 1024 * 1024:
            raise HTTPException(413, "PDF is too large. Keep it under 10 MB.")
        return {
            "type": "document",
            "source": {
                "type": "base64",
                "media_type": "application/pdf",
                "data": base64.standard_b64encode(content).decode(),
            },
        }
    try:
        img = Image.open(io.BytesIO(content))
        img = ImageOps.exif_transpose(img)  # fix sideways phone photos
        img = img.convert("RGB")
        img.thumbnail((MAX_IMAGE_EDGE, MAX_IMAGE_EDGE))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=88)
    except (UnidentifiedImageError, OSError):
        raise HTTPException(
            422, "Could not read the file. Send a JPG, PNG, WEBP or PDF invoice."
        )
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/jpeg",
            "data": base64.standard_b64encode(buf.getvalue()).decode(),
        },
    }


_client: Optional[AsyncAnthropic] = None


def get_client() -> AsyncAnthropic:
    global _client
    if _client is None:
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise HTTPException(500, "Server is missing ANTHROPIC_API_KEY.")
        _client = AsyncAnthropic(timeout=90.0, max_retries=2)
    return _client


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------


@app.get("/health", include_in_schema=False)
async def health() -> dict:
    """Open endpoint for Render health checks and waking the server."""
    return {"ok": True}


@app.post(
    "/v1/parse/invoice",
    response_model=APIResponse,
    dependencies=[Depends(verify_rapidapi_proxy)],
    summary="Extract an invoice",
)
async def parse_invoice(file: UploadFile = File(...)) -> APIResponse:
    started = time.time()

    content = await file.read()
    if not content:
        raise HTTPException(422, "The uploaded file is empty.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File is too large. Keep it under 15 MB.")

    block = build_document_block(content)

    try:
        response = await get_client().messages.create(
            model=MODEL,
            max_tokens=4000,
            system=SYSTEM_PROMPT,
            tools=[
                {
                    "name": "record_invoice",
                    "description": "Record the data extracted from the invoice.",
                    "input_schema": TOOL_SCHEMA,
                }
            ],
            tool_choice={"type": "tool", "name": "record_invoice"},
            messages=[
                {
                    "role": "user",
                    "content": [
                        block,
                        {"type": "text", "text": "Extract this invoice."},
                    ],
                }
            ],
        )
    except APIError as e:
        log.error("Anthropic API error: %s", e)
        raise HTTPException(502, "The AI service failed. Please try again.")

    tool_input = next(
        (b.input for b in response.content if b.type == "tool_use"), None
    )
    if tool_input is None:
        raise HTTPException(502, "The AI returned no data. Please try again.")

    try:
        invoice = Invoice.model_validate(tool_input)
    except ValidationError as e:
        log.error("Bad AI output: %s", e)
        raise HTTPException(502, "The AI returned malformed data. Please try again.")

    if not invoice.document_is_invoice:
        raise HTTPException(422, "This does not look like an invoice or bill.")

    return APIResponse(
        status="success",
        processing_time_ms=round((time.time() - started) * 1000, 1),
        data=invoice,
    )
