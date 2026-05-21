"""
pricing.py  --  AWS Price List lookup for S3 Vectors and embedding rates.

Fetches current rates once at startup and caches them for the process lifetime.
Falls back to config.py rates if the Price List API is unavailable or returns
unrecognised data.  The fake-backend path (DEMO_FAKE=1) never calls this module.

Verified 2026-05-20:
  - AmazonS3Vectors service exists in Price List but returns 0 items (too new).
    S3 Vectors storage and query rates fall back to config.py values.

  - Claude 4.x pricing is NOT in the AmazonBedrock service code (which only
    contains Claude 2.x / 3.x).  It IS in AmazonBedrockFoundationModels.
    The PRICING dict in config.py is sourced from that service.
    https://aws.amazon.com/bedrock/pricing/
    NOTE: these are Bedrock on-demand prices, not Anthropic API prices.

  - AmazonBedrock Price List DOES contain Titan Embed V2 pricing:
    USW2-TitanEmbeddingV2-Text-input-tokens = $0.00002 / 1K tokens on-demand
    ($0.02 / 1M tokens).  We fetch this live.

  - AmazonBedrock also has Nova Pro pricing live ($0.0008/$0.0032 per 1K =
    $0.80/$3.20 per 1M) for the on-demand tier (matches config.py).
"""

from __future__ import annotations

import json
import logging

log = logging.getLogger(__name__)

# Hard-coded floor values used only if BOTH Price List and config.py are absent.
_HARD_DEFAULTS: dict[str, float] = {
    "s3v_storage_usd_per_gb_month": 0.05,
    "s3v_query_usd_per_1k": 0.40,
    "embed_usd_per_1m_tokens": 0.02,
    "s3_standard_usd_per_gb_month": 0.023,  # S3 standard, first 50 TB, us-west-2
}

_CACHE: dict[str, float] | None = None


def fetch_rates(region: str) -> dict[str, float]:
    """Return current S3 Vectors / embedding rates, fetched once and cached.

    Titan Embed V2 rate comes from AmazonBedrock Price List (works).
    S3 Vectors storage/query rates fall back to config.py (Price List empty).
    Model inference prices (PRICING dict) come from config.py; Claude 4.x is
    not yet in the Price List API — source them from config.py directly.
    """
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    # Start from config fallback, then overlay any live prices we can fetch.
    base = _from_config()
    live_embed = _fetch_bedrock_embed_rate(region)
    if live_embed:
        base.update(live_embed)
        log.info("Bedrock embed rate fetched from Price List: %s", live_embed)
    live_s3 = _fetch_s3_standard_rate(region)
    if live_s3:
        base.update(live_s3)
        log.info("S3 standard rate fetched from Price List: %s", live_s3)
    _CACHE = base
    return _CACHE


def reset_cache() -> None:
    """Clear the module-level cache (for testing only)."""
    global _CACHE
    _CACHE = None


# ── Price List: Titan Embed V2 ────────────────────────────────────────────────


def _region_prefix(region: str) -> str:
    """Map AWS region to the Price List usagetype prefix."""
    _MAP = {
        "us-east-1": "USE1",
        "us-east-2": "USE2",
        "us-west-1": "USW1",
        "us-west-2": "USW2",
        "eu-west-1": "EU",
        "eu-west-2": "EUW2",
        "eu-west-3": "EUW3",
        "eu-central-1": "EUC1",
        "ap-southeast-1": "APS1",
        "ap-southeast-2": "APS2",
        "ap-northeast-1": "APN1",
    }
    return _MAP.get(region, "USE1")


def _fetch_bedrock_embed_rate(region: str) -> dict[str, float] | None:
    """Fetch Titan Embed V2 on-demand rate from the AmazonBedrock Price List.

    Note: Claude 4.x model prices are in AmazonBedrockFoundationModels, not
    AmazonBedrock.  This function only fetches the embedding rate used for
    the KB ingestion cost calculation.
    """
    try:
        import boto3

        client = boto3.client("pricing", region_name="us-east-1")
        prefix = _region_prefix(region)
        usage_type = f"{prefix}-TitanEmbeddingV2-Text-input-tokens"

        resp = client.get_products(
            ServiceCode="AmazonBedrock",
            FormatVersion="aws_v1",
            Filters=[{"Type": "TERM_MATCH", "Field": "usagetype", "Value": usage_type}],
        )
        items = resp.get("PriceList", [])
        if not items:
            return None

        item = json.loads(items[0])
        for term in item.get("terms", {}).get("OnDemand", {}).values():
            for dim in term.get("priceDimensions", {}).values():
                usd_per_1k_str = dim.get("pricePerUnit", {}).get("USD", "")
                usd_per_1k = float(usd_per_1k_str)
                if usd_per_1k > 0:
                    # Price List gives USD per 1K tokens; we store per 1M tokens.
                    return {"embed_usd_per_1m_tokens": round(usd_per_1k * 1000, 6)}
    except Exception as exc:
        log.info("Bedrock Price List unavailable (%s) — using config fallback", exc)
    return None


def _fetch_s3_standard_rate(region: str) -> dict[str, float] | None:
    """Fetch S3 standard storage rate (first 50 TB tier) from the AmazonS3 Price List."""
    try:
        import boto3

        client = boto3.client("pricing", region_name="us-east-1")
        resp = client.get_products(
            ServiceCode="AmazonS3",
            FormatVersion="aws_v1",
            Filters=[
                {"Type": "TERM_MATCH", "Field": "regionCode", "Value": region},
                {"Type": "TERM_MATCH", "Field": "storageClass", "Value": "General Purpose"},
                {"Type": "TERM_MATCH", "Field": "volumeType", "Value": "Standard"},
            ],
        )
        import json

        for item_str in resp.get("PriceList", []):
            item = json.loads(item_str)
            attrs = item.get("product", {}).get("attributes", {})
            if "TimedStorage" not in attrs.get("usagetype", ""):
                continue
            for term in item.get("terms", {}).get("OnDemand", {}).values():
                for dim in term.get("priceDimensions", {}).values():
                    desc = dim.get("description", "")
                    # First 50 TB tier is the one to use for a small corpus
                    if "first 50 TB" not in desc:
                        continue
                    usd = float(dim.get("pricePerUnit", {}).get("USD", "0") or "0")
                    if usd > 0:
                        return {"s3_standard_usd_per_gb_month": usd}
    except Exception as exc:
        log.info("S3 Price List unavailable (%s) — using config fallback", exc)
    return None


# ── config fallback ───────────────────────────────────────────────────────────


def _from_config() -> dict[str, float]:
    """Load fallback rates from config.py, then hard defaults for missing keys."""
    try:
        import config  # type: ignore[import]

        return {
            "s3v_storage_usd_per_gb_month": getattr(
                config,
                "S3V_STORAGE_USD_PER_GB_MONTH",
                _HARD_DEFAULTS["s3v_storage_usd_per_gb_month"],
            ),
            "s3v_query_usd_per_1k": getattr(
                config,
                "S3V_QUERY_USD_PER_1K",
                _HARD_DEFAULTS["s3v_query_usd_per_1k"],
            ),
            "embed_usd_per_1m_tokens": getattr(
                config,
                "EMBED_USD_PER_1M_TOKENS",
                _HARD_DEFAULTS["embed_usd_per_1m_tokens"],
            ),
            "s3_standard_usd_per_gb_month": getattr(
                config,
                "S3_STANDARD_USD_PER_GB_MONTH",
                _HARD_DEFAULTS["s3_standard_usd_per_gb_month"],
            ),
        }
    except ImportError:
        log.info("config.py not found — using hard-coded default rates")
        return dict(_HARD_DEFAULTS)
