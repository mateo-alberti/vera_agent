from __future__ import annotations

import logging
from typing import Any, Dict

import requests
from langchain_core.tools import StructuredTool

from app.core.config import Settings
from app.prompts import STOCK_PRICE_TOOL_DESCRIPTION


def get_stock_price_from_api(
    symbol: str,
) -> Dict[str, Any]:
    """Fetch the latest stock price quote from Alpha Vantage."""
    settings = Settings()
    logger = logging.getLogger("vera.stock_price_tool")

    if not settings.alphavantage_api_key:
        result = {"status": "error", "message": "Alpha Vantage API key is not configured"}
        logger.info("tool_end tool=%s output=%s", "get_stock_price", result)
        return result

    logger.info("tool_start tool=%s symbol=%s", "get_stock_price", symbol)

    params = {
        "function": "GLOBAL_QUOTE",
        "symbol": symbol,
        "apikey": settings.alphavantage_api_key,
    }

    response = requests.get(settings.alphavantage_base_url, params=params, timeout=15)
    response.raise_for_status()
    payload = response.json()

    if "Error Message" in payload:
        result = {"status": "error", "message": payload["Error Message"]}
        logger.info("tool_end tool=%s output=%s", "get_stock_price", result)
        return result

    if "Note" in payload:
        result = {"status": "error", "message": payload["Note"]}
        logger.info("tool_end tool=%s output=%s", "get_stock_price", result)
        return result

    quote = payload.get("Global Quote", {})
    if not quote:
        result = {"status": "error", "message": "No quote returned from Alpha Vantage"}
        logger.info("tool_end tool=%s output=%s", "get_stock_price", result)
        return result

    result = {
        "status": "ok",
        "symbol": quote.get("01. symbol"),
        "open": quote.get("02. open"),
        "high": quote.get("03. high"),
        "low": quote.get("04. low"),
        "price": quote.get("05. price"),
        "volume": quote.get("06. volume"),
        "latest_trading_day": quote.get("07. latest trading day"),
        "previous_close": quote.get("08. previous close"),
        "change": quote.get("09. change"),
        "change_percent": quote.get("10. change percent"),
    }
    logger.info("tool_end tool=%s output=%s", "get_stock_price", result)
    return result


def get_stock_price_tool() -> StructuredTool:
    return StructuredTool.from_function(
        func=get_stock_price_from_api,
        name="get_stock_price",
        description=STOCK_PRICE_TOOL_DESCRIPTION,
    )
