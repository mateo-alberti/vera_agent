from app.services.tools import stock_price_tool


def test_stock_price_returns_error_without_key(monkeypatch):
    class DummySettings:
        alphavantage_api_key = None
        alphavantage_base_url = "https://example.test"

    monkeypatch.setattr(stock_price_tool, "Settings", DummySettings)

    def _fail(*args, **kwargs):
        raise AssertionError("requests.get should not be called when key is missing")

    monkeypatch.setattr(stock_price_tool.requests, "get", _fail)

    result = stock_price_tool.get_stock_price_from_api("AAPL")

    assert result["status"] == "error"
    assert "API key" in result["message"]


def test_stock_price_returns_error_from_payload(monkeypatch):
    class DummySettings:
        alphavantage_api_key = "key"
        alphavantage_base_url = "https://example.test"

    class DummyResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"Error Message": "bad request"}

    monkeypatch.setattr(stock_price_tool, "Settings", DummySettings)
    monkeypatch.setattr(stock_price_tool.requests, "get", lambda *args, **kwargs: DummyResponse())

    result = stock_price_tool.get_stock_price_from_api("AAPL")

    assert result["status"] == "error"
    assert result["message"] == "bad request"


def test_stock_price_success(monkeypatch):
    class DummySettings:
        alphavantage_api_key = "key"
        alphavantage_base_url = "https://example.test"

    class DummyResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "Global Quote": {
                    "01. symbol": "AAPL",
                    "02. open": "1",
                    "03. high": "2",
                    "04. low": "0.5",
                    "05. price": "1.5",
                    "06. volume": "100",
                    "07. latest trading day": "2024-01-01",
                    "08. previous close": "1.4",
                    "09. change": "0.1",
                    "10. change percent": "7%",
                }
            }

    monkeypatch.setattr(stock_price_tool, "Settings", DummySettings)
    monkeypatch.setattr(stock_price_tool.requests, "get", lambda *args, **kwargs: DummyResponse())

    result = stock_price_tool.get_stock_price_from_api("AAPL")

    assert result["status"] == "ok"
    assert result["symbol"] == "AAPL"
    assert result["price"] == "1.5"
