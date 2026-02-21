from app.services.tools import weather_tool


def test_weather_no_response(monkeypatch):
    class DummySettings:
        openmeteo_base_url = "https://example.test"

    class DummyClient:
        def __init__(self, session=None):
            self.session = session

        def weather_api(self, base_url, params=None):
            return []

    monkeypatch.setattr(weather_tool, "Settings", DummySettings)
    monkeypatch.setattr(weather_tool.openmeteo_requests, "Client", DummyClient)

    result = weather_tool.get_current_weather_from_api(1.0, 2.0)

    assert result["status"] == "error"
    assert "No response" in result["message"]


def test_weather_success(monkeypatch):
    class DummySettings:
        openmeteo_base_url = "https://example.test"

    class DummyVar:
        def __init__(self, value):
            self._value = value

        def Value(self):
            return self._value

    class DummyCurrent:
        def Variables(self, idx):
            return DummyVar([20.0, 5.0][idx])

    class DummyResponse:
        def Latitude(self):
            return 10.0

        def Longitude(self):
            return 11.0

        def Timezone(self):
            return "UTC"

        def Current(self):
            return DummyCurrent()

    class DummyClient:
        def __init__(self, session=None):
            self.session = session

        def weather_api(self, base_url, params=None):
            return [DummyResponse()]

    monkeypatch.setattr(weather_tool, "Settings", DummySettings)
    monkeypatch.setattr(weather_tool.openmeteo_requests, "Client", DummyClient)

    result = weather_tool.get_current_weather_from_api(1.0, 2.0)

    assert result["status"] == "ok"
    assert result["latitude"] == 10.0
    assert result["temperature_2m"] == 20.0
    assert result["wind_speed_10m"] == 5.0
