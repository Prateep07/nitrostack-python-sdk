import asyncio
import json
import warnings

import pytest

from nitrostack import injectable, module
from nitrostack.core.additional_decorators import HealthCheckRegistry, health_check
from nitrostack.testing import NitroTestingModule


@pytest.fixture(autouse=True)
def clean_registry():
    saved = (dict(HealthCheckRegistry._checks), dict(HealthCheckRegistry._bound_checks))
    HealthCheckRegistry._checks.clear()
    HealthCheckRegistry._bound_checks.clear()
    yield
    HealthCheckRegistry._checks.clear()
    HealthCheckRegistry._bound_checks.clear()
    HealthCheckRegistry._checks.update(saved[0])
    HealthCheckRegistry._bound_checks.update(saved[1])


def _app_module():
    @injectable()
    class Svc:
        @health_check("async_ok")
        async def async_ok(self):
            await asyncio.sleep(0)
            return True

        @health_check("sync_ok")
        def sync_ok(self):
            return True

        @health_check("async_boom")
        async def async_boom(self):
            raise RuntimeError("db down")

    @module(name="m", providers=[Svc])
    class M:
        pass

    return M


def _read_status():
    async def run():
        harness = await NitroTestingModule.create(_app_module())
        raw = await harness.read_resource("health://status")
        return json.loads(raw) if isinstance(raw, str) else raw

    return asyncio.run(run())


def test_async_and_sync_checks_both_report_healthy():
    status = _read_status()
    assert "event loop" not in str(status["async_ok"])
    assert status["async_ok"] == status["sync_ok"]


def test_raising_check_reports_error_without_affecting_others():
    status = _read_status()
    assert "db down" in status["async_boom"]
    assert "error" not in str(status["async_ok"])
    assert "error" not in str(status["sync_ok"])


def test_no_never_awaited_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        _read_status()