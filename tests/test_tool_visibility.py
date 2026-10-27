from __future__ import annotations

import asyncio
from types import SimpleNamespace

from pydantic import BaseModel

from nitrostack import ExecutionContext, injectable, module, tool
from nitrostack.core.app import McpApplication
from nitrostack.core.context import AuthContext
from nitrostack.core.visibility import SessionVisibilityStore, VisibilityTransform
from nitrostack.runtime.request_ctx import RequestContext, request_ctx


class EmptyInput(BaseModel):
    pass


@injectable()
class VisibilityTools:
    @tool("public_tool", "public", EmptyInput)
    async def public(self, input: EmptyInput, context: ExecutionContext):
        return "public"

    @tool("admin_tool", "admin", EmptyInput, visibility="hidden", default_visible=False)
    async def admin(self, input: EmptyInput, context: ExecutionContext):
        return "admin"

    @tool("unlock", "unlock", EmptyInput)
    async def unlock(self, input: EmptyInput, context: ExecutionContext):
        context.enable_tools(["admin_tool", "not_registered"])
        return "ok"

    @tool("lock", "lock", EmptyInput)
    async def lock(self, input: EmptyInput, context: ExecutionContext):
        context.disable_tools(["admin_tool"])
        return "ok"


@module(name="visibility", providers=[VisibilityTools])
class VisibilityModule:
    pass


def _app():
    return McpApplication(VisibilityModule)


def _request(session_id="session-a"):
    return RequestContext(session=SimpleNamespace(id=session_id), request_id="1")


def test_session_store_lru_eviction():
    now = [0.0]
    store = SessionVisibilityStore(ttl_seconds=10, max_entries=2, clock=lambda: now[0])
    store.update(("a", "u"), ["tool"], True)
    store.update(("b", "u"), ["tool"], True)
    assert store.is_visible(("a", "u"), "tool", False)
    store.update(("c", "u"), ["tool"], True)
    assert not store.is_visible(("b", "u"), "tool", False)


def test_session_store_ttl_expiry_uses_fake_clock():
    now = [0.0]
    store = SessionVisibilityStore(ttl_seconds=10, max_entries=2, clock=lambda: now[0])
    store.update(("session", "subject"), ["tool"], True)
    now[0] = 11
    assert not store.is_visible(("session", "subject"), "tool", False)


def test_session_key_requires_session_and_includes_verified_subject():
    assert VisibilityTransform.session_key(None, "user-a") is None
    assert VisibilityTransform.session_key("client-session", "user-a") != (
        VisibilityTransform.session_key("client-session", "user-b")
    )


def test_tools_list_and_call_enforce_visibility_and_isolate_sessions():
    async def run():
        app = _app()
        list_handler = app.mcp_server.request_handlers[
            __import__("mcp.types", fromlist=["ListToolsRequest"]).ListToolsRequest
        ]
        request = _request()
        token = request_ctx.set(request)
        try:
            listed = await list_handler(SimpleNamespace(params=None))
            assert "admin_tool" not in [item.name for item in listed.root.tools]
            denied = await app._call_tool("admin_tool", {})
            assert denied.is_error

            enabled = await app._call_tool("unlock", {})
            assert not enabled.is_error
            listed = await list_handler(SimpleNamespace(params=None))
            assert "admin_tool" in [item.name for item in listed.root.tools]

            entry = app._tools["admin_tool"]
            other_session = app._visibility_session_key(_request("session-b"))
            assert not app.visibility_transform.is_visible(entry, other_session)
            other_subject = ("session-a", "verified-user-b")
            assert not app.visibility_transform.is_visible(entry, other_subject)

            disabled = await app._call_tool("lock", {})
            assert not disabled.is_error
            listed = await list_handler(SimpleNamespace(params=None))
            assert "admin_tool" not in [item.name for item in listed.root.tools]
        finally:
            request_ctx.reset(token)

    asyncio.run(run())


def test_sessionless_mutation_is_noop_but_static_default_applies():
    app = _app()
    context = ExecutionContext(request_id="r")
    app._configure_visibility_context(context, None)
    context.enable_tools(["admin_tool"])
    assert not app.visibility_transform.is_visible(app._tools["admin_tool"], None)
    assert app.visibility_transform.is_visible(app._tools["public_tool"], None)


def test_http_session_id_is_transport_owned_and_keyed_with_verified_subject(monkeypatch):
    """An arbitrary client session header cannot choose another session's state."""
    from starlette.testclient import TestClient

    from nitrostack import DIContainer
    from nitrostack.auth.jwt import JWTModule, JWTService
    from nitrostack.core.app import ServerConfig
    from nitrostack.transports.http import build_http_app

    async def build():
        from nitrostack.core.app import McpApplicationFactory, mcp_app

        @mcp_app(module=VisibilityModule, server=ServerConfig(name="visibility-http", protocol_era="legacy"))
        class HttpVisibilityApp:
            pass

        return await McpApplicationFactory.create(HttpVisibilityApp)

    monkeypatch.setenv("JWT_SECRET", "visibility-test-secret")
    DIContainer.reset()
    JWTModule.for_root(secret_env_var="JWT_SECRET")
    app = asyncio.run(build())
    http_app = build_http_app(
        app, protocol_era="legacy", wire_mode="sessionful", enable_cors=True
    )
    token = DIContainer.get_instance().resolve(JWTService).create_token({"sub": "owner"})
    auth_headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }
    init_body = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "visibility-test", "version": "1"},
        },
    }
    with TestClient(http_app) as client:
        session_a = client.post("/mcp", headers=auth_headers, json=init_body)
        session_b = client.post("/mcp", headers=auth_headers, json=init_body)
        assert session_a.status_code == session_b.status_code == 200
        id_a = session_a.headers["mcp-session-id"]
        id_b = session_b.headers["mcp-session-id"]
        assert id_a != id_b

        # A client-supplied random ID is rejected; it cannot select stored state.
        arbitrary = client.post(
            "/mcp",
            headers={**auth_headers, "mcp-session-id": "client-chosen"},
            json={"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
        )
        assert arbitrary.status_code == 404

        # Verified subject is included in the server-owned session key.
        key_a = app._visibility_session_key(
            SimpleNamespace(session=SimpleNamespace(id=id_a), request=SimpleNamespace(headers=auth_headers))
        )
        key_b = app._visibility_session_key(
            SimpleNamespace(session=SimpleNamespace(id=id_b), request=SimpleNamespace(headers=auth_headers))
        )
        assert key_a == (id_a, "owner")
        assert key_b == (id_b, "owner")
        assert key_a != key_b

        # A state change in one live session does not affect the other.
        app.visibility_transform.update(key_a, ["admin_tool"], True)
        assert app.visibility_transform.is_visible(app._tools["admin_tool"], key_a)
        assert not app.visibility_transform.is_visible(app._tools["admin_tool"], key_b)
    DIContainer.reset()
