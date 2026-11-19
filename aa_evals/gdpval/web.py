"""Stirrup's web_fetch through the cluster's HTTP proxy.

Stirrup's fetch client ignores proxy env vars (`trust_env=False`). It resolves the host, checks every address is public,
then connects to the pinned IP with the hostname in Host/SNI. Cluster nodes have no direct internet access, and the
proxy refuses CONNECT to a raw IP (403), so as shipped every web_fetch fails ("All connection attempts failed"; the M3.2
Qwen3.8 smoke: 0 of 9 fetches succeeded). Models fell back to curl in the sandbox.

ProxiedWebToolProvider keeps Stirrup's tools, checks and limits unchanged and swaps only the fetch client's
transport: requests go through the proxy by hostname (taken back from the Host header Stirrup sets). The public-IP check
still runs first; the proxy resolves the name itself. web_search (Brave, when a key is set) already honours the proxy
env.
"""
from __future__ import annotations

from contextlib import AsyncExitStack

import httpx
from anyio import CancelScope

from stirrup.tools.web import WebToolProvider

PROXY = 'http://proxy.internal:8080'


class _ByHostname(httpx.AsyncBaseTransport):
    """Send the pinned-IP request by its logical hostname through the proxy."""

    def __init__(self, proxy: str):
        self._inner = httpx.AsyncHTTPTransport(proxy=proxy, trust_env=False)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        host = request.headers.get('host')
        if host:
            h, _, port = host.rpartition(':') if host.count(':') == 1 else (host, '', '')
            url = request.url.copy_with(host=h or host, port=int(port) if port.isdigit() else request.url.port)
            request = httpx.Request(request.method, url, headers=request.headers, stream=request.stream,
                                    extensions={k: v for k, v in request.extensions.items() if k != 'sni_hostname'})
        return await self._inner.handle_async_request(request)

    async def aclose(self) -> None:
        await self._inner.aclose()


class ProxiedWebToolProvider(WebToolProvider):
    def __init__(self, *, proxy: str = PROXY, **kw):
        super().__init__(**kw)
        self._proxy = proxy

    async def __aenter__(self):
        stack = AsyncExitStack()
        await stack.__aenter__()
        try:
            self._fetch_client = await stack.enter_async_context(httpx.AsyncClient(
                timeout=self._timeout, follow_redirects=False, trust_env=False,
                transport=_ByHostname(self._proxy)))
            if self._brave_api_key:
                self._search_client = await stack.enter_async_context(
                    httpx.AsyncClient(timeout=self._timeout, follow_redirects=True))
            tools = self.get_tools()
        except BaseException:
            self._fetch_client = None
            self._search_client = None
            with CancelScope(shield=True):
                await stack.aclose()
            raise
        self._client_stack = stack
        return tools
