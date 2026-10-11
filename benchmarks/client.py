"""Client for benchmarking LLM inference."""
import httpx

class HTTPClient:
    def __init__(self):
        self.client = httpx.AsyncClient()

    async def post(self, url, json):
        return self.client.stream("POST", url, json=json)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.client.aclose()
