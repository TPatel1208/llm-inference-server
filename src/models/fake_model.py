import asyncio 
from collections.abc import AsyncGenerator

class FakeLLM:
    async def generate_response(self, query: str, max_tokens: int = 50) -> AsyncGenerator:
        # Simulate a response by returning the query repeated
        stream_response =  (f"Fake response to '{query}' with max tokens {max_tokens}").split()
        for word in stream_response:
            yield word + " "
            await asyncio.sleep(0.5)

        