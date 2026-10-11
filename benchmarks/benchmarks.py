"""Metrics for benchmarking LLM inference."""
import time
from typing import Any
import asyncio
from benchmarks.client import HTTPClient
class Benchmark():
    def __init__ (self, concurrency: int, num_requests: int, url: str ):
        self.concurrency = concurrency
        self.num_requests = num_requests
        self.url = url
    
    async def benchmark_request(self, prompt: str,client: HTTPClient, semaphore: asyncio.Semaphore)->dict[str, Any]:
        async with semaphore:
            start_time = time.perf_counter()

            async with client.post(
                self.url,
                {
                    "query": prompt,
                    "max_tokens": 50
                }) as response:
            
                first_output_chunk_time = None
                output = []
                async for chunk in response.aiter_text():
                    if first_output_chunk_time is None:
                        first_output_chunk_time = time.perf_counter()
                    output.append(chunk)

                end_time = time.perf_counter()

            return {
                "ttft": first_output_chunk_time - start_time, #time from sending the request until the first streamed output arrives 
                "latency": end_time - start_time,
                "generation": end_time - first_output_chunk_time,
                "output": "".join(output),
            }

    async def run(self):
        semaphore = asyncio.Semaphore(self.concurrency)
        async with HTTPClient() as client:
            tasks = [
                self.benchmark_request(prompt="hello", client=client,semaphore=semaphore)
                for _ in range(self.num_requests)
            ]
            results = await asyncio.gather(*tasks)


        return results       