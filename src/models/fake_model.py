class FakeLLM:
    def generate_response(self, query: str, max_tokens: int = 50) -> str:
        # Simulate a response by returning the query repeated
        return f"Fake response to '{query}' with max tokens {max_tokens}"