from pydantic import Basemodel, Field

class GenerateRequest(Basemodel):
    query: str
    max_tokens: int = Field(default = 50,ge = 1, le = 512)