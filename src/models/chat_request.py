from pydantic import BaseModel, Field

class GenerateRequest(BaseModel):
    query: str
    max_tokens: int = Field(default = 50,ge = 1, le = 512)