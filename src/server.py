"""Fast API Server"""
from fastapi import FastAPI

from model import LLM_Model
from models.chat_request import Generate_Request
test_inf = LLM_Model("nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16")
app = FastAPI()


@app.post("/generate")
def generate(request: GenerateRequest):
    return test_inf.generate_response(query=request.query, max_tokens=request.max_tokens)