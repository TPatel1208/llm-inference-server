"""Fast API Server"""
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
from dotenv import load_dotenv
import asyncio
import os

from model import LLM_Model
from models.chat_request import GenerateRequest
from models.fake_model import FakeLLM

load_dotenv()  # Load environment variables from .env file

@asynccontextmanager
async def lifespan(app: FastAPI):
    if os.getenv("use_fake_model", "true").lower() == "true":
        app.state.model = FakeLLM()
    else:
        model_name = os.getenv("model_name", "ibm-granite/granite-4.1-8b")
        app.state.model = LLM_Model(model_name=model_name)
    yield 
    
    app.state.model = None
app = FastAPI(lifespan=lifespan)


@app.post("/generate")
async def generate(request: GenerateRequest):
    return StreamingResponse(app.state.model.generate_response(
        request.query, request.max_tokens),
        media_type="text/plain")