import asyncio
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
import torch
from threading import Thread

class LLM_Model:
    def __init__(self, model_name: str, device_map: str = "auto", dtype: str = torch.float16):

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            device_map=device_map, 
            torch_dtype= dtype
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            padding_side="left"
            )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.eval()

    async def generate_response(self, query:str, max_tokens: int = 50): 
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt = True,
            skip_special_tokens = True
            )

        model_inputs = self.tokenizer(
            query, 
            return_tensors="pt"
            ).to(self.model.device)

        with torch.inference_mode():
            generate_kwargs = dict(
                **model_inputs,
                streamer=streamer,
                max_new_tokens=max_tokens,
                            )
            t = Thread(target=self.model.generate, kwargs=generate_kwargs)
            t.start()
            for new_text in streamer:
                yield new_text
                await asyncio.sleep(0)

class LLM_Pool:
    def __init__(self,model_name: str, num_replicas: int = 2):
        self.num_replicas = num_replicas
        self.model_name = model_name
        self.num_replicas = []
        for i in range(self.num_replicas):
            self.replicas[i] = LLM_Model(model_name=model_name)

    def create_pool(self):
        pass