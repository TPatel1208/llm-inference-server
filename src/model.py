from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

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

        self.model.eval()

    def generate_response(self, query:str, max_tokens: int = 50): 

        model_inputs = self.tokenizer(
            query, 
            return_tensors="pt"
            ).to(self.model.device)

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **model_inputs, 
                max_new_tokens=max_tokens
                )
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

