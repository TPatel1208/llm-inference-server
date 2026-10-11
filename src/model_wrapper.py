import torch

class ModelWrapper:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @torch.inference_mode()
    def step(self, input_ids, attention_mask, position_ids=None, past_key_values=None):
        out = self.model(input_ids=input_ids,
                         attention_mask=attention_mask,
                         position_ids=position_ids,
                           past_key_values=past_key_values, 
                           use_cache=True)
        return out.logits[:,-1,:], out.past_key_values

    def generate_greedy(self, prompt : str, max_output_tokens: int = 50):
        enc = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0).to(self.model.device)
        past_key_values = None
        generated = []
        for _ in range(max_output_tokens):
            next_logits, past_key_values = self.step(input_ids,attention_mask, past_key_values=past_key_values, position_ids=position_ids)
            next_id = next_logits.argmax(dim=1)
            next_position = position_ids[:, -1:] + 1
            position_ids = next_position
            generated.append(next_id.item())
            attention_mask = torch.cat([attention_mask, torch.ones_like(attention_mask[:,:1])], dim=1)
            if next_id.item() == self.model.generation_config.eos_token_id:
                break
            input_ids = next_id.unsqueeze(1)
        return generated
