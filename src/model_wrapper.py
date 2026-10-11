import torch

class ModelWrapper:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        eos = model.generation_config.eos_token_id
        self.eos_ids = set(eos) if isinstance(eos, (list, tuple)) else {eos}
        self.pad_id = model.generation_config.pad_token_id if model.generation_config.pad_token_id is not None else self.eos_ids.pop()

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
            if next_id.item() in self.eos_ids:
                break
            input_ids = next_id.unsqueeze(1)
        return generated
    
    def generate_greedy_batched(self, prompts : list[str], max_output_tokens: int = 50):
        enc = self.tokenizer(prompts, return_tensors="pt",padding=True,padding_side = "left").to(self.model.device)
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        position_ids = (torch.cumsum(attention_mask, dim=1)-1)
        position_ids = position_ids.masked_fill(attention_mask == 0, 0)
        position_ids = position_ids.to(self.model.device)
        past_key_values = None
        steps = []
        eos_ids_tensor = torch.tensor(list(self.eos_ids)).to(self.model.device)
        end = torch.zeros(input_ids.shape[0], dtype=torch.long, device=input_ids.device)
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        for _ in range(max_output_tokens):
            next_logits, past_key_values = self.step(input_ids,attention_mask, past_key_values=past_key_values, position_ids=position_ids)
            next_ids = next_logits.argmax(dim=1)
            next_ids = next_ids.masked_fill(finished, self.pad_id)
            is_eos = torch.isin(next_ids, eos_ids_tensor)
            next_position = position_ids[:, -1:] + 1
            position_ids = next_position
            steps.append(next_ids)
            attention_mask = torch.cat([attention_mask, torch.ones_like(attention_mask[:,:1])], dim=1)
            end+=~finished
            finished|=is_eos
            if finished.all():
                break
            input_ids = next_ids.unsqueeze(1)
        generated = torch.stack(steps, dim=1).tolist()
        end = end.tolist()
        for i, row in enumerate(generated):
            generated[i] = row[:end[i]]

        return generated

def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32).to(device).eval()
    wrapper = ModelWrapper(model, tok)
    prompts = ["Hello", "What is the capital of France?"]
    out = wrapper.generate_greedy_batched(prompts, 20)
    print([o == wrapper.generate_greedy(p, 20) for p, o in zip(prompts, out)])
if __name__ == "__main__":
    main()