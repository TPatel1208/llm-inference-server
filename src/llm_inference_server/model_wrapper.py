import torch

class ModelWrapper:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        eos = model.generation_config.eos_token_id
        self.eos_ids = set(eos) if isinstance(eos, (list, tuple)) else {eos}
        self.pad_id = model.generation_config.pad_token_id if model.generation_config.pad_token_id is not None else next(iter(self.eos_ids))

    @torch.inference_mode()
    def step(self, input_ids, attention_mask, position_ids=None, past_key_values=None):
        out = self.model(input_ids=input_ids,
                         attention_mask=attention_mask,
                         position_ids=position_ids,
                           past_key_values=past_key_values, 
                           use_cache=True)
        return out.logits[:,-1,:], out.past_key_values

    def generate_greedy(self, prompt : str, max_output_tokens: int = 50, ignore_eos: bool = False):
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
            if next_id.item() in self.eos_ids and not ignore_eos:
                break
            input_ids = next_id.unsqueeze(1)
        return generated
    
    def generate_greedy_batched(self, prompts : list[str], max_output_tokens: int = 50, ignore_eos: bool = False):
        # Tokenize the prompts and prepare input tensors
        enc = self.tokenizer(prompts, return_tensors="pt",padding=True,padding_side = "left").to(self.model.device)

        # Extract input_ids and attention_mask from the encoded prompts
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]

        #Derive position_ids from attention_mask
        position_ids = (torch.cumsum(attention_mask, dim=1)-1)
        position_ids = position_ids.masked_fill(attention_mask == 0, 0)
        position_ids = position_ids.to(self.model.device)

        past_key_values = None
        steps = [] #Store the generated tokens for each prompt in a list

        # Convert the eos_ids set to a tensor for comparison
        eos_ids_tensor = torch.tensor(list(self.eos_ids)).to(self.model.device)

        # Initialize tensors to track the end positions and finished status for each prompt
        if not ignore_eos:
            end = torch.zeros(input_ids.shape[0], dtype=torch.long, device=input_ids.device)
            finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)

        for _ in range(max_output_tokens):
            # Call the step function to get the next logits and past key values
            next_logits, past_key_values = self.step(input_ids,attention_mask, past_key_values=past_key_values, position_ids=position_ids)
            next_ids = next_logits.argmax(dim=1)
            if not ignore_eos:
                next_ids = next_ids.masked_fill(finished, self.pad_id) #Use finished mask to replace next_ids with pad_id for finished prompts
                is_eos = torch.isin(next_ids, eos_ids_tensor)
            next_position = position_ids[:, -1:] + 1
            position_ids = next_position
            steps.append(next_ids)
            attention_mask = torch.cat([attention_mask, torch.ones_like(attention_mask[:,:1])], dim=1)
            if not ignore_eos:
                end+=~finished
                finished|=is_eos
                if finished.all():
                    break
            input_ids = next_ids.unsqueeze(1)
        generated = torch.stack(steps, dim=1).tolist()
        
        if not ignore_eos:
            end = end.tolist()
            for i, row in enumerate(generated):
                generated[i] = row[:end[i]]

        return generated