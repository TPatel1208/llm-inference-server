import pytest
import torch
from src.model_wrapper import ModelWrapper
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

@pytest.fixture(scope="module")
def model_and_tokenizer():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32).to(device).eval()
    return model, tok
 
@pytest.fixture(scope="module")
def wrapper(model_and_tokenizer):
    model, tok = model_and_tokenizer
    return ModelWrapper(model, tok)



def hf_generate(model, tokenizer, prompt, n):
    enc = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        hf = model.generate(
        **enc, do_sample=False, max_new_tokens=n,
        repetition_penalty=1.0, temperature=None, top_p=None, top_k=None,)
        return hf[0, enc.input_ids.shape[1]:].tolist()


class TestModelWrapper:
    @pytest.mark.parametrize("prompt,n", [
        ("Hello", 1),
        ("Hello", 10),
        ("What is the capital of France?", 20),
        ("Explain how a computer processor executes instructions.", 30),
        (
        "Computers have changed significantly over the past several decades. "
        "Early computers occupied entire rooms, while modern devices can fit "
        "in a pocket. Advances in processor design, memory, and software have "
        "made computing faster and more accessible.",
        50,
        ),
        ])
    def test_matches_greedy_output(self, wrapper,prompt, n):
        mine = wrapper.generate_greedy(prompt, max_output_tokens=n)
        model, tokenizer = wrapper.model, wrapper.tokenizer
        hf_new = hf_generate(model, tokenizer, prompt, n)
        assert mine == hf_new

    def test_eos_behavior(self, wrapper):
        n = 100
        model, tokenizer = wrapper.model, wrapper.tokenizer
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": "Reply with only the word: hi"}],
            tokenize=False,
            add_generation_prompt=True,
        )
        mine = wrapper.generate_greedy(prompt, max_output_tokens=n)
        hf_new = hf_generate(model, tokenizer, prompt, n)
        assert mine == hf_new
        assert len(mine) < n

    def test_batched_generate_greedy(self, wrapper):
        model, tokenizer = wrapper.model, wrapper.tokenizer
        short_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": "Reply with only the word: hi"}],
            tokenize=False,
            add_generation_prompt=True,
        )
        long_prompt =  ("Computers have changed significantly over the past several decades. "
                        "Early computers occupied entire rooms, while modern devices can fit "
                        "in a pocket. Advances in processor design, memory, and software have "
                        "made computing faster and more accessible.")
        prompts = ["Hello", "What is the capital of France?", short_prompt, long_prompt]
        n = 50
        out = wrapper.generate_greedy_batched(prompts, max_output_tokens=n)
        for p, o in zip(prompts, out):
            assert o == wrapper.generate_greedy(p, max_output_tokens=n)
        assert len(out[2]) < n
        assert out[2][-1] in wrapper.eos_ids