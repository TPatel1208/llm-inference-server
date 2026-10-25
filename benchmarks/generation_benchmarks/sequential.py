import time 
import torch
from llm_inference_server.model_wrapper import ModelWrapper

class SequentialGenerationBenchmark:
    def __init__(self, wrapper: ModelWrapper):
        self.wrapper = wrapper

    def _warmup(self, prompt: str, max_output_tokens: int = 10, ignore_eos: bool = False):
        for _ in range(3):
            self.wrapper.generate_greedy(prompt, max_output_tokens, ignore_eos=ignore_eos)

    def run(self, prompts: list[str], max_output_tokens: int = 50, warmup: bool = True,iterations: int = 10, ignore_eos: bool = False):
        results = {
            "outputs": [],
            "times": [],
            "token_counts": [],
        }
        if warmup:
            self._warmup(prompts[0], 10, ignore_eos=True)

        for _ in range(iterations):
            output_batch = []
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_time = time.perf_counter()
            # Run the generation for each prompt sequentially
            for prompt in prompts:
                output_batch.append(self.wrapper.generate_greedy(prompt, max_output_tokens, ignore_eos=ignore_eos))
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.perf_counter()
            results["outputs"].append(output_batch)
            results["times"].append(end_time - start_time)
            results["token_counts"].append(sum(len(output) for output in output_batch))
        return results