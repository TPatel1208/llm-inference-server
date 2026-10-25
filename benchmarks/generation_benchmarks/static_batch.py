import torch
import time 

from llm_inference_server.model_wrapper import ModelWrapper

class StaticBatchGenerationBenchmark:
    def __init__(self, wrapper: ModelWrapper):
        self.wrapper = wrapper

    def _warmup(self, prompts: list[str], max_output_tokens: int = 10, ignore_eos: bool = False):
        for _ in range(3):
            self.wrapper.generate_greedy_batched(prompts, max_output_tokens, ignore_eos=ignore_eos)

    def run(self, prompts: list[str], max_output_tokens: int = 50, warmup: bool = True, iterations: int = 10,batch_size: int = 8,ignore_eos:bool = False):
        results = {
            "outputs": [],
            "times": [],
            "token_counts": [],
        }
        if warmup:
            self._warmup(prompts[:batch_size], 10, ignore_eos=True)

        for _ in range(iterations):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_time = time.perf_counter()
            batch_outputs = []
            for i in range(0, len(prompts), batch_size):
                batch_prompts = prompts[i:i + batch_size]
                batch_output = self.wrapper.generate_greedy_batched(batch_prompts, max_output_tokens, ignore_eos=ignore_eos)
                batch_outputs.extend(batch_output)
            output = batch_outputs
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.perf_counter()
            results["outputs"].append(output)
            results["times"].append(end_time - start_time)
            results["token_counts"].append(sum([len(o) for o in output]))
        return results
