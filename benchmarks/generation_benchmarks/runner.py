from benchmarks.generation_benchmarks.sequential import SequentialGenerationBenchmark
from benchmarks.generation_benchmarks.static_batch import StaticBatchGenerationBenchmark
from llm_inference_server.model_wrapper import ModelWrapper

class GenerationBenchmarkRunner:
    def __init__(self, wrapper: ModelWrapper):
        self.wrapper = wrapper
        self.sequential_benchmark = SequentialGenerationBenchmark(wrapper)
        self.static_batch_benchmark = StaticBatchGenerationBenchmark(wrapper)

    def run_sequential_benchmark(self, prompts: list[str], max_output_tokens: int = 50, warmup: bool = True, iterations: int = 10, ignore_eos: bool = False):
        return self.sequential_benchmark.run(prompts, max_output_tokens, warmup, iterations, ignore_eos=ignore_eos)

    def run_static_batch_benchmark(self, prompts: list[str], max_output_tokens: int = 50, warmup: bool = True, iterations: int = 10, ignore_eos: bool = False, batch_size: int = 8):
        return self.static_batch_benchmark.run(prompts, max_output_tokens, warmup, iterations, ignore_eos=ignore_eos, batch_size=batch_size)