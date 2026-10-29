"""Command-line interface for benchmarking LLM inference."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import typer

from llm_inference_server.model_wrapper import ModelWrapper
from benchmarks.generation_benchmarks.runner import GenerationBenchmarkRunner
from benchmarks.prompts import PROMPTS
from benchmarks.results import save_result, load_results, select_records
from benchmarks.stats import compute_stats
from benchmarks.workload import shuffle_prompts
from benchmarks.plots import plot_throughput_vs_batch_size

app = typer.Typer()
def _load_model_and_tokenizer(model_name: str = "Qwen/Qwen2.5-0.5B-Instruct", dtype: str = "float16"):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        dtype= getattr(torch, dtype)
        )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        padding_side="left"
        )
    return model, tokenizer

def _create_benchmark_runner(model_name: str = "Qwen/Qwen2.5-0.5B-Instruct", dtype: str = "float16"):
    model, tokenizer = _load_model_and_tokenizer(model_name, dtype)
    wrapper = ModelWrapper(model, tokenizer)
    return GenerationBenchmarkRunner(wrapper)

def _parse_batch_sizes(batch_sizes: str) -> list[int]:
    return [int(size) for size in batch_sizes.split(",")]

def _report(strategy: str, runner: GenerationBenchmarkRunner, results: dict, **settings) -> None:
    """Print the benchmark statistics and save the result to the results folder.

    Args:
        strategy (str): Name of the generation strategy that was benchmarked.
        runner (GenerationBenchmarkRunner): Runner used for the benchmark, to read the model details.
        results (dict): Raw results returned by the benchmark with the times and token counts.
        **settings: Benchmark settings, for example batch_size and max_output_tokens.
    """
    model = runner.wrapper.model
    settings = {
        "model": model.name_or_path,
        "dtype": str(model.dtype),
        "device": str(model.device),
        **settings,
    }
    stats = compute_stats(results["times"], results["token_counts"])

    print(f"\n{strategy.capitalize()} Generation Benchmark Results:")
    for key, value in settings.items():
        print(f"{key}: {value}")
    for key, value in stats.items():
        print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")

    path = save_result({
        "strategy": strategy,
        "settings": settings,
        "stats": stats,
        "times": results["times"],
        "token_counts": results["token_counts"],
    })
    print(f"Saved to {path}")

@app.command()
def run_seq(
    max_output_tokens: int = typer.Option(50, help="Maximum number of output tokens to generate."),
    warmup: bool = typer.Option(True, help="Whether to perform a warmup before benchmarking."),
    iterations: int = typer.Option(10, help="Number of iterations to run the benchmark."),
    seed: int = typer.Option(42, help="Random seed for shuffling prompts."),
    ignore_eos: bool = typer.Option(False, help="Ignore eos, will generate until max_output_tokens is reached"),
    number_of_prompts: int = typer.Option(64, help="Number of prompts to use for benchmarking."),
    model_name: str = typer.Option("Qwen/Qwen2.5-0.5B-Instruct", "--model", help="Hugging Face model to benchmark."),
    dtype: str = typer.Option("float16", help="Model dtype, for example float16, bfloat16 or float32."),
):
    runner = _create_benchmark_runner(model_name, dtype)
    prompts = shuffle_prompts(PROMPTS, number_of_prompts, seed)
    results = runner.run_sequential_benchmark(prompts, max_output_tokens, warmup, iterations,ignore_eos=ignore_eos)
    _report(
        "sequential", runner, results,
        batch_size=1,
        max_output_tokens=max_output_tokens,
        number_of_prompts=len(prompts),
        iterations=iterations,
        ignore_eos=ignore_eos,
        seed=seed,
    )

@app.command()
def run_static(
    max_output_tokens: int = typer.Option(50, help="Maximum number of output tokens to generate."),
    warmup: bool = typer.Option(True, help="Whether to perform a warmup before benchmarking."),
    iterations: int = typer.Option(10, help="Number of iterations to run the benchmark."),
    seed: int = typer.Option(42, help="Random seed for shuffling prompts."),
    batch_size: int = typer.Option(8, help="Batch size for static batch generation."),
    ignore_eos: bool = typer.Option(False, help="Ignore eos, will generate until max_output_tokens is reached"),
    number_of_prompts: int = typer.Option(64, help="Number of prompts to use for benchmarking."),
    sweep: bool = typer.Option(False, help="Run every size in --batch-sizes instead of the single --batch-size."),
    batch_sizes: str = typer.Option("1,2,4,8,16,32,64", help="Comma separated batch sizes to run when --sweep is set."),
    model_name: str = typer.Option("Qwen/Qwen2.5-0.5B-Instruct", "--model", help="Hugging Face model to benchmark."),
    dtype: str = typer.Option("float16", help="Model dtype, for example float16, bfloat16 or float32."),
):
    runner = _create_benchmark_runner(model_name, dtype)
    prompts = shuffle_prompts(PROMPTS, number_of_prompts, seed)
    sizes =_parse_batch_sizes(batch_sizes) if sweep else [batch_size]
    for size in sizes:
        print(f"\nRunning static batch benchmark with batch size: {size} ...")
        if len(prompts) % size != 0:
            print(f"Warning: {len(prompts)} prompts do not divide evenly into batches of {size}, the last batch is smaller.")
        try:
            results = runner.run_static_batch_benchmark(prompts, max_output_tokens, warmup, iterations, ignore_eos=ignore_eos, batch_size=size)
        except torch.cuda.OutOfMemoryError:
            print(f"Batch size {size} ran out of GPU memory, skipping.")
            torch.cuda.empty_cache()
            continue
        _report(
            "static", runner, results,
            batch_size=size,
            max_output_tokens=max_output_tokens,
            number_of_prompts=len(prompts),
            iterations=iterations,
            ignore_eos=ignore_eos,
            seed=seed,
        )
    records = load_results()
    selected_records = select_records(records, "static", model=model_name, dtype=f'torch.{dtype}', max_output_tokens=max_output_tokens, ignore_eos=ignore_eos, device=str(runner.wrapper.model.device), number_of_prompts=len(prompts))
    if selected_records:
        batch_sizes = [record["settings"]["batch_size"] for record in selected_records]
        throughputs = [record["stats"]["throughput"] for record in selected_records]
        plot_path = plot_throughput_vs_batch_size(batch_sizes, throughputs, "static", model_name, dtype)
        print(f"Saved throughput vs batch size plot to {plot_path}")

if __name__ == "__main__":
    app()
