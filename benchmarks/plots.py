"""Plots for benchmarking LLM inference."""
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "plots"


def plot_throughput_vs_batch_size(batch_sizes: list[int], throughputs: list[float], strategy: str, model_name: str, dtype: str) -> Path:
    """Plot throughput vs batch size for a given generation strategy.

    Args:
        batch_sizes (list[int]): List of batch sizes used in the benchmark.
        throughputs (list[float]): List of throughput values corresponding to the batch sizes.
        strategy (str): Name of the generation strategy that was benchmarked.
        model_name (str): Name of the model used in the benchmark.
        dtype (str): Data type used in the benchmark.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(batch_sizes, throughputs, marker='o')
    plt.title(f'Throughput vs Batch Size for {strategy.capitalize()} Generation\nModel: {model_name}, Dtype: {dtype}')
    plt.xlabel('Batch Size')
    plt.ylabel('Throughput (tokens/sec)')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.xticks(batch_sizes, labels=batch_sizes)
    plt.tight_layout()
    path = RESULTS_DIR / f'throughput_vs_batch_size_{strategy}_{model_name.replace("/", "_")}_{dtype}.png'
    plt.savefig(path)
    return path