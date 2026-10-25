"""Stats helper functions for benchmarks."""
import statistics


def percentile(values: list[float], p: float) -> float:
    """Compute the p-th percentile of a list using interpolation.

    Args:
        values (list[float]): Values to compute the percentile over.
        p (float): Percentile to compute, between 0 and 100.

    Returns:
        float: The interpolated percentile value.
    """
    if not values:
        raise ValueError("values must not be empty")
    ordered = sorted(values)
    rank = (len(ordered) - 1) * p / 100
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def compute_stats(times: list[float], token_counts: list[int]) -> dict[str, float]:
    """Compute statistics from benchmark results.

    Args:
        times (list[float]): List of time taken for each iteration.
        token_counts (list[int]): List of total output token counts for each iteration.

    Returns:
        dict[str, float]: Dictionary containing the time statistics (average, median,
        min, max, standard deviation) and the median throughput in tokens per second.
    """
    if not times:
        raise ValueError("times must not be empty")
    if len(times) != len(token_counts):
        raise ValueError("times and token_counts must have the same length")

    throughputs = [tokens / time for tokens, time in zip(token_counts, times) if time > 0]
    stdev_time = statistics.stdev(times) if len(times) > 1 else 0.0

    return {
        "iterations": len(times),
        "average_time": statistics.mean(times),
        "median_time": statistics.median(times),
        "min_time": min(times),
        "max_time": max(times),
        "stdev_time": stdev_time,
        "total_tokens": statistics.median(token_counts),
        "throughput": statistics.median(throughputs) if throughputs else 0.0,
    }
