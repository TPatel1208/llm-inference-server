"""Workload helpers for benchmarking LLM inference."""
import random


def shuffle_prompts(prompts: list[str], num_prompts: int, seed: int = 42) -> list[str]:
    shuffled = random.Random(seed).sample(prompts, min(num_prompts, len(prompts)))
    return [shuffled[i % len(shuffled)] for i in range(num_prompts)]
