"""Save and load benchmark results."""
import json
from datetime import datetime
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"


def save_result(record: dict) -> Path:
    """Save a benchmark result to a timestamped JSON file in the results folder.

    Args:
        record (dict): Benchmark record with the strategy, settings, stats and raw numbers.

    Returns:
        Path: Path of the JSON file that was written.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_size = record["settings"].get("batch_size", 1)
    path = RESULTS_DIR / f"{record['strategy']}_bs{batch_size}_{timestamp}.json"
    path.write_text(json.dumps(record, indent=2))
    return path


def load_results() -> list[dict]:
    """Load every saved benchmark result from the results folder.

    Returns:
        list[dict]: Saved benchmark records, oldest first.
    """
    return [json.loads(path.read_text()) for path in sorted(RESULTS_DIR.glob("*.json"))]


def select_records(records: list[dict], strategy: str, **settings) -> list[dict]:
    """Select benchmark records that match the given strategy and settings.

    Args:
        records (list[dict]): List of benchmark records to filter.
        strategy (str): Generation strategy to filter by.
        **settings: Settings to filter by, for example batch_size and max_output_tokens.
    """
    selected = []
    for record in records:
        if record["strategy"] != strategy:
            continue
        if all(record["settings"].get(k) == v for k, v in settings.items()):
            selected.append(record)
    return sorted(selected, key=lambda record: record["settings"]["batch_size"])