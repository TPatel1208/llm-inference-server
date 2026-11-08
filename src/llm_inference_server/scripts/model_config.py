import json
from huggingface_hub import hf_hub_download
from pathlib import Path
for name in ["config.json", "generation_config.json", "tokenizer_config.json"]:
    path = hf_hub_download("Qwen/Qwen2.5-0.5B-Instruct", name)
    conf = json.load(open(path))
    print(name)
    print(json.dumps(conf,indent=4))
