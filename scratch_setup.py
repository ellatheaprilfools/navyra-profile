import json
from navyra_profile import analyse

prompts = [json.loads(l)["prompt"] for l in open("examples/sample_traffic.jsonl")]
r = analyse(prompts)

print(f"loaded {len(prompts)} prompts, analysed, r is ready")