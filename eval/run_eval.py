"""Run: python -m eval.run_eval   (from project root). Add ~20 pairs to eval/pairs.json."""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv
load_dotenv()
from matcher import analyze

pairs = json.loads((pathlib.Path(__file__).parent / "pairs.json").read_text())
passed = 0
print(f"{'id':<14}{'score':>6}  expected   result")
for p in pairs:
    score = analyze(p["resume"], p["jd"])["score"]
    ok = p["expected_min"] <= score <= p["expected_max"]
    passed += ok
    print(f"{p['id']:<14}{score:>6}  {p['expected_min']}-{p['expected_max']:<7} {'PASS' if ok else 'FAIL'}")
print(f"\n{passed}/{len(pairs)} within expected range ({100 * passed // len(pairs)}%)")
