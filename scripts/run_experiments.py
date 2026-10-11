import sys
import json
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag_chain import RAGChain

def run_experiments():
    eval_file = PROJECT_ROOT / "data" / "evaluation_set.json"
    if not eval_file.exists():
        print(f"File not found: {eval_file}")
        return

    with open(eval_file, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    in_scope_cases = [t for t in test_cases if t["type"] != "out_of_scope"]
    total = len(in_scope_cases)

    k_values = [2, 4, 8]
    results_table = []

    print("=" * 75)
    print(" [EXPERIMENTS] RUNNING RAG RETRIEVAL EXPERIMENTS (COMPARING TOP-K VALUES)")
    print("=" * 75)

    rag = RAGChain()

    for k in k_values:
        hits = 0
        mrr_sum = 0.0
        latencies = []

        for item in in_scope_cases:
            t0 = time.perf_counter()
            res = rag.query(item["question"], top_k=k)
            elapsed = time.perf_counter() - t0
            latencies.append(elapsed)

            sources = [s.get("source", s.get("pdf_name", "")) for s in res.get("citations", [])]
            expected = item["expected_doc"]

            # Check hit and rank
            found = False
            for rank, s in enumerate(sources, start=1):
                if expected and expected in s:
                    hits += 1
                    mrr_sum += (1.0 / rank)
                    found = True
                    break

        hit_rate = (hits / total) * 100
        mrr = mrr_sum / total
        avg_latency = sum(latencies) / len(latencies)

        results_table.append({
            "top_k": k,
            "hit_rate": hit_rate,
            "mrr": mrr,
            "avg_latency_ms": avg_latency * 1000
        })

        print(f"Top-K = {k:2d} | Hit Rate: {hit_rate:5.1f}% | MRR: {mrr:.4f} | Avg Latency: {avg_latency*1000:6.1f} ms")

    print("\n" + "=" * 75)
    print(" [SUMMARY] EXPERIMENT COMPARISON SUMMARY TABLE")
    print("=" * 75)
    print(f" {'Top-K':<8} | {'Hit Rate (%)':<15} | {'MRR':<10} | {'Avg Latency (ms)':<18}")
    print("-" * 65)
    for row in results_table:
        print(f" {row['top_k']:<8} | {row['hit_rate']:<15.2f} | {row['mrr']:<10.4f} | {row['avg_latency_ms']:<18.2f}")
    print("=" * 75)

    # Save results to experiments/results.json
    out_dir = PROJECT_ROOT / "experiments"
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results_table, f, indent=2)
    print(f"Saved results to: {out_dir / 'results.json'}")

if __name__ == "__main__":
    run_experiments()
