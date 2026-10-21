"""
Musicion 离线评测套件
一键运行：消融实验 + 延迟拆解 + Intent Routing 准确率

用法：
  python scripts/eval_offline.py --full
  python scripts/eval_offline.py --ablation
  python scripts/eval_offline.py --latency
  python scripts/eval_offline.py --intent

前置条件：Neo4j 运行中，歌曲已入库
"""
import argparse, asyncio, json, os, sys, time
from collections import defaultdict
from pathlib import Path
from typing import Any
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def normalize(s: str) -> str:
    import re, unicodedata
    t = unicodedata.normalize("NFKC", s).casefold().strip()
    return re.sub(r"[,，、/\\\s()（）【】\[\]《》'\"`]+", "", t)

def recall_at_k(hits: set, gt: list, k: int) -> float:
    if not gt: return 1.0
    gt_set = {normalize(t) for t in gt}
    return len(hits & gt_set) / len(gt_set)

def mrr(ranked: list, gt: list) -> float:
    gt_set = {normalize(t) for t in gt}
    for i, t in enumerate(ranked, 1):
        if normalize(t) in gt_set: return 1.0 / i
    return 0.0

def ndcg_at_k(ranked: list, gt: list, k: int) -> float:
    gt_set = {normalize(t) for t in gt}
    dcg = sum(1.0 / np.log2(i+2) for i, t in enumerate(ranked[:k]) if normalize(t) in gt_set)
    ideal = min(len(gt_set), k)
    idcg = sum(1.0 / np.log2(i+2) for i in range(ideal))
    return dcg / idcg if idcg > 0 else 0.0

def load_queries():
    path = Path(__file__).with_name("eval_queries.json")
    return json.loads(path.read_text(encoding="utf-8"))["queries"]

def run_ablation(queries, k_values=None):
    """消融实验：graph-only vs vector-only vs hybrid"""
    if k_values is None: k_values = [5, 10, 20, 50]
    from retrieval.recall_sources import graph_candidate_recall
    from tools.semantic_search import semantic_search
    from retrieval.retrieval_fusion import weighted_rrf

    results = {}
    for v in ["graph_only", "vector_only", "hybrid"]:
        results[v] = {f"R@{k}": [] for k in k_values}
        results[v]["MRR"] = []
        results[v]["NDCG@10"] = []

    valid = [q for q in queries if q.get("ground_truth")]
    for qi, q in enumerate(valid):
        gt = q["ground_truth"]
        tags = q.get("tags", {})
        hc = {"artist_entities": tags.get("artist_entities", []),
              "song_entities": tags.get("song_entities", []),
              "language": tags.get("language"),
              "region": tags.get("region")}
        hints = {"genres": tags.get("genres", []),
                 "mood": (tags.get("moods") or [None])[0],
                 "scenario": (tags.get("scenarios") or [None])[0]}
        limit = max(k_values)

        # Graph
        try:
            gr = graph_candidate_recall(hc, hints, limit=limit)
            gi = json.loads(gr) if gr else []
            gt_set = {normalize(item.get("title","")) for item in gi}
            for k in k_values: results["graph_only"][f"R@{k}"].append(recall_at_k(gt_set, gt, k))
            results["graph_only"]["MRR"].append(mrr([i.get("title","") for i in gi], gt))
            results["graph_only"]["NDCG@10"].append(ndcg_at_k([i.get("title","") for i in gi], gt, 10))
        except Exception as e:
            for k in k_values: results["graph_only"][f"R@{k}"].append(0)
            results["graph_only"]["MRR"].append(0); results["graph_only"]["NDCG@10"].append(0)

        # Vector
        try:
            vr = semantic_search.invoke({"query": q["query"], "limit": limit})
            vi = json.loads(vr) if vr else []
            vt_set = {normalize(item.get("title","")) for item in vi}
            for k in k_values: results["vector_only"][f"R@{k}"].append(recall_at_k(vt_set, gt, k))
            results["vector_only"]["MRR"].append(mrr([i.get("title","") for i in vi], gt))
            results["vector_only"]["NDCG@10"].append(ndcg_at_k([i.get("title","") for i in vi], gt, 10))
        except Exception as e:
            for k in k_values: results["vector_only"][f"R@{k}"].append(0)
            results["vector_only"]["MRR"].append(0); results["vector_only"]["NDCG@10"].append(0)

        # Hybrid RRF
        try:
            from retrieval.hybrid_retrieval import MusicHybridRetrieval as MHR
            gp = MHR._parse_engine_results(gr, "graph") if gr else []
            vp = MHR._parse_engine_results(vr, "dense") if vr else []
            fused = weighted_rrf({"graph": gp, "dense": vp}, {"graph": 1.0, "dense": 1.0})
            ht_set = {normalize(item.get("song",{}).get("title","")) for item in fused}
            hr = [item.get("song",{}).get("title","") for item in fused]
            for k in k_values: results["hybrid"][f"R@{k}"].append(recall_at_k(ht_set, gt, k))
            results["hybrid"]["MRR"].append(mrr(hr, gt))
            results["hybrid"]["NDCG@10"].append(ndcg_at_k(hr, gt, 10))
        except Exception as e:
            for k in k_values: results["hybrid"][f"R@{k}"].append(0)
            results["hybrid"]["MRR"].append(0); results["hybrid"]["NDCG@10"].append(0)

        print(f"  [{qi+1}/{len(valid)}] {q['query'][:50]}")

    return {v: {m: round(np.mean(vals), 4) for m, vals in metrics.items() if vals}
            for v, metrics in results.items()}

def run_latency_breakdown(queries):
    """端到端延迟拆解"""
    from retrieval.hybrid_retrieval import MusicHybridRetrieval
    ret = MusicHybridRetrieval()
    stage_times = defaultdict(list)

    async def _m(q):
        t0 = time.perf_counter()
        plan = {"hard_constraints": q.get("tags", {}),
                "soft_intent": {}, "hints": q.get("tags", {}),
                "_intent_type": q.get("intent", "hybrid_search")}
        r = await ret.retrieve(q["query"], limit=15, precomputed_plan=plan)
        ts = r.metadata.get("timings", {})
        ts["total_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return ts

    for qi, q in enumerate(queries[:5]):
        try:
            ts = asyncio.run(_m(q))
            for s, ms in ts.items(): stage_times[s].append(float(ms))
            print(f"  [{qi+1}] total={ts.get('total_ms','?')}ms")
        except Exception as e:
            print(f"  [{qi+1}] failed: {e}")

    return {s: {"mean_ms": round(np.mean(v), 1), "median_ms": round(np.median(v), 1),
                "p99_ms": round(np.percentile(v, 99), 1)}
            for s, v in sorted(stage_times.items()) if v}

INTENT_CASES = [
    ("推荐周杰伦的歌", "graph_search", False),
    ("适合跑步听的快节奏电音", "hybrid_search", False),
    ("和晴天风格很像的歌", "vector_search", False),
    ("最近Billboard新歌", "web_search", True),
    ("来点我的收藏", "favorites", False),
    ("下雨天听的安静民谣", "hybrid_search", False),
    ("粤语老歌推荐", "graph_search", False),
    ("类似Coldplay风格的", "vector_search", False),
    ("心情不好随便来几首", "hybrid_search", False),
    ("周杰伦和林俊杰合唱的歌", "graph_search", False),
    ("适合学习听的纯音乐", "hybrid_search", False),
    ("抖音最火的BGM", "web_search", True),
    ("帮我找几首K-Pop", "graph_search", False),
    ("古典钢琴曲", "hybrid_search", False),
    ("和星际穿越配乐差不多的", "vector_search", False),
]

def run_intent_test():
    """Intent Routing 准确率测试"""
    from agent.intent.planner import UnifiedPlanner
    try:
        planner = UnifiedPlanner()
    except Exception:
        return {"note": "Planner unavailable (no LLM API)", "accuracy": None}

    correct, errors = 0, []
    for query, exp_intent, exp_web in INTENT_CASES:
        try:
            plan = planner.plan(query)
            if plan.intent_type == exp_intent: correct += 1
            else: errors.append({"query": query, "expected": exp_intent, "actual": plan.intent_type})
        except Exception as e:
            errors.append({"query": query, "error": str(e)})
    return {"accuracy": round(correct/len(INTENT_CASES), 3), "correct": correct,
            "total": len(INTENT_CASES), "errors": errors[:5]}

def main():
    p = argparse.ArgumentParser(description="Musicion Offline Eval")
    p.add_argument("--full", action="store_true")
    p.add_argument("--ablation", action="store_true")
    p.add_argument("--latency", action="store_true")
    p.add_argument("--intent", action="store_true")
    p.add_argument("--output", type=str, default=None)
    args = p.parse_args()
    if not any([args.full, args.ablation, args.latency, args.intent]):
        p.print_help(); return

    run_all = args.full
    queries = load_queries()
    output = {"eval_queries_count": len(queries)}

    if run_all or args.ablation:
        print("\n=== Part 1: Ablation Study ===")
        ab = run_ablation(queries)
        output["ablation"] = ab
        for v, m in ab.items():
            print(f"  {v}: { {k: round(vv,4) for k,vv in m.items()} }")

    if run_all or args.latency:
        print("\n=== Part 2: Latency Breakdown ===")
        lat = run_latency_breakdown(queries)
        output["latency"] = lat
        for s, st in sorted(lat.items()):
            print(f"  {s}: mean={st['mean_ms']}ms p99={st['p99_ms']}ms")

    if run_all or args.intent:
        print("\n=== Part 3: Intent Routing ===")
        ir = run_intent_test()
        output["intent_routing"] = ir
        print(f"  Accuracy: {ir.get('accuracy', 'N/A')}")
        if ir.get("errors"):
            for e in ir["errors"][:3]: print(f"  {e}")

    if args.output:
        Path(args.output).write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nSaved: {args.output}")

    print("\nDone.")

if __name__ == "__main__":
    main()