#!/usr/bin/env python3
"""Fire N traced calls at a local Ollama model, then print the collector's
latency percentiles for them. No third-party dependencies.

    python examples/simulate_calls.py --model qwen2.5:0.5b --n 50 --tag version=v1
"""
import argparse
import json
import random
import sys
import time
import urllib.request

from llm_obs import Tracer

PROMPTS = [
    "Summarize the plot of a heist movie in two sentences.",
    "List three uses for a paperclip.",
    "Explain what a hash table is to a ten year old.",
    "Write a haiku about a slow database.",
    "What is the capital of Australia? Answer briefly.",
]


def ollama_generate(base, model, prompt, num_predict, num_ctx):
    body = json.dumps({
        "model": model, "prompt": prompt, "stream": False,
        "keep_alive": -1,
        "options": {"num_predict": num_predict, "num_ctx": num_ctx},
    }).encode()
    req = urllib.request.Request(base + "/api/generate", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read().decode())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5:0.5b")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--name", default="summarize_endpoint")
    ap.add_argument("--tag", action="append", default=[], help="key=value, repeatable")
    ap.add_argument("--collector", default="http://localhost:8765")
    ap.add_argument("--ollama", default="http://localhost:11434")
    ap.add_argument("--num-predict", type=int, default=48)
    ap.add_argument("--num-ctx", type=int, default=2048)
    ap.add_argument("--long-prompt-every", type=int, default=0,
                    help="every k-th call gets a very long prompt (to exercise context-pressure diagnosis)")
    args = ap.parse_args()

    tags = dict(kv.split("=", 1) for kv in args.tag)
    tracer = Tracer(url=args.collector, default_tags=tags)

    print(f"sending {args.n} calls to {args.model} via {args.ollama} ...")
    t0 = time.time()
    for i in range(args.n):
        prompt = random.choice(PROMPTS)
        if args.long_prompt_every and i % args.long_prompt_every == args.long_prompt_every - 1:
            prompt = ("The quick brown fox jumps over the lazy dog. " * 400) + "\nSummarize the above."
        with tracer.trace(args.name, model=args.model, runtime="ollama") as span:
            resp = ollama_generate(args.ollama, args.model, prompt, args.num_predict, args.num_ctx)
            span.record_ollama_response(resp)
        sys.stdout.write(f"\r  {i + 1}/{args.n}  last={span.completion_tokens} tok "
                         f"{span.tokens_per_sec:.1f} tok/s  {(span.end - span.start) * 1000:.0f} ms   ")
        sys.stdout.flush()
    print(f"\ndone in {time.time() - t0:.1f}s; client stats: {tracer.stats}")
    tracer.flush(10)
    print(f"after flush: {tracer.stats}")

    url = f"{args.collector}/traces?name={args.name}&window=1h"
    with urllib.request.urlopen(url, timeout=5) as resp:
        data = json.loads(resp.read().decode())
    for g in data["groups"]:
        print(f"\n{g['name']}: n={g['count']} errors={g['errors']} models={g['models']}")
        print(f"  latency  p50={g['p50_ms']:.0f}ms  p95={g['p95_ms']:.0f}ms  p99={g['p99_ms']:.0f}ms  mean={g['mean_ms']:.0f}ms")
        print(f"  tokens/s p50={g['tokens_per_sec_p50']:.1f}  mean={g['tokens_per_sec_mean']:.1f}")


if __name__ == "__main__":
    main()
