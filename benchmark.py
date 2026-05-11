#!/usr/bin/env python3
"""
Benchmark llama-server OpenAI-compatible API
Parameterized via environment variables:
  LLAMA_BENCH_URL      - base URL (default: http://localhost:8080)
  LLAMA_BENCH_MODEL    - model name (default: Qwen3.6-35B-MoE)
  LLAMA_BENCH_BACKEND  - backend tag for results (default: vulkan)
  LLAMA_IMAGE          - docker image tag for results (default: unknown)
Uses only stdlib (no requests dependency).
"""
import os, time, json, sys, statistics, urllib.request

BASE = os.environ.get("LLAMA_BENCH_URL", "http://localhost:8080")
MODEL = os.environ.get("LLAMA_BENCH_MODEL", "Qwen3.6-35B-MoE")
BACKEND = os.environ.get("LLAMA_BENCH_BACKEND", "vulkan")
IMAGE = os.environ.get("LLAMA_IMAGE", "unknown")

def api_call(path, payload=None, method="GET", timeout=300, expect_json=True):
    url = f"{BASE}{path}"
    headers = {"Content-Type": "application/json"}
    data = json.dumps(payload).encode("utf-8") if payload else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
        if expect_json:
            return json.loads(raw)
        return raw

def stream_chat(messages, max_tokens=256):
    url = f"{BASE}/v1/chat/completions"
    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    
    tokens = 0
    first = True
    ttft = None
    t0 = time.perf_counter()
    
    with urllib.request.urlopen(req, timeout=300) as resp:
        for line in resp:
            line = line.decode("utf-8").strip()
            if not line or not line.startswith("data: "):
                continue
            line = line[6:]
            if line == "[DONE]":
                break
            try:
                chunk = json.loads(line)
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                has_token = bool(delta.get("content") or delta.get("reasoning_content"))
                if has_token:
                    tokens += 1
                    if first:
                        ttft = time.perf_counter() - t0
                        first = False
            except json.JSONDecodeError:
                continue
    
    t1 = time.perf_counter()
    dur = t1 - t0
    gen_dur = dur - ttft if ttft else dur
    return {
        "ttft_ms": round(ttft * 1000, 1) if ttft else None,
        "tokens": tokens,
        "total_s": round(dur, 2),
        "gen_tok_s": round(tokens / gen_dur, 2) if gen_dur > 0 else 0,
    }

def non_stream_chat(messages, max_tokens=256):
    t0 = time.perf_counter()
    try:
        data = api_call("/v1/chat/completions", {
            "model": MODEL,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": False,
        }, method="POST")
        t1 = time.perf_counter()
        tokens = data.get("usage", {}).get("completion_tokens", 0)
        dur = t1 - t0
        return {
            "ttft_ms": round(dur * 1000, 1),
            "tokens": tokens,
            "total_s": round(dur, 2),
            "gen_tok_s": round(tokens / dur, 2) if dur > 0 else 0,
        }
    except Exception as e:
        print(f"  ERROR: {e}")
        return None

def make_long_prompt(target_tokens):
    """Generate a prompt that pre-fills to approximately target_tokens tokens."""
    # Rough heuristic: ~4 chars per token for English text
    para = "The quick brown fox jumps over the lazy dog. " * 50
    repeats = max(1, target_tokens // 200)
    return [{"role": "user", "content": (para + "\n") * repeats}]

def run():
    try:
        health = api_call("/health", expect_json=False)
        print(f"Server health: {health.strip()}")
    except Exception as e:
        print(f"Server not responding at {BASE}: {e}")
        sys.exit(1)

    try:
        models = api_call("/v1/models")
        loaded = [m['id'] for m in models.get('data', []) if m.get('status',{}).get('value') == 'loaded']
        print(f"Loaded models: {loaded}")
    except Exception:
        pass

    print(f"\n=== Benchmarking {MODEL} ({BACKEND}/{IMAGE}) ===")
    results = []
    
    short_prompt = [{"role": "user", "content": "Say exactly: one two three four five six seven eight nine ten."}]
    medium_prompt = [{"role": "user", "content": "Explain quantum computing in exactly three sentences."}]
    long_prompt = [{"role": "user", "content": "Write a detailed analysis of the causes and consequences of the 2008 financial crisis, covering housing bubble, subprime mortgages, and global impact. Be thorough."}]
    
    print("\n[1] Short prompt + 128 tok gen (stream)...")
    res = stream_chat(short_prompt, max_tokens=128)
    res["name"] = "short_128_stream"
    results.append(res)
    print(f"  TTFT: {res['ttft_ms']}ms | Tokens: {res['tokens']} | Total: {res['total_s']}s | Gen: {res['gen_tok_s']} tok/s")

    print("\n[2] Medium prompt + 256 tok gen (non-stream)...")
    res = non_stream_chat(medium_prompt, max_tokens=256)
    if res:
        res["name"] = "medium_256_nostream"
        results.append(res)
        print(f"  TTFT: {res['ttft_ms']}ms | Tokens: {res['tokens']} | Total: {res['total_s']}s | Gen: {res['gen_tok_s']} tok/s")

    print("\n[3] Long prompt + 128 tok gen (stream)...")
    res = stream_chat(long_prompt, max_tokens=128)
    res["name"] = "long_128_stream"
    results.append(res)
    print(f"  TTFT: {res['ttft_ms']}ms | Tokens: {res['tokens']} | Total: {res['total_s']}s | Gen: {res['gen_tok_s']} tok/s")

    print("\n[4] Short prompt + 512 tok gen (stream, throughput test)...")
    res = stream_chat(short_prompt, max_tokens=512)
    res["name"] = "short_512_throughput"
    results.append(res)
    print(f"  TTFT: {res['ttft_ms']}ms | Tokens: {res['tokens']} | Total: {res['total_s']}s | Gen: {res['gen_tok_s']} tok/s")

    # 65K smoke test
    print("\n[5] 65K context prefill + 128 tok gen (smoke test)...")
    prefill_65k = make_long_prompt(10000)  # ~10K tokens prefill
    res = stream_chat(prefill_65k, max_tokens=128)
    res["name"] = "long_context_65k_prefill"
    results.append(res)
    print(f"  TTFT: {res['ttft_ms']}ms | Tokens: {res['tokens']} | Total: {res['total_s']}s | Gen: {res['gen_tok_s']} tok/s")

    # 256K stress test (if context configured high enough)
    print("\n[6] 256K context prefill + 128 tok gen (stress test)...")
    try:
        prefill_256k = make_long_prompt(50000)  # ~50K tokens prefill
        res = stream_chat(prefill_256k, max_tokens=128)
        res["name"] = "long_context_256k_prefill"
        results.append(res)
        print(f"  TTFT: {res['ttft_ms']}ms | Tokens: {res['tokens']} | Total: {res['total_s']}s | Gen: {res['gen_tok_s']} tok/s")
    except Exception as e:
        print(f"  SKIPPED: {e}")
        results.append({"name": "long_context_256k_prefill", "error": str(e), "gen_tok_s": 0})

    print("\n=== Summary ===")
    gen_rates = [r["gen_tok_s"] for r in results]
    print(f"Best generation throughput: {max(gen_rates)} tok/s")
    print(f"Average generation throughput: {round(statistics.mean(gen_rates), 2)} tok/s")
    for r in results:
        print(f"  {r['name']}: {r['gen_tok_s']} tok/s (TTFT: {r['ttft_ms']}ms)")

    # Append to results history with metadata
    try:
        with open("benchmark_results.json", "r") as f:
            history = json.load(f)
    except Exception:
        history = {}
    
    from datetime import datetime
    timestamp = datetime.now().isoformat()
    history[timestamp] = {
        "backend": BACKEND,
        "image": IMAGE,
        "model": MODEL,
        "results": results,
    }
    
    with open("benchmark_results.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nResults saved to benchmark_results.json (timestamp: {timestamp})")

if __name__ == "__main__":
    run()
