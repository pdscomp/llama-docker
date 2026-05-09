#!/usr/bin/env python3
"""
Benchmark llama-server OpenAI-compatible API on legion:8080
Uses only stdlib (no requests dependency).
Handles Qwen3.6 reasoning_content + content delta fields.
"""
import time, json, sys, statistics, urllib.request

BASE = "http://legion.home.swenson.co:8080"
MODEL = "Qwen3.6-27B"

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

def run():
    try:
        health = api_call("/health", expect_json=False)
        print(f"Server health: {health.strip()}")
    except Exception as e:
        print(f"Server not responding: {e}")
        sys.exit(1)

    try:
        models = api_call("/v1/models")
        loaded = [m['id'] for m in models.get('data', []) if m.get('status',{}).get('value') == 'loaded']
        print(f"Loaded models: {loaded}")
    except Exception:
        pass

    short_prompt = [{"role": "user", "content": "Say exactly: one two three four five six seven eight nine ten."}]
    medium_prompt = [{"role": "user", "content": "Explain quantum computing in exactly three sentences."}]
    long_prompt = [{"role": "user", "content": "Write a detailed analysis of the causes and consequences of the 2008 financial crisis, covering housing bubble, subprime mortgages, and global impact. Be thorough."}]
    
    print(f"\n=== Benchmarking llama.cpp ({MODEL}) ===")
    results = []
    
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

    print("\n=== Summary ===")
    gen_rates = [r["gen_tok_s"] for r in results]
    print(f"Best generation throughput: {max(gen_rates)} tok/s")
    print(f"Average generation throughput: {round(statistics.mean(gen_rates), 2)} tok/s")
    for r in results:
        print(f"  {r['name']}: {r['gen_tok_s']} tok/s (TTFT: {r['ttft_ms']}ms)")

    try:
        with open("benchmark_results.json", "r") as f:
            history = json.load(f)
    except Exception:
        history = {}
    
    from datetime import datetime
    timestamp = datetime.now().isoformat()
    history[timestamp] = results
    
    with open("benchmark_results.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nResults saved to benchmark_results.json (timestamp: {timestamp})")

if __name__ == "__main__":
    run()
