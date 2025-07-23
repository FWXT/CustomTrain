import asyncio
import aiohttp
import time
import json
import argparse
from typing import Any, List, Tuple
from tqdm import tqdm
from transformers import AutoTokenizer
from datetime import datetime
import numpy as np


TOKENIZER_PATH = r""

URL = ""
HEADERS = {
    'Content-Type': 'application/json'
}
REQUEST_BODY = {
    "max_tokens": 2000,
    "stop": ["<|editable_region_end|>"],
    "include_stop_str_in_output": True,
    "temperature": 0.0
}

# 初始化tokenizer
_tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)

def count_tokens(text: str) -> int:
    return len(_tokenizer.encode(text))

def compute_percentiles(times: List[float]) -> Tuple[Any, ...]:
    p20 = np.percentile(times, 20)
    p50 = np.percentile(times, 50)  # 或 np.median(times)
    p90 = np.percentile(times, 90)
    return p20, p50, p90

def read_jsonl(jsonl_file: str) -> List[Any]:
    data = []
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f.readlines():
            obj = json.loads(line)
            data.append(obj)
    return data

async def fetch_stream(session, model, prompt, stats, semaphore, output_file):
    async with semaphore:
        start_time = time.time()

        first_token_time = None
        full_response = ""

        url = URL
        headers = {
            **HEADERS
        }
        data = {
            **REQUEST_BODY,
            "model": model,
            "prompt": prompt,
            "stream": True
        }

        try:
            async with session.post(
                url, headers=headers, json=data,
                ssl=False,
                timeout=aiohttp.ClientTimeout(total=60)
            ) as resp:
                if resp.status != 200:
                    print(f"Request failed with status {resp.status}")
                    return None
                
                async for line in resp.content:
                    if line.startswith(b"data:"):
                        line = line[len(b"data:"):].strip()
                        if line == b"[DONE]":
                            break
                        payload = json.loads(line)
                        content = payload["choices"][0]["text"]
                        full_response += content
                        now = time.time()
                        if first_token_time is None:
                            first_token_time = now
                        # try:
                        #     payload = json.loads(line)
                        #     content = payload["choices"][0]["text"]
                        #     full_response += content
                        #     now = time.time()
                        #     if first_token_time is None:
                        #         first_token_time = now
                        # except json.JSONDecodeError:
                        #     print(f"Failed to decode JSON: {line}")
                        #     continue
        except Exception as e:
            print(f"Request failed: {e}")
            return None

        end_time = time.time()
        wall_time = end_time - start_time
        decode_time = end_time - first_token_time if first_token_time else 0
        
        # 使用tokenizer计算output_tokens
        output_tokens = count_tokens(full_response) if full_response else 0
        decode_speed = output_tokens / decode_time if decode_time > 0 else 0

        prompt_tokens = count_tokens(prompt)
        
        record = {
            "start_timestamp": datetime.fromtimestamp(start_time).isoformat(),
            "end_timestamp": datetime.fromtimestamp(end_time).isoformat(),
            "prompt": prompt,
            "response": full_response,
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "wall_time": wall_time,
            "first_token_latency": (first_token_time - start_time) if first_token_time else None,
            "decode_time": decode_time,
            "decode_speed": decode_speed,
            "throughput": (prompt_tokens + output_tokens) / wall_time if wall_time > 0 else 0
        }
        
        stats.append(record)
        
        # 将结果写入文件
        with open(output_file, "a", encoding="utf-8") as f_out:
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
        
        return record

async def run_benchmark(prompts: List[str], model, concurrency: int, output_file: str):
    semaphore = asyncio.Semaphore(concurrency)
    stats = []
    
    # 清空或创建输出文件
    with open(output_file, "w", encoding="utf-8") as _:
        pass
    
    async with aiohttp.ClientSession() as session:
        tasks = [
            fetch_stream(session, model, prompt, stats, semaphore, output_file)
            for prompt in prompts
        ]
        
        # 使用tqdm显示进度
        for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=f"Running with concurrency={concurrency}"):
            await f  # 我们不再需要处理返回值，因为结果已经在fetch_stream中写入文件
    
    return stats

def analyze_stats(stats):
    if not stats:
        return {
            "requests": 0,
            "total_prompt_tokens": 0,
            "total_output_tokens": 0,
            "total_tokens": 0,
            "wall_time": 0,
            "throughput": 0,
            "avg_first_token_latency": 0,
            "avg_decode_speed": 0
        }
    
    total_prompt = sum(s["prompt_tokens"] for s in stats)
    total_output = sum(s["output_tokens"] for s in stats)

    earliest_start = min(datetime.fromisoformat(s["start_timestamp"]).timestamp() for s in stats)
    latest_end = max(datetime.fromisoformat(s["end_timestamp"]).timestamp() for s in stats)
    total_wall_time = latest_end - earliest_start
    
    # 计算有有效值的延迟
    valid_latencies = [s["first_token_latency"] for s in stats if s["first_token_latency"] is not None]
    avg_latency = sum(valid_latencies) / len(valid_latencies) if valid_latencies else 0
    
    # 计算有有效值的解码速度
    valid_speeds = [s["decode_speed"] for s in stats if s["decode_speed"] > 0]
    avg_decode_speed = sum(valid_speeds) / len(valid_speeds) if valid_speeds else 0
    
    throughput = (total_prompt + total_output) / total_wall_time if total_wall_time > 0 else 0

    # 计算p50和p90响应时间
    wall_times = [s["wall_time"] for s in stats]
    _, p50, p90 = compute_percentiles(wall_times)
    
    return {
        "requests": len(stats),
        "total_prompt_tokens": total_prompt,
        "total_output_tokens": total_output,
        "total_tokens": total_prompt + total_output,
        "total_wall_time": total_wall_time,
        "throughput": throughput,
        "avg_first_token_latency": avg_latency,
        "avg_decode_speed": avg_decode_speed,
        "p50": p50,
        "p90": p90
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="question.jsonl", help="Input file containing prompts")
    parser.add_argument("--model", default="step2_0.5b_cpt_merge", help="Served model name")
    parser.add_argument("--concurrency", type=int, default=1, help="Number of concurrent requests")
    parser.add_argument("--output", default="speed_stat_ascend/ascend_0.5b_1p_1tp.jsonl", help="Output file for detailed results")
    args = parser.parse_args()

    raw_data = read_jsonl(args.file)[:]
    prompts = [obj["prompt"] for obj in raw_data]

    print(f"Starting benchmark with {len(prompts)} prompts...")
    stats = asyncio.run(run_benchmark(prompts, args.model, args.concurrency, args.output))
    result = analyze_stats(stats)

    print("\n=== Benchmark Results ===")
    print(f"并发数: {args.concurrency}")
    print(f"请求数: {result['requests']}")
    print(f"总 Prompt Tokens: {result['total_prompt_tokens']}")
    print(f"总 Output Tokens: {result['total_output_tokens']}")
    print(f"总 Wall Time: {result['total_wall_time']:.2f} 秒")
    print(f"吞吐量: {result['throughput']:.2f} tokens/sec")
    print(f"平均首 token 延迟: {result['avg_first_token_latency']:.2f} 秒")
    print(f"平均解码速度: {result['avg_decode_speed']:.2f} tokens/sec")
    print(f"响应时间 P50: {result['p50']:.2f} 秒")
    print(f"响应时间 P90: {result['p90']:.2f} 秒")
    print(f"\n详细结果已保存到: {args.output}")

if __name__ == "__main__":
    main()