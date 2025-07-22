import aiohttp
import asyncio
import json
import time
import numpy as np


async def test_stream_completion():
    url = "http://localhost:10000/v1/completions"  # 修改为你的接口地址
    headers = {
        "Content-Type": "application/json",
        # "Authorization": "Bearer YOUR_API_KEY"  # ← 如有需要请取消注释
    }
    payload = {
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "prompt": "Once upon a time,",
        "max_tokens": 50,
        "temperature": 0.7,
        "stream": True
    }

    print(f"[INFO] Sending stream request to {url} ...")
    timeout = aiohttp.ClientTimeout(total=3600)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        request_start_time = time.perf_counter()
        async with session.post(url, headers=headers, json=payload) as resp:
            if resp.status != 200:
                print(f"[ERROR] Request failed with status {resp.status}")
                print(await resp.text())
                return

            print("\n[INFO] Streamed output:\n")
            first = True
            ttft = 0.0
            token_latencies = []
            previous_token_time = time.perf_counter()
            output_text = []

            async for line in resp.content:
                decoded = line.decode("utf-8").strip()
                if decoded.startswith("data: "):
                    content = decoded[len("data: "):]
                    if content == "[DONE]":
                        print("\n[INFO] [DONE] received.")
                        break
                    try:
                        chunk = json.loads(content)
                        text = chunk["choices"][0]["text"]
                        print(text, end="", flush=True)
                        output_text.append(text)

                        now_time = time.perf_counter()
                        if first:
                            ttft = now_time - request_start_time
                            first = False
                        else:
                            tbt = now_time - previous_token_time
                            token_latencies.append(tbt)
                        previous_token_time = now_time
                    except Exception as e:
                        print(f"\n[ERROR] Failed to parse chunk: {content}\n{e}")
                        break

    print("\n\n=== ⏱ LATENCY STATS ===")
    print(f"TTFT (Time to First Token): {ttft:.4f} seconds")
    if token_latencies:
        print(f"TBT mean: {np.mean(token_latencies):.4f} s")
        print(f"TBT P50 : {np.percentile(token_latencies, 50):.4f} s")
        print(f"TBT P90 : {np.percentile(token_latencies, 90):.4f} s")
        print(f"TBT P99 : {np.percentile(token_latencies, 99):.4f} s")
    else:
        print("No additional tokens streamed after the first.")

    print("\n[INFO] Full output:")
    print("".join(output_text))


if __name__ == "__main__":
    asyncio.run(test_stream_completion())
