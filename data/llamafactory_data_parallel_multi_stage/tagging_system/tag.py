
import os
import asyncio
from pydantic import BaseModel
from openai import OpenAI # 导入 OpenAI 客户端


class LLMConfig(BaseModel):
    model_config = {'protected_namespaces': ()}
    model_name: str = "deepseek-v3-250324" #"doubao-seed-1-6-250615"
    api_base: str = "https://ark.cn-beijing.volces.com/api/v3"
    api_key: str = ""


async def ask_model(client, model_name, prompt, timeout=60):
    try:
        if asyncio.iscoroutinefunction(client.chat.completions.create):
            # 异步调用
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    extra_body={"thinking": {"type": "disabled"}},
                ),
                timeout=timeout
            )
        else:
            # 同步调用，转异步
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    client.chat.completions.create,
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    extra_body={"thinking": {"type": "disabled"}},
                ),
                timeout=timeout
            )
        return response
    except asyncio.TimeoutError:
        raise TimeoutError(f"Request timed out after {timeout} seconds")

async def main():
  config = LLMConfig()
  # 初始化 OpenAI 客户端
  # base_url 应该指向你的 LLMConfig 中的 api_base
  # api_key 可以直接从 config 中获取
  client = OpenAI(
      base_url=config.api_base,
      api_key=config.api_key,
  )

  # 示例调用
  prompt = "你好，请问你是谁？"
  try:
    response = await ask_model(client, config.model_name, prompt)
    print("模型回复:", response.choices[0].message.content)
  except asyncio.TimeoutError:
    print("请求超时！")
  except Exception as e:
    print(f"发生错误: {e}")

if __name__ == "__main__":
  asyncio.run(main())