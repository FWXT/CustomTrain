import asyncio
import json
import os
from collections import Counter
from pathlib import Path
from typing import Optional

from openai import OpenAI  # 导入 OpenAI 客户端
from pydantic import BaseModel
from tqdm import tqdm


TAGGING_FEATURE_LIST = [
    '组件内的状态', '状态变量更改通知',
    '行列与堆叠', '栅格与分栏', '滚动与滑动', '按钮与选择', '文本与输入', '图片与视频', '空白与分隔', '菜单', '动画', '弹窗', '自定义组件',
    '尺寸设置', '位置设置', '边框设置', '图片边框设置', '背景设置', '透明度设置', '显隐控制', '禁用控制', '组件标识', '文本通用',
    '点击事件', '触摸事件', '挂载卸载事件', '拖拽事件', '按键事件'
]

# 获取当前文件的目录和父目录
current_dir = Path(__file__).parent  # 当前文件所在目录

class TokenLogprob(BaseModel):
    token: str
    logprob: float

class TopLogprob(BaseModel):
    token: str
    logprob: float

class SerializableLogprob(BaseModel):
    token: str
    logprob: float
    bytes: list[int]
    top_logprobs: Optional[list[TopLogprob]] = None

def read_json(json_file):
    with open(json_file, encoding='utf-8') as f:
        data = json.load(f)
    return data

def write_json(data, output_file: str):
    """将字典数据写入JSON文件, 支持中文字符.

    Args:
        data (dict): 要写入的字典数据。
        output_file (str): 输出文件的路径。
    """
    with open(output_file, 'w', encoding='utf-8') as f: # 确保以 UTF-8 编码写入文件
        json.dump(data, f, indent=2, ensure_ascii=False) # <--- 关键修改: ensure

def parse_string_to_dict(input_string: str) -> dict:
    """尝试将输入的字符串反序列化为字典.

    如果字符串包含 '```json' 和 '```', 会先去除Markdown代码块标记。
    否则，直接尝试将其反序列化。

    Args:
        input_string (str): 需要解析的字符串。

    Returns:
        dict: 反序列化后的字典对象。

    Raises:
        json.JSONDecodeError: 如果字符串不是有效的JSON格式。
        ValueError: 如果字符串处理后仍无法解析。
    """
    cleaned_string = input_string.strip()

    # 检查是否包含Markdown JSON代码块标记
    if cleaned_string.startswith('```json') and cleaned_string.endswith('```'):
        # 移除 '```json\n' 和 '\n```'
        # 使用 replace 的第二个参数 1 确保只替换第一个匹配项，避免内部文本干扰
        cleaned_string = cleaned_string.replace('```json\n', '', 1).replace('\n```', '', 1).strip()
    if cleaned_string.rstrip().endswith("```"):
        cleaned_string = cleaned_string.rstrip().replace('\n```', '', 1)
    try:
        # 尝试将处理后的字符串反序列化为字典
        data = json.loads(cleaned_string)
        if not isinstance(data, dict):
            raise ValueError("Parsed JSON is not a dictionary.")
        return data
    except json.JSONDecodeError as e:
        # 如果不是有效的JSON，抛出错误
        raise json.JSONDecodeError(f"无法将字符串反序列化为字典: {e}", e.doc, e.pos)
    except ValueError as e:
        # 抛出自定义的ValueError
        raise ValueError(f"字符串解析错误: {e}")

def fill_prompt_template(template_path, data_items, batch=False):
    """根据模板文件填充提示词,支持单条或批量数据处理.

    Args:
        template_path: prompt模板文件路径
        data_items: 单条JSON数据或数据列表
        batch: 是否批量处理,默认False

    Returns:
        str或list: 填充后的提示词(单条)或提示词列表(批量)
    """
    try:
        with open(template_path, encoding='utf-8') as f:
            template = f.read()

        # 定义替换规则模板
        def get_replacements(item):
            return {
                '{{DIFF}}': item.get('diff', ''),
            }

        # 批量处理模式
        if batch:
            if not isinstance(data_items, list):
                raise ValueError("批量处理模式下data_items必须是列表类型")

            filled_prompts = []
            for item in data_items:
                prompt = template
                replacements = get_replacements(item)
                for placeholder, value in replacements.items():
                    prompt = prompt.replace(placeholder, value.strip())
                filled_prompts.append(prompt)
            return filled_prompts

        # 单条处理模式
        else:
            filled_prompt = template
            if isinstance(data_items, list):
                data_items = data_items[0]
            replacements = get_replacements(data_items)
            for placeholder, value in replacements.items():
                filled_prompt = filled_prompt.replace(placeholder, value.strip())
            return [filled_prompt]

    except FileNotFoundError:
        print(f"找不到模板文件: {template_path}")
        return None
    except Exception as e:
        print(f"填充模板时出错: {e}")
        return None

class LLMConfig(BaseModel):
    """LLM配置."""
    model_config = {'protected_namespaces': ()}
    model_name: str = "doubao-seed-1-6-250615" # "deepseek-v3-250324"
    api_base: str = "https://ark.cn-beijing.volces.com/api/v3"
    api_key: Optional[str] = os.getenv("LLM_API_KEY")

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
                    logprobs=True
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
                    logprobs=True
                ),
                timeout=timeout
            )
        return response
    except asyncio.TimeoutError:
        raise TimeoutError(f"Request timed out after {timeout} seconds")

async def tag_data(data_file: str, save_file: str, template: str):
    config = LLMConfig()
    # 初始化 OpenAI 客户端
    # base_url 应该指向你的 LLMConfig 中的 api_base
    # api_key 可以直接从 config 中获取
    client = OpenAI(
      base_url=config.api_base,
      api_key=config.api_key,
    )

    data_items = read_json(data_file)

    template_md_file = current_dir / "template" / template
    prompts = fill_prompt_template(template_md_file, data_items, batch=True)

    async def process_item(data_item, prompt):
        new_obj = data_item.copy()  # 创建副本以避免修改原始数据
        new_obj["tags"] = []
        new_obj["tag_reasons"] = []
        try:
            response = await ask_model(client, config.model_name, prompt)
            content = response.choices[0].message.content
            logprobs = response.choices[0].logprobs.content
            output_map = parse_string_to_dict(content)
            new_obj["tags"] = output_map["tags"]
            new_obj["tag_reasons"] = output_map["tag_reasons"]

            serializable_probs = [
                SerializableLogprob(
                    token=logprob.token,
                    logprob=logprob.logprob,
                    bytes=logprob.bytes,
                    top_logprobs=[
                        TopLogprob(token=lp.token, logprob=lp.logprob)
                        for lp in (logprob.top_logprobs or [])
                    ]
                ).dict()  # 转换为可序列化的字典
                for logprob in logprobs
            ]
            new_obj["tag_probs"] = serializable_probs
            return new_obj
        except asyncio.TimeoutError:
            print("请求超时！")
            return new_obj
        except Exception as e:
            print(f"发生错误: {e}")
            return new_obj

    # 创建并执行所有任务
    tasks = [process_item(data_items[i], prompts[i]) for i in range(len(data_items))]
    results = []
    for task in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        result = await task
        results.append(result)

    # tag statistics
    tag_counter = Counter(dict.fromkeys(TAGGING_FEATURE_LIST, 0))
    unknown_tag_counter = Counter()
    for obj in results:
        for tag in obj['tags']:
            if tag in TAGGING_FEATURE_LIST:
                tag_counter[tag] += 1
            else:
                unknown_tag_counter[tag] += 1
    print(f'Total samples: {len(results)}')
    print(f'Known tags: {json.dumps(tag_counter, indent=2, ensure_ascii=False)}')
    print(f'Unknown tags: {json.dumps(unknown_tag_counter, indent=2, ensure_ascii=False)}')

    write_json(results, save_file)

if __name__ == "__main__":
  data_file = ""
  save_file = ""
  template = ""
  asyncio.run(tag_data(data_file, save_file, template))
