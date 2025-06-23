# app.py

import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from jsonschema import validate, ValidationError

# -------------------------------
# 1. 加载环境变量并读取 API Key
# -------------------------------
load_dotenv()  # 从 .env 文件中加载环境变量
api_key = os.getenv("DEEPSEEK_API_KEY")
if not api_key:
    raise RuntimeError("请先在 .env 中配置 DEEPSEEK_API_KEY")

# -------------------------------
# 2. 加载 JSON Schema 定义
# -------------------------------
with open("schema.json", encoding="utf-8") as f:
    schema = json.load(f)
    schema_str = json.dumps(schema["output"], ensure_ascii=False, indent=2)

# -------------------------------
# 3. 初始化 OpenAI 客户端（指向 DeepSeek 接口）
# -------------------------------
client = OpenAI(
    api_key=api_key,
    base_url="https://api.siliconflow.cn/v1"  # SiliconFlow DeepSeek 的基础 URL
)

# -------------------------------
# 4. 流式调用函数：生成并打印增量输出
# -------------------------------
def generate_scene_text(lyrics: str) -> str:
    """
    对传入的歌词发起流式请求，实时打印增量内容，返回完整的响应文本。
    """

    system_content = (
        "你是一个专业的 MV 场景描述助手。\n"
        "请**严格**按照以下 JSON Schema 输出，**只返回 JSON**，"
        "不要任何多余文字：\n" + schema_str
    )

    response = client.chat.completions.create(
        model="Pro/deepseek-ai/DeepSeek-R1",  # DeepSeek 模型名称
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user",   "content": lyrics}
        ],
        temperature=0,
        top_p=0.8,
        stream=True
    )

    full_text = ""
    for chunk in response:
        # 有时 chunk.choices 可能为空，需要跳过
        if not chunk.choices:
            continue

        delta = chunk.choices[0].delta
        # delta.content 包含正常输出
        if hasattr(delta, "content") and delta.content:
            print(delta.content, end="", flush=True)
            full_text += delta.content
        # delta.reasoning_content 包含额外推理片段（可选）
        if hasattr(delta, "reasoning_content") and delta.reasoning_content:
            print(delta.reasoning_content, end="", flush=True)
            full_text += delta.reasoning_content

    print()  # 流式结束后换行
    return full_text

# -------------------------------
# 5. 主流程：读取输入、校验、调用、校验、输出
# -------------------------------
def main():
    # 5.1 从用户读取歌词
    lyrics = input("请输入歌词：\n> ")

    # 5.2 校验输入格式
    try:
        validate({"lyrics": lyrics}, schema["input"])
    except ValidationError as e:
        print("输入校验失败：", e.message)
        return

    # 5.3 调用 DeepSeek，流式打印并收集完整响应
    raw_json = generate_scene_text(lyrics)
    print("模型返回的原始 JSON：", raw_json)

    # 5.4 尝试将响应解析为 JSON
    try:
        output_data = json.loads(raw_json)
    except json.JSONDecodeError as e:
        print("返回内容不是合法的 JSON：", e)
        return

    # 5.5 校验输出格式
    try:
        validate(output_data, schema["output"])
    except ValidationError as e:
        print("输出校验失败：", e.message)
        return

    # 5.6 结构化打印最终场景列表
    print("\n解析后的场景列表：")
    for idx, scene in enumerate(output_data["scenes"], start=1):
        timecode = scene.get("timecode", f"第{idx}段")
        desc = scene["description"]
        print(f"{idx}. [{timecode}] {desc}")

# -------------------------------
# 6. 程序入口
# -------------------------------
if __name__ == "__main__":
    main()
