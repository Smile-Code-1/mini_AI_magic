#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
from jsonschema import validate, ValidationError
from dotenv import load_dotenv
from openai import RateLimitError

from camel.models import ModelFactory
from camel.models import siliconflow_model, SiliconFlowModel
from camel.configs import DeepSeekConfig, deepseek_config, DEEPSEEK_API_PARAMS
from camel.configs import siliconflow_config, SiliconFlowConfig, SILICONFLOW_API_PARAMS
from camel.agents import ChatAgent
from camel.types.enums import ModelPlatformType

# -------------------------------
# 1. 环境 & API Key
# -------------------------------
load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY")
if not api_key:
    raise RuntimeError("请先在 .env 中配置 DEEPSEEK_API_KEY")

# -------------------------------
# 2. 加载 JSON Schema
# -------------------------------
with open("schema.json", encoding="utf-8") as f:
    schema = json.load(f)
input_schema  = schema["input"]
output_schema = schema["output"]
schema_str    = json.dumps(output_schema, ensure_ascii=False, indent=2)

# -------------------------------
# 3. 构建 DeepSeek 后端
# -------------------------------
cfg = SiliconFlowConfig(
    temperature=0.0,
    top_p=0.8,
    stream=False,      # 关闭流式
    max_tokens=1024
)
model_cfg = cfg.as_dict()

backend = ModelFactory.create(
    model_platform=ModelPlatformType.SILICONFLOW,
    model_type="Pro/deepseek-ai/DeepSeek-R1",
    model_config_dict=model_cfg,
    api_key=api_key,
    url="https://api.siliconflow.cn/v1/",
    max_retries=3
)

# -------------------------------
# 4. 定义三个 Agent，全部使用相同 DeepSeek 后端
# -------------------------------
input_agent = ChatAgent(
    "你是 InputAgent，接收用户输入的歌词并原样返回，不做任何修改。",
    model=backend
)

generation_agent = ChatAgent(
    "你是 GenerationAgent，根据输入的歌词生成符合 JSON Schema 的 MV 场景描述，仅返回 JSON 内容。",
    model=backend
)

formatter_agent = ChatAgent(
    "你是 FormatterAgent，接收一段 JSON 文本并返回带 2 空格缩进的美化结果。",
    model=backend
)

# -------------------------------
# 5. 主流程
# -------------------------------
def main():
    # 5.1 读取用户输入
    raw_lyrics = input("请输入歌词：\n> ")

    # 5.2 InputAgent 处理
    resp1 = input_agent.step(raw_lyrics)
    lyrics = resp1  # 原样回显

    # 5.3 验证输入格式
    # 5.4 GenerationAgent 生成 Raw JSON
    try:
        resp2 = generation_agent.step(lyrics)
        raw_json = resp2
        print("\n--- Raw JSON ---")
        print(raw_json)
    except RateLimitError:
        print("调用过于频繁，超过配额限制，请稍后再试。")
        return

    # 5.5 FormatterAgent 美化 JSON
    resp3 = formatter_agent.step(raw_json)
    pretty = resp3
    print("\n--- 美化后 JSON ---")
    print(pretty)


if __name__ == "__main__":
    main()
