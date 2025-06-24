#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
from dotenv import load_dotenv
from openai import RateLimitError, OpenAI

from camel.models import ModelFactory, SiliconFlowModel
from camel.configs import SiliconFlowConfig
from camel.agents import ChatAgent, deductive_reasoner_agent, task_agent
from camel.types.enums import ModelPlatformType

# -------------------------------
# 环境 & API Key
# -------------------------------
load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY")
if not api_key:
    raise RuntimeError("请先在 .env 中配置 DEEPSEEK_API_KEY")

# -------------------------------
# 加载 JSON Schema（仅为后续美化参考）
# -------------------------------
with open("schema.json", encoding="utf-8") as f:
    schema = json.load(f)
schema_str = json.dumps(schema["output"], ensure_ascii=False, indent=2)

# -------------------------------
# 构建 DeepSeek 后端
# -------------------------------
cfg = SiliconFlowConfig(
    temperature=0.0,
    top_p=0.8,
    stream=True,
    max_tokens=1024
)

backend = ModelFactory.create(
    model_platform=ModelPlatformType.SILICONFLOW,
    model_type="Pro/deepseek-ai/DeepSeek-R1",
    model_config_dict=cfg.as_dict(),
    api_key=api_key,
    url="https://api.siliconflow.cn/v1/",
    max_retries=3
)

backend_fast = ModelFactory.create(
    model_platform=ModelPlatformType.SILICONFLOW,
    model_type="Pro/deepseek-ai/DeepSeek-V3",
    model_config_dict=cfg.as_dict(),
    api_key=api_key,
    url="https://api.siliconflow.cn/v1/",
    max_retries=3
)


# -------------------------------
# 定义三个 Agent
# -------------------------------
input_agent = ChatAgent(
    "你是 InputAgent，只需原样返回用户输入的歌词，同时负责内容安全审核。检查输入的歌词是否包含血腥、暴力、色情或政治敏感内容。"
    "如果发现敏感内容，请将其转换为健康、积极的版本，同时保持歌词的艺术性。"
    "如果没有敏感内容，请原样返回。"
    "转换规则："
    "1. 血腥内容 -> 用象征性表达替代"
    "2. 暴力内容 -> 转化为冲突的和平解决"
    "3. 色情内容 -> 转化为浪漫的抽象表达"
    "4. 政治敏感内容 -> 转化为普世价值观表达"
    "注意：不要解释修改原因，只返回修改后的歌词。",
    model=backend_fast
)

sorting_agent=ChatAgent(
    
    model=backend_fast
)

reasoning_agent = ChatAgent(
    "你是 ReasoningAgent，请基于输入的歌词，用自然语言描述 MV 场景要点。",
    model=backend_fast
)

# -------------------------------
# OutputAgent：流式输出严格 JSON
# -------------------------------
output_agent=ChatAgent(
     "你是 OutputAgent，接收自然语言描述，"
        "请严格按照以下 JSON Schema 输出，仅返回 JSON，不要多余文字：\n"
        + schema_str,
    model=backend

)

# -------------------------------
# 主流程
# -------------------------------
def main():
    # 1) 读取输入 & 原样回显
    raw = input("请输入歌词：\n> ")
    resp1 = input_agent.step(raw)
    lyrics = resp1.msg.content
    print(f"\n【InputAgent 输出】\n{lyrics}")
    if  lyrics!= raw:
        print("\n⚠️ 检测到敏感内容，已进行安全过滤")

    # 2) 自然语言描述
    resp2 = reasoning_agent.step(lyrics)
    natural = resp2.msg.content
    print(f"\n【ReasoningAgent 输出】\n{natural}")

    # 3) JSON 流式生成
    print("\n【OutputAgent 开始流式生成 JSON】")
    raw_json = output_agent.step(natural)

    # 4) 美化并打印（无任何校验）
    print("\n【Raw JSON】")
    print(raw_json)

    print("\n【美化后的 JSON】")
    data = json.loads(raw_json.msg.content)
    pretty = json.dumps(data, ensure_ascii=False, indent=2)
    print(pretty)

if __name__ == "__main__":
    main()
