#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import sys
import re
from dotenv import load_dotenv
from openai import RateLimitError

from camel.models import ModelFactory
from camel.configs import SiliconFlowConfig
from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.types.enums import ModelPlatformType
from camel.societies.workforce import Workforce
from camel.tasks import Task

# -------------------------------
# 环境 & API Key
# -------------------------------
load_dotenv()
api_key = os.getenv("API_KEY")
if not api_key:
    raise RuntimeError("请先在 .env 中配置 API_KEY")

# -------------------------------
# 加载 JSON Schema（仅为 OutputAgent 美化参考）
# -------------------------------
with open("schema.json", encoding="utf-8") as f:
    schema = json.load(f)
schema_str = json.dumps(schema["output"], ensure_ascii=False, indent=2)

# -------------------------------
# 构建两个 DeepSeek 后端：快速 & 完整
# -------------------------------
cfg = SiliconFlowConfig(
    temperature=0,
    top_p=0.8,
    stream=True,
    # max tokens:
    max_tokens=8192
)

cfg_high_temp=SiliconFlowConfig(
    temperature=1.2,
    top_p=1,
    stream=True,
    # max tokens:
    max_tokens=8192
)

backend_fast = ModelFactory.create(
    model_platform=ModelPlatformType.SILICONFLOW,
    model_type="Pro/deepseek-ai/DeepSeek-V3",
    model_config_dict=cfg.as_dict(),
    api_key=api_key,
    url="https://api.siliconflow.cn/v1/",
    max_retries=3,
)

backend_normal = ModelFactory.create(
    model_platform=ModelPlatformType.SILICONFLOW,
    model_type="Pro/deepseek-ai/DeepSeek-R1",
    model_config_dict=cfg.as_dict(),
    api_key=api_key,
    url="https://api.siliconflow.cn/v1/",
    max_retries=3,
)

backend_fast_highTemp = ModelFactory.create(
    model_platform=ModelPlatformType.SILICONFLOW,
    model_type="Pro/deepseek-ai/DeepSeek-V3",
    model_config_dict=cfg_high_temp.as_dict(),
    api_key=api_key,
    url="https://api.siliconflow.cn/v1/",
    max_retries=3,
)

backend_normal_highTemp = ModelFactory.create(
    model_platform=ModelPlatformType.SILICONFLOW,
    model_type="Pro/deepseek-ai/DeepSeek-R1",
    model_config_dict=cfg_high_temp.as_dict(),
    api_key=api_key,
    url="https://api.siliconflow.cn/v1/",
    max_retries=3,
)

# -------------------------------
# 处理输入
# -------------------------------
def get_user_input() -> str:
    """安全获取用户输入（按 Ctrl+D 后回车结束）"""
    print("请粘贴歌词（输入完成后按 Ctrl+D，然后按回车结束）：")

    try:
        # 使用 sys.stdin.read() 读取所有输入，直到 Ctrl+D
        raw_input = sys.stdin.read()
        if not raw_input.strip():  # 检查是否为空输入
            print("错误：未接收到任何输入内容")
            sys.exit(1)
        return raw_input.strip()  # 去除首尾空白
    except KeyboardInterrupt:  # 用户按 Ctrl+C 取消
        print("\n输入已取消")
        sys.exit(0)
# -------------------------------
# 纯指令注入防护函数
# -------------------------------
def prevent_command_injection(text: str) -> str:
    """唯一目标：防止命令注入攻击"""
    # 仅检查可能用于命令注入的特殊字符
    command_chars = { '|', '&', '$', '`', '\\'}
    if any(char in text for char in command_chars):
        raise ValueError("检测到潜在命令注入字符")
    return text


# -------------------------------
# 定义五个 Agent
# -------------------------------

# 1) 输入过滤 Agent
input_agent = ChatAgent(
    system_message=BaseMessage.make_assistant_message(
        role_name="InputAgent",
        content=(
            "你是 InputAgent，只需原样返回用户输入的歌词，同时负责内容安全审核。\n"
            "检查输入的歌词是否包含血腥、暴力、色情或政治敏感内容。\n"
            "如果发现敏感内容，请将其转换为健康、积极的版本，同时保持歌词的艺术性。\n"
            "转换规则：\n"
            "  1. 血腥内容 -> 用象征性表达替代\n"
            "  2. 暴力内容 -> 转化为冲突的和平解决\n"
            "  3. 色情内容 -> 转化为浪漫的抽象表达\n"
            "  4. 政治敏感内容 -> 转化为普世价值观表达\n"
            "注意：不要解释修改原因，只返回修改后的歌词。"
        ),
    ),
    model=backend_fast,
)

# 2) 片段分组 Agent
grouping_agent = ChatAgent(
    system_message=BaseMessage.make_assistant_message(
        role_name="GroupingAgent",
        content=(
            "你是 GroupingAgent，请将输入的歌词按背景或场景逻辑切分成若干段落。\n"
            "输出格式：\n"
            "  段落 1：<对应的歌词文本>\n"
            "  段落 2：<对应的歌词文本>\n"
            "依照歌词中的情感、意象或场景变化来分组即可。\n"
            "如果输入只有一行，则不用分组。 \n"
            "每一组不要太多歌词，各个组之间长度要保持大致以助。\n"
        ),
    ),
    model=backend_fast_highTemp,
)

# 3) 场景描述 Agent
reasoning_agent = ChatAgent(
    system_message=BaseMessage.make_assistant_message(
        role_name="SceneAgent",
        content=(
            "你是 SceneAgent，请基于 GroupingAgent 输出的各段歌词，用自然语言分别描述对应的 MV 场景要点。\n"
            "对每个段落，尽可能给出画面感：视觉元素、主要角色、氛围色调、动作等。\n"
            "并尝试为每段提供一个大致的时间码范围。"
        ),
    ),
    model=backend_fast_highTemp,
)

# 4) 输出格式化 Agent
output_agent = ChatAgent(
    system_message=BaseMessage.make_assistant_message(
        role_name="OutputAgent",
        content=(
                "你是 OutputAgent，负责将 SceneAgent 对每个段落的自然语言场景描述\n"
                "整合成一个单一的 JSON 对象并直接输出。\n"
                "请严格遵守以下要求：\n"
                "1. **仅输出 JSON**，禁止任何 Markdown 代码块（```）、注释或额外说明。\n"
                "2. 输出必须以 “{” 开头，以 “}” 结尾，整体上是一个有效的 JSON 对象。\n"
                "3. 顶层必须包含键 “scenes”，其值是一个数组，且数组项按 timecode 升序排列。\n"
                "4. timecode应该是一个时间段，比如"'00:32-00:57'""
                "5. **不要**输出任何额外字段。\n"
                "6. 最终输出必须符合以下 JSON Schema（不要修改此结构）：\n"
                f"{schema_str}"
        ),
    ),
    model=backend_normal,
)

# 5) 输出格式化 Agent
format_agent = ChatAgent(
    system_message=BaseMessage.make_assistant_message(
        role_name="FormatAgent",
        content=(
                "你是 FormatAgent，负责校验 OutputAgent 是否输出的是一个标准的JSON\n"
                "如果是标准的JSON，则原封不动直接输出。\n"
                "如果不是标准的JSON，则转变为标准的JSON再输出。\n"
                "以下是如果输入不是标准JSON的输出规则：\n"
                "整合成一个单一的 JSON 对象并直接输出。\n"
                "请严格遵守以下要求：\n"
                "1. **仅输出 JSON**，禁止任何 Markdown 代码块（```）、注释或额外说明。\n"
                "2. 输出必须以 “{” 开头，以 “}” 结尾，整体上是一个有效的 JSON 对象。\n"
                "3. 顶层必须包含键 “scenes”，其值是一个数组，且数组项按 timecode 升序排列。\n"
                "4. timecode应该是一个时间段，比如"'00:32-00:57'""
                "5. **不要**输出任何额外字段。\n"
                "6. 最终输出必须符合以下 JSON Schema（不要修改此结构）：\n"
                f"{schema_str}"
        ),
    ),
    model=backend_fast,
)

# -------------------------------
# 主流程
# -------------------------------
def main():
    # 1) 读取输入（按 Ctrl+D 后回车结束）
    print("请粘贴歌词，输入完成后按一下顺序结束输入：\n"
          "（回车、Ctrl+D或者Ctrl+Z、回车） \n ")
    try:
        raw = sys.stdin.read().strip()  # 读取直到 EOF (Ctrl+D)
        if not raw:
            print("错误：未接收到任何输入内容")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n输入已取消")
        sys.exit(0)
        # 唯一的安全检查：防止指令注入
    try:
        raw = prevent_command_injection(raw)  # 仅此一处修改
    except ValueError as e:
        print(f"安全拒绝: {str(e)}")
        sys.exit(1)

    print ("处理歌词中：\n")
    # 发送给 InputAgent 处理
    resp1 = input_agent.step(raw)
    lyrics = resp1.msg.content

    # 标准化比较：去除首尾空白，统一换行符
    def normalize_text(text):
        # 统一换行符为 \n
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        # 去除首尾空白
        text = text.strip()
        # 合并连续空行
        lines = [line.strip() for line in text.split('\n')]
        return '\n'.join(filter(None, lines))  # 移除空行但保留换行结构

    normalized_raw = normalize_text(raw)
    normalized_lyrics = normalize_text(lyrics)

    print(f"\n【InputAgent 输出】\n{lyrics}")
    if normalized_lyrics != normalized_raw:
        print("\n⚠️ 检测到敏感内容，已进行安全过滤")

    # 2）分段
    resp2=grouping_agent.step(lyrics)
    grouped_lyrics=resp2.msg.content
    print(f"\n【GroupingAgent 输出】\n{grouped_lyrics}")

    # 3) 自然语言描述
    resp3 = reasoning_agent.step(grouped_lyrics)
    natural = resp3.msg.content
    print(f"\n【ReasoningAgent 输出】\n{natural}")

    # 4) JSON 流式生成
    print("\n【OutputAgent 开始流式生成 JSON】")
    raw_json = output_agent.step(natural)

    # 4) 美化并打印（无任何校验）
    print("\n【Raw JSON】")
    print(raw_json)

    # 5）校验
    output_json = raw_json.msg.content
    print ("\n【第一个输出的 JSON】")
    print (output_json)

    # 5）校验agent
    print("\n【美化后的 JSON（RAW）】")
    validate_json=format_agent.step(output_json)
    json_str = validate_json.msg.content
    print(json_str)
    data     = json.loads(json_str)
    pretty   = json.dumps(data, ensure_ascii=False, indent=2)
    print("\n【美化后的 JSON】")
    print(pretty)

if __name__ == "__main__":
    main()
