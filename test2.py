
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
from camel.types.enums import ModelPlatformType,OpenAIBackendRole
from camel.societies.workforce import Workforce
from camel.tasks import Task

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
from camel.types.enums import ModelPlatformType,OpenAIBackendRole
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

backend_normal = ModelFactory.create(
    model_platform=ModelPlatformType.SILICONFLOW,
    model_type="Pro/deepseek-ai/DeepSeek-R1",
    model_config_dict=cfg.as_dict(),
    api_key=api_key,
    url="https://api.siliconflow.cn/v1/",
    max_retries=3,
)


# -------------------------------
# 处理输入
# -------------------------------
def get_user_input() -> str:
    """获取用户输入"""
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
    return raw

# -------------------------------
# 定义流式响应输出函数，对话结束时可返回累计输出
# -------------------------------
def resp(agent: ChatAgent, input: str) -> str:
    """流式响应输出函数，参数为Agent和提示词，将Agent对于提示词的响应实时流式广播到终端，对话结束时返回累计输出"""
    accumulated_resp = str()  # 累计响应，用于作为返回值
    agent.update_memory(BaseMessage.make_user_message(role_name="User", content=input), OpenAIBackendRole.USER)
    messages = agent.memory.get_context()  # 用输入内容更新语境记忆，以确保响应内容是针对用户输入而非仅仅系统信息
    for chunk in (agent.model_backend.run(messages[0])):
        chunk_content = chunk.choices[0].delta.content  # 记录单次响应传输内容
        reasoning_output=chunk.choices[0].delta.reasoning_content
        if isinstance(chunk_content, str):  # 确保响应为str，避免NoneType等数值类型干扰
            print(chunk_content, end='')  # 响应字符串打印
            accumulated_resp += chunk_content  # 响应字符串累加
        if isinstance(reasoning_output,str):
            print(reasoning_output,end='',flush=True)
    print()
    return accumulated_resp  # 返回累计响应

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

# 4) 输出格式化 Agent
output_agent = ChatAgent(
    system_message=BaseMessage.make_assistant_message(
        role_name="OutputAgent",
        content=(
                "Make thoughts and response be brief"
        ),
    ),
    model=backend_normal,
)

# -------------------------------
# 主流程
# -------------------------------
def main():
    # 1) 读取输入（按 Ctrl+D 后回车结束）
    user_input=get_user_input()

    output=resp(output_agent,user_input)
    print (output)

if __name__ == "__main__":
    main()
