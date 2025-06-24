# app.py

import os
import json
import re
from jsonschema import validate, ValidationError
from dotenv import load_dotenv
from openai import RateLimitError

from camel.models import ModelFactory
from camel.models import siliconflow_model, SiliconFlowModel
from camel.configs import DeepSeekConfig, deepseek_config, DEEPSEEK_API_PARAMS
from camel.configs import siliconflow_config, SiliconFlowConfig, SILICONFLOW_API_PARAMS
from camel.agents import ChatAgent
from camel.messages import BaseMessage  
from camel.types import ModelType
from camel.types.enums import ModelType, ModelPlatformType


# -------------------------------
# 1. 加载环境变量 & 读取 API Key
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

# ====================================
# 模块一：InputAgent（本地敏感词 & 注入过滤）
# ====================================
SENSITIVE_WORDS = {"涉黄", "涉政", "违法"}

def InputAgent(raw: str) -> str:
    """
    替换敏感词、去掉以 / 或 # 开头的注入行，返回过滤后的歌词。
    """
    # 敏感词屏蔽
    for w in SENSITIVE_WORDS:
        raw = raw.replace(w, "*" * len(w))
    # 去掉以 “/” 或 “#” 开头的行
    lines = []
    for line in raw.splitlines():
        if re.match(r"^\s*(/|#)", line):
            continue
        lines.append(line)
    return "\n".join(lines)

# ============================================
# 模块二：GenerationAgent（SiliconFlowModel + 流式调用）
# ============================================
# 2.1 构造 DeepSeek / SiliconFlow 的配置
#    这里选用 SiliconFlowConfig，确保包含 max_tokens 避免警告
cfg = SiliconFlowConfig(
    temperature=0.0,
    top_p=0.8,
    stream=True,
    max_tokens=1024
)
model_cfg = cfg.as_dict()

# 2.2 创建后端 & 包装成 ChatAgent
backend = ModelFactory.create(
    model_platform=ModelPlatformType.SILICONFLOW,
    model_type="Pro/deepseek-ai/DeepSeek-R1",
    model_config_dict=model_cfg,
    api_key=api_key,
    url="https://api.siliconflow.cn/v1/",
    max_retries=3
)

agent = ChatAgent(
    system_message="You are a helpful assistant.",
    model=backend
)

response = agent.step("Say hi to CAMEL AI community.")
print(response.msg.content)