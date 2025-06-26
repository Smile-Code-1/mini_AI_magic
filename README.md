# mini\_AI\_magic

这是一个基于 Camel-AI 和 SiliconFlow DeepSeek 模型的命令行 AI 应用，可将用户输入的歌词自动转换为 MV 场景描述并输出符合预定义 JSON Schema 的结构化结果。

## 目录

* [功能概述](#功能概述)
* [环境要求](#环境要求)
* [安装依赖](#安装依赖)
* [配置 API Key](#配置-api-key)
* [项目结构](#项目结构)
* [使用方法](#使用方法)
* [配置参数说明](#配置参数说明)
* [常见问题](#常见问题)

## 功能概述

1. **InputAgent**：对输入歌词做安全过滤与风格审查，屏蔽不良或敏感内容并保持艺术性。
2. **GroupingAgent**：根据情感与意象将歌词分段，便于后续场景生成。
3. **SceneAgent**：为每一段歌词生成自然语言的 MV 场景要点，包括画面元素、角色、动作等描述，并给出大致时间码。
4. **OutputAgent**：将场景要点整合为符合 `schema.json` 的原始 JSON 字符流，并以流式输出的方式打印。
5. **FormatAgent**：校验并美化 OutputAgent 的 JSON 结果，确保最终输出符合 Schema 要求。

## 环境要求

* **操作系统**：Linux/macOS/Windows
* **Python**：3.10-3.12
* **网络**：可访问 `https://api.siliconflow.cn/v1/` 及 OpenAI 服务 (如用 OpenAI API)

## 安装依赖

1. 克隆本仓库：

   ```bash
   git clone https://github.com/Smile-Code-1/mini_AI_magic.git
   cd mini_AI_magic
   ```
2. 安装 Python 依赖：

   ```bash
   pip install -r requirements.txt  
   ```

依赖列表见 `requirements.txt`：

```text
camel-ai
python-dotenv
jsonschema
openai
```

## 配置 API Key

在项目根目录创建一个名为 `.env` 的文件，并添加：

```dotenv
API_KEY=你的SiliconFlow或OpenAI API Key
```

> 若不配置或 KEY 无效，程序会报错并退出。

## 项目结构

```plain
├── .env                  # 环境变量文件，存放 API_KEY
├── requirements.txt      # Python 依赖清单
├── schema.json           # 输出 JSON Schema 定义
├── mini_AI.py                # 主程序入口
└── README.md             # 使用说明文档
```

## 使用方法

1. 在命令行中运行主程序：

   ```bash
   python mini_AI.py
   ```

2. 按提示在终端粘贴整首歌词，输入完成后按 `Ctrl+D` (Unix/macOS) 或 `Ctrl+Z` + 回车 (Windows)。

3. 程序依次输出各 Agent 的处理结果：

   * InputAgent 过滤与审核后的歌词
   * GroupingAgent 分段结果
   * SceneAgent 场景描述
   * OutputAgent 原始 JSON 流式输出
   * FormatAgent 校验并美化后的最终 JSON

4. 最终在终端可以看到结构化且符合 `schema.json` 的 MV 场景 JSON。

## 配置参数说明

在 `mini_AI.py` 顶部，可通过修改 `SiliconFlowConfig` 对象调整模型行为：

```python
# 快速通道（低温度、0.8 top_p）
cfg = SiliconFlowConfig(temperature=0, top_p=0.8, stream=True, max_tokens=8192)
# 高随机通道（高温度、1.0 top_p）
cfg_high_temp = SiliconFlowConfig(temperature=1.2, top_p=1, stream=True, max_tokens=8192)
```

你可以根据需要：

* 改变 `temperature`、`top_p` 控制创意与随机性
* 切换不同模型：`DeepSeek-V3` / `DeepSeek-R1` 等
* 修改 `max_tokens` 或重试次数 (`max_retries`)

## 常见问题

* **报错：请先在 .env 中配置 API\_KEY**
  未正确设置环境变量，检查 `.env` 文件和 Key 是否书写正确。

* **网络问题导致请求失败**
  确认网络环境可以访问 `api.siliconflow.cn`

* **JSON 校验错误**
  查看 `schema.json` 中的字段是否被改动，或检查模型输出是否符合 Schema 规范。
