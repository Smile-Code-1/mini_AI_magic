# app.py

import os
from dotenv import load_dotenv

# TODO: 后面会在这里引入 camel-ai、jsonschema 等
def main():
    load_dotenv()
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("请先在 .env 中配置 DEEPSEEK_API_KEY")
        return

    # TODO: 编写调用模型和校验输入输出的逻辑

if __name__ == "__main__":
    main()
