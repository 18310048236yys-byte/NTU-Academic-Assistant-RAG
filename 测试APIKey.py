import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")

if api_key:
    print("✅ OpenRouter API Key 已成功读取")
    print("Key 开头：", api_key[:10] + "...")
else:
    print("❌ 没有读取到 OpenRouter API Key")