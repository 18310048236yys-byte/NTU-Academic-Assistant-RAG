import pandas as pd
from pathlib import Path


INPUT_FILE = Path("开发集结果.csv")
OUTPUT_FILE = Path("开发集人工复核.csv")


df = pd.read_csv(INPUT_FILE)


# 添加人工检查列
df["Answer Correct"] = ""
df["Citation Correct"] = ""
df["Citation Minimal"] = ""
df["Review Notes"] = ""


df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)


print("=" * 60)
print("✅ 人工复核表已生成")
print(f"文件：{OUTPUT_FILE}")
print("=" * 60)