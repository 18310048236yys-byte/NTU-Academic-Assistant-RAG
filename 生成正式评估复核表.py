import pandas as pd
from pathlib import Path


INPUT_FILE = Path("正式测试结果.csv")
OUTPUT_FILE = Path("正式评估人工复核.csv")


df = pd.read_csv(INPUT_FILE)


# 你的结果文件一共应该有16列
# 这里直接按位置重新指定正确列名，解决之前表头异常的问题
correct_columns = [
    "ID",
    "Question",
    "Type",
    "Reference Answer",
    "Expected Document",
    "Expected Page",
    "Top1 Score",
    "Top1 Document",
    "Top1 Page",
    "Predicted Status",
    "Model Answer",
    "Cited Documents",
    "Response Seconds",
    "Input Tokens",
    "Output Tokens",
    "Total Tokens"
]


if len(df.columns) != len(correct_columns):
    raise ValueError(
        f"当前CSV有 {len(df.columns)} 列，"
        f"但预期应该有 {len(correct_columns)} 列。"
    )


df.columns = correct_columns


# 增加正式人工评分字段
df["Answer Correct"] = ""
df["Citation Correct"] = ""
df["Review Notes"] = ""


df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)


print("=" * 60)
print("✅ 正式评估人工复核表已生成")
print(f"文件：{OUTPUT_FILE}")
print("=" * 60)