import pandas as pd
import random
import time
from pathlib import Path


TEST_FILE = Path("正式测试集.csv")
OUTPUT_FILE = Path("CtrlF基线结果.csv")


# ==========================================
# 读取正式测试集
# ==========================================

df = pd.read_csv(TEST_FILE)

# 只保留 50 道 answerable
questions = df[
    df["Type"] == "answerable"
].copy()


# ==========================================
# 随机顺序
# 固定 seed 保证实验可复现
# ==========================================

questions = questions.sample(
    frac=1,
    random_state=6201
).reset_index(drop=True)


results = []


print("=" * 70)
print("Ctrl-F Manual Baseline")
print("=" * 70)

print("""
规则：
1. 只使用 D1-D10 PDF。
2. 只允许 Ctrl-F / 手动阅读。
3. 不查看 Reference Answer。
4. 找到答案和来源后返回这里。
5. 每题最多 180 秒。
""")


input("准备好后按 Enter 开始第一题...")


# ==========================================
# 正式测试
# ==========================================

for i, row in questions.iterrows():

    print("\n" + "=" * 70)

    print(
        f"[{i + 1}/50] {row['ID']}"
    )

    print(
        f"\nQuestion:\n{row['Question']}\n"
    )

    input(
        "按 Enter 后开始计时..."
    )

    start = time.perf_counter()

    print("\n⏱️ 计时开始！去 PDF 中 Ctrl-F 查找答案。\n")

    answer = input(
        "找到后输入你的答案：\n"
    )

    document = input(
        "来源文档（例如 D3）："
    )

    page = input(
        "PDF页码："
    )

    elapsed = time.perf_counter() - start

    timeout = elapsed > 180

    print(
        f"\n用时：{elapsed:.2f} 秒"
    )

    if timeout:
        print("⚠️ 超过 180 秒，记为 Timeout")


    results.append({
        "ID": row["ID"],
        "Question": row["Question"],
        "Manual Answer": answer,
        "Manual Document": document,
        "Manual Page": page,
        "Time Seconds": round(elapsed, 2),
        "Timeout": "Y" if timeout else "N",
        "Reference Answer": row["Reference Answer"],
        "Expected Document": row["Expected Document"],
        "Expected Page": row["Expected Page"],
        "Manual Answer Correct": "",
        "Manual Source Correct": ""
    })


# ==========================================
# 保存
# ==========================================

result_df = pd.DataFrame(results)

result_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)


print("\n" + "=" * 70)
print("✅ Ctrl-F baseline 完成")
print(f"结果文件：{OUTPUT_FILE}")
print("=" * 70)

print(
    f"\n平均搜索时间："
    f"{result_df['Time Seconds'].mean():.2f} 秒"
)

print(
    f"中位数搜索时间："
    f"{result_df['Time Seconds'].median():.2f} 秒"
)

print(
    f"Timeout："
    f"{(result_df['Timeout'] == 'Y').sum()}/50"
)

