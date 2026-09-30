from pathlib import Path
import pandas as pd
import numpy as np
import re


# =========================================================
# 1. 文件位置
# =========================================================

ROOT = Path(__file__).resolve().parent

RAG_FILE = ROOT / "正式测试结果.csv"
REVIEW_FILE = ROOT / "正式评估人工复核.csv"
CTRLF_FILE = ROOT / "CtrlF基线结果.csv"

DETAIL_OUTPUT = ROOT / "RAG_vs_CtrlF比较结果.csv"
SUMMARY_OUTPUT = ROOT / "RAG_vs_CtrlF汇总.csv"

# Ctrl-F 超时上限
TIMEOUT_SECONDS = 180


# =========================================================
# 2. 读取 CSV
# =========================================================

def read_csv_safely(path):
    if not path.exists():
        raise FileNotFoundError(f"找不到文件：{path}")

    for encoding in ["utf-8-sig", "utf-8", "gbk"]:
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue

    return pd.read_csv(path)


rag = read_csv_safely(RAG_FILE)
review = read_csv_safely(REVIEW_FILE)
ctrlf = read_csv_safely(CTRLF_FILE)

print("RAG正确性原始值：", review["Answer Correct"].drop_duplicates().tolist())
print("Ctrl-F正确性原始值：", ctrlf["Manual Answer Correct"].drop_duplicates().tolist())

print("\n人工复核 Answer Correct 的实际取值：")
print(review["Answer Correct"].value_counts(dropna=False))

print("\nCtrl-F Manual Answer Correct 的实际取值：")
print(ctrlf["Manual Answer Correct"].value_counts(dropna=False))

print("\n==============================")
print("读取文件成功")
print("==============================")

print("\n正式测试结果.csv 列名：")
print(list(rag.columns))

print("\n正式评估人工复核.csv 列名：")
print(list(review.columns))

print("\nCtrlF基线结果.csv 列名：")
print(list(ctrlf.columns))


# =========================================================
# 3. 工具函数
# =========================================================

def normalise_name(name):
    return re.sub(
        r"[\s_\-\(\)\[\]/]+",
        "",
        str(name).strip().lower()
    )


def find_id_column(df):
    for col in df.columns:
        n = normalise_name(col)
        if n in ["id", "questionid", "题号"]:
            return col

    # 再尝试寻找内容类似 Q01、Q02 的列
    for col in df.columns:
        values = df[col].astype(str).str.strip()
        ratio = values.str.match(r"^Q\d{1,2}$", na=False).mean()
        if ratio > 0.5:
            return col

    raise ValueError("无法找到 ID 列。")


def clean_id(value):
    value = str(value).strip().upper()

    m = re.search(r"Q(\d+)", value)
    if m:
        return f"Q{int(m.group(1)):02d}"

    return value


def is_q01_to_q50(value):
    value = clean_id(value)

    m = re.fullmatch(r"Q(\d{2})", value)
    if not m:
        return False

    number = int(m.group(1))
    return 1 <= number <= 50


def find_time_column(df, dataset_type):
    """
    自动寻找时间列。
    dataset_type:
        rag   -> RAG 响应时间
        ctrlf -> Ctrl-F 搜索时间
    """

    # ---------- 优先找明确列名 ----------
    if dataset_type == "rag":
        exact_candidates = [
            "Total Response Time",
            "Total Time",
            "Response Time",
            "Elapsed Time",
            "RAG Time",
            "总响应时间",
            "响应时间",
            "总耗时",
            "耗时"
        ]

    else:
        exact_candidates = [
            "Search Time",
            "CtrlF Time",
            "Ctrl-F Time",
            "Time",
            "Elapsed Time",
            "搜索时间",
            "用时",
            "耗时",
            "搜索用时"
        ]

    normalised = {
        normalise_name(col): col
        for col in df.columns
    }

    for candidate in exact_candidates:
        n = normalise_name(candidate)

        if n in normalised:
            return normalised[n]

    # ---------- 找包含 time / 时间 / 耗时 的列 ----------
    for col in df.columns:
        n = normalise_name(col)

        if (
            "time" in n
            or "时间" in str(col)
            or "耗时" in str(col)
            or "用时" in str(col)
        ):
            if "timeout" not in n:
                return col

    # ---------- 最后的数值推断 ----------
    possible = []

    for col in df.columns:
        n = normalise_name(col)

        # 排除明显不是时间的数据
        banned = [
            "id",
            "index",
            "score",
            "page",
            "token",
            "document",
            "source"
        ]

        if any(word in n for word in banned):
            continue

        numeric = pd.to_numeric(df[col], errors="coerce")

        valid_ratio = numeric.notna().mean()

        if valid_ratio < 0.7:
            continue

        median = numeric.median()

        if dataset_type == "rag":
            # RAG 一般几秒
            if 0.2 <= median <= 30:
                possible.append((col, median))

        else:
            # Ctrl-F 一般几十秒
            if 2 <= median <= TIMEOUT_SECONDS:
                possible.append((col, median))

    if possible:
        if dataset_type == "rag":
            # RAG 候选中优先较小的合理时间
            possible.sort(key=lambda x: x[1])
        else:
            # Ctrl-F 候选中优先接近几十秒的
            possible.sort(key=lambda x: abs(x[1] - 60))

        return possible[0][0]

    raise ValueError(
        f"无法自动找到 {dataset_type} 的时间列。"
    )


def yn_to_bool(value):
    if pd.isna(value):
        return np.nan

    # Python / numpy 布尔值
    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    # 数字 1 / 0
    if isinstance(value, (int, float, np.integer, np.floating)):
        if value == 1:
            return True
        if value == 0:
            return False

    text = str(value).strip().upper()

    # 清理可能出现的符号
    text = (
        text.replace("✅", "")
            .replace("❌", "")
            .strip()
    )

    # 正确
    if (
        text in {"Y", "YES", "TRUE", "1", "1.0", "CORRECT", "PASS", "正确", "是", "对"}
        or text.startswith("YES")
        or text.startswith("CORRECT")
        or text.startswith("Y ")
    ):
        return True

    # 错误
    if (
        text in {"N", "NO", "FALSE", "0", "0.0", "INCORRECT", "FAIL", "错误", "否", "错"}
        or text.startswith("NO")
        or text.startswith("INCORRECT")
        or text.startswith("N ")
    ):
        return False

    return np.nan


def find_answer_correct_column(df):
    candidates = [
        "Answer Correct",
        "Answer Correct?",
        "Answer Correctness",
        "Correct",
        "Correct Answer",
        "回答正确",
        "答案正确",
        "回答是否正确",
        "Correct(Y/N)",
        "Answer Correct (Y/N)"
    ]

    normalised = {
        normalise_name(col): col
        for col in df.columns
    }

    for candidate in candidates:
        n = normalise_name(candidate)

        if n in normalised:
            return normalised[n]

    # 找包含 answer + correct 的列
    for col in df.columns:
        n = normalise_name(col)

        if (
            ("answer" in n and "correct" in n)
            or "回答正确" in str(col)
            or "答案正确" in str(col)
        ):
            return col

    return None


def find_timeout_column(df):
    for col in df.columns:
        n = normalise_name(col)

        if (
            "timeout" in n
            or "超时" in str(col)
        ):
            return col

    return None


def find_ctrlf_correct_column(df):
    candidates = [
        "Correct",
        "Answer Correct",
        "Correctness",
        "回答正确",
        "答案正确",
        "是否正确"
    ]

    normalised = {
        normalise_name(col): col
        for col in df.columns
    }

    for candidate in candidates:
        n = normalise_name(candidate)

        if n in normalised:
            return normalised[n]

    for col in df.columns:
        n = normalise_name(col)

        if "correct" in n or "正确" in str(col):
            return col

    return None


# =========================================================
# 4. 找关键列
# =========================================================

rag_id_col = find_id_column(rag)
review_id_col = find_id_column(review)
ctrlf_id_col = find_id_column(ctrlf)

rag_time_col = find_time_column(rag, "rag")
ctrlf_time_col = find_time_column(ctrlf, "ctrlf")

rag_correct_col = find_answer_correct_column(review)

timeout_col = find_timeout_column(ctrlf)
ctrlf_correct_col = find_ctrlf_correct_column(ctrlf)


print("\n==============================")
print("自动识别到的列")
print("==============================")

print("RAG ID：", rag_id_col)
print("RAG 时间：", rag_time_col)

print("人工复核 ID：", review_id_col)
print("RAG 正确性：", rag_correct_col)

print("Ctrl-F ID：", ctrlf_id_col)
print("Ctrl-F 时间：", ctrlf_time_col)
print("Ctrl-F Timeout：", timeout_col)
print("Ctrl-F 正确性：", ctrlf_correct_col)


# =========================================================
# 5. 整理 RAG 数据（只取 Q01-Q50）
# =========================================================

rag_data = rag.copy()

rag_data["ID_clean"] = rag_data[rag_id_col].apply(clean_id)

rag_data = rag_data[
    rag_data["ID_clean"].apply(is_q01_to_q50)
].copy()

rag_data["RAG_Time_s"] = pd.to_numeric(
    rag_data[rag_time_col],
    errors="coerce"
)

rag_data = rag_data[
    ["ID_clean", "RAG_Time_s"]
].drop_duplicates(
    subset="ID_clean",
    keep="first"
)


# =========================================================
# 6. 整理人工复核结果
# =========================================================

review_data = review.copy()

review_data["ID_clean"] = review_data[
    review_id_col
].apply(clean_id)

review_data = review_data[
    review_data["ID_clean"].apply(is_q01_to_q50)
].copy()

if rag_correct_col is not None:

    review_data["RAG_Correct"] = review_data[
        rag_correct_col
    ].apply(yn_to_bool)

    review_data = review_data[
        ["ID_clean", "RAG_Correct"]
    ].drop_duplicates(
        subset="ID_clean",
        keep="first"
    )

else:
    print(
        "\n⚠️ 没有自动找到人工复核中的“回答正确”列。"
    )

    review_data = review_data[
        ["ID_clean"]
    ].drop_duplicates()

    review_data["RAG_Correct"] = np.nan


# =========================================================
# 7. 整理 Ctrl-F 数据
# =========================================================

ctrlf_data = ctrlf.copy()

ctrlf_data["ID_clean"] = ctrlf_data[
    ctrlf_id_col
].apply(clean_id)

ctrlf_data = ctrlf_data[
    ctrlf_data["ID_clean"].apply(is_q01_to_q50)
].copy()

ctrlf_data["CtrlF_Time_s"] = pd.to_numeric(
    ctrlf_data[ctrlf_time_col],
    errors="coerce"
)


# ---------- Timeout ----------
if timeout_col is not None:

    ctrlf_data["Timeout"] = ctrlf_data[
        timeout_col
    ].apply(yn_to_bool)

    # 有些文件可能写 TIMEOUT 字符串
    raw_timeout = ctrlf_data[
        timeout_col
    ].astype(str).str.upper()

    ctrlf_data.loc[
        raw_timeout.str.contains("TIMEOUT|超时", regex=True),
        "Timeout"
    ] = True

else:
    ctrlf_data["Timeout"] = False


# 如果 timeout 题没有时间，按 180 秒计
ctrlf_data.loc[
    (ctrlf_data["Timeout"] == True)
    & (ctrlf_data["CtrlF_Time_s"].isna()),
    "CtrlF_Time_s"
] = TIMEOUT_SECONDS


# ---------- Ctrl-F 正确性 ----------
if ctrlf_correct_col is not None:

    parsed_correct = ctrlf_data[
        ctrlf_correct_col
    ].apply(yn_to_bool)

    # 如果这一列虽然存在，但实际上全部是空值，
    # 则按实验记录：非 Timeout = 成功，Timeout = 失败
    if parsed_correct.notna().sum() == 0:

        print(
            "\n⚠️ Manual Answer Correct 列存在，但全部为空。"
            "\n因此按 Ctrl-F 实验结果处理："
            "\n非 Timeout = 正确；Timeout = 不正确。"
        )

        ctrlf_data["CtrlF_Correct"] = (
            ctrlf_data["Timeout"] != True
        )

    else:
        ctrlf_data["CtrlF_Correct"] = parsed_correct

else:

    ctrlf_data["CtrlF_Correct"] = (
        ctrlf_data["Timeout"] != True
    )

    print(
        "\n⚠️ Ctrl-F 文件中没有正确性列。"
        "\n按：非 Timeout = 正确；Timeout = 不正确。"
    )


ctrlf_data = ctrlf_data[
    [
        "ID_clean",
        "CtrlF_Time_s",
        "Timeout",
        "CtrlF_Correct"
    ]
].drop_duplicates(
    subset="ID_clean",
    keep="first"
)


# =========================================================
# 8. 合并三份数据
# =========================================================

merged = rag_data.merge(
    review_data,
    on="ID_clean",
    how="left"
)

merged = merged.merge(
    ctrlf_data,
    on="ID_clean",
    how="left"
)

merged = merged.sort_values("ID_clean").reset_index(drop=True)


# =========================================================
# 9. 计算逐题效率
# =========================================================

merged["Time_Saved_s"] = (
    merged["CtrlF_Time_s"]
    - merged["RAG_Time_s"]
)

merged["Saving_Percent"] = (
    merged["Time_Saved_s"]
    / merged["CtrlF_Time_s"]
    * 100
)

merged["Speedup_x"] = (
    merged["CtrlF_Time_s"]
    / merged["RAG_Time_s"]
)


# =========================================================
# 10. Overall 时间统计
# =========================================================

valid = merged[
    merged["RAG_Time_s"].notna()
    & merged["CtrlF_Time_s"].notna()
].copy()

rag_mean = valid["RAG_Time_s"].mean()
rag_median = valid["RAG_Time_s"].median()

ctrlf_mean = valid["CtrlF_Time_s"].mean()
ctrlf_median = valid["CtrlF_Time_s"].median()

mean_saved = valid["Time_Saved_s"].mean()
median_saved = valid["Time_Saved_s"].median()

mean_saving_percent = valid[
    "Saving_Percent"
].mean()

overall_saving_percent = (
    (ctrlf_mean - rag_mean)
    / ctrlf_mean
    * 100
)

mean_speedup = valid["Speedup_x"].mean()

ratio_speedup = (
    ctrlf_mean / rag_mean
    if rag_mean > 0
    else np.nan
)

timeout_count = int(
    valid["Timeout"].fillna(False).sum()
)


# =========================================================
# 11. Paired comparison：双方都正确
# =========================================================

paired = valid[
    (valid["RAG_Correct"] == True)
    & (valid["CtrlF_Correct"] == True)
].copy()

paired_n = len(paired)

if paired_n > 0:

    paired_rag_mean = paired[
        "RAG_Time_s"
    ].mean()

    paired_ctrlf_mean = paired[
        "CtrlF_Time_s"
    ].mean()

    paired_mean_saved = paired[
        "Time_Saved_s"
    ].mean()

    paired_median_saved = paired[
        "Time_Saved_s"
    ].median()

    paired_saving_percent = (
        (
            paired_ctrlf_mean
            - paired_rag_mean
        )
        / paired_ctrlf_mean
        * 100
    )

    paired_speedup = (
        paired_ctrlf_mean
        / paired_rag_mean
        if paired_rag_mean > 0
        else np.nan
    )

else:

    paired_rag_mean = np.nan
    paired_ctrlf_mean = np.nan
    paired_mean_saved = np.nan
    paired_median_saved = np.nan
    paired_saving_percent = np.nan
    paired_speedup = np.nan


# =========================================================
# 12. 保存逐题结果
# =========================================================

output_detail = merged.rename(
    columns={
        "ID_clean": "ID"
    }
)

output_detail.to_csv(
    DETAIL_OUTPUT,
    index=False,
    encoding="utf-8-sig"
)


# =========================================================
# 13. 保存汇总结果
# =========================================================

summary = pd.DataFrame([
    ["Compared Questions", len(valid), "questions"],

    ["RAG Mean Time", rag_mean, "seconds"],
    ["RAG Median Time", rag_median, "seconds"],

    ["Ctrl-F Mean Time", ctrlf_mean, "seconds"],
    ["Ctrl-F Median Time", ctrlf_median, "seconds"],

    ["Mean Time Saved", mean_saved, "seconds"],
    ["Median Time Saved", median_saved, "seconds"],

    [
        "Mean Per-question Saving",
        mean_saving_percent,
        "%"
    ],

    [
        "Overall Mean-Time Saving",
        overall_saving_percent,
        "%"
    ],

    [
        "Mean Per-question Speed-up",
        mean_speedup,
        "x"
    ],

    [
        "Mean-time Speed-up",
        ratio_speedup,
        "x"
    ],

    ["Ctrl-F Timeouts", timeout_count, "questions"],

    [
        "Both-correct Paired Questions",
        paired_n,
        "questions"
    ],

    [
        "Paired RAG Mean Time",
        paired_rag_mean,
        "seconds"
    ],

    [
        "Paired Ctrl-F Mean Time",
        paired_ctrlf_mean,
        "seconds"
    ],

    [
        "Paired Mean Time Saved",
        paired_mean_saved,
        "seconds"
    ],

    [
        "Paired Median Time Saved",
        paired_median_saved,
        "seconds"
    ],

    [
        "Paired Time Saving",
        paired_saving_percent,
        "%"
    ],

    [
        "Paired Speed-up",
        paired_speedup,
        "x"
    ]
], columns=[
    "Metric",
    "Value",
    "Unit"
])

summary["Value"] = pd.to_numeric(
    summary["Value"],
    errors="coerce"
)

summary.to_csv(
    SUMMARY_OUTPUT,
    index=False,
    encoding="utf-8-sig"
)


# =========================================================
# 14. 打印结果
# =========================================================

print("\n")
print("=" * 65)
print("RAG vs Ctrl-F 效率比较")
print("=" * 65)

print(f"\n有效比较题数：{len(valid)}/50")

print("\n--- RAG ---")
print(f"平均回答时间：{rag_mean:.2f} 秒")
print(f"中位回答时间：{rag_median:.2f} 秒")

print("\n--- Ctrl-F ---")
print(f"平均搜索时间：{ctrlf_mean:.2f} 秒")
print(f"中位搜索时间：{ctrlf_median:.2f} 秒")
print(f"Timeout：{timeout_count}/50")

print("\n--- Overall comparison ---")
print(f"平均每题节省：{mean_saved:.2f} 秒")
print(f"中位每题节省：{median_saved:.2f} 秒")

print(
    f"平均逐题节省比例："
    f"{mean_saving_percent:.2f}%"
)

print(
    f"按平均时间计算的节省比例："
    f"{overall_saving_percent:.2f}%"
)

print(
    f"按平均时间计算，RAG 约快："
    f"{ratio_speedup:.2f} 倍"
)

print("\n--- Both-correct paired comparison ---")
print(f"双方都正确：{paired_n} 题")

if paired_n > 0:

    print(
        f"RAG 平均时间："
        f"{paired_rag_mean:.2f} 秒"
    )

    print(
        f"Ctrl-F 平均时间："
        f"{paired_ctrlf_mean:.2f} 秒"
    )

    print(
        f"平均节省："
        f"{paired_mean_saved:.2f} 秒"
    )

    print(
        f"中位节省："
        f"{paired_median_saved:.2f} 秒"
    )

    print(
        f"Paired 时间节省比例："
        f"{paired_saving_percent:.2f}%"
    )

    print(
        f"Paired speed-up："
        f"{paired_speedup:.2f} 倍"
    )

else:
    print(
        "⚠️ 没有检测到可用于双方都正确比较的数据。"
    )


print("\n")
print("=" * 65)
print("✅ 比较完成")
print(f"逐题结果：{DETAIL_OUTPUT.name}")
print(f"汇总结果：{SUMMARY_OUTPUT.name}")
print("=" * 65)