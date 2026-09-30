from pathlib import Path
import pandas as pd
import numpy as np
import re


# =========================================================
# 1. 文件位置
# =========================================================

ROOT = Path(__file__).resolve().parent

TEST_FILE = ROOT / "正式测试结果.csv"
REVIEW_FILE = ROOT / "正式评估人工复核.csv"
RECALL_FILE = ROOT / "Recall@5结果.csv"
TIME_FILE = ROOT / "RAG_vs_CtrlF汇总.csv"
COST_FILE = ROOT / "成本汇总.csv"

OUTPUT_FILE = ROOT / "最终实验结果汇总.csv"


# =========================================================
# 2. 通用读取函数
# =========================================================

def read_csv_safely(path, required=True):
    if not path.exists():
        if required:
            raise FileNotFoundError(f"找不到文件：{path.name}")
        return None

    for encoding in ["utf-8-sig", "utf-8", "gbk"]:
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue

    return pd.read_csv(path)


def normalise_name(name):
    return re.sub(
        r"[\s_\-\(\)\[\]/:@]+",
        "",
        str(name).strip().lower()
    )


def find_column(df, candidates):
    normalised = {
        normalise_name(col): col
        for col in df.columns
    }

    for candidate in candidates:
        key = normalise_name(candidate)

        if key in normalised:
            return normalised[key]

    return None


def clean_id(value):
    text = str(value).strip().upper()

    match = re.search(r"Q(\d+)", text)

    if match:
        return f"Q{int(match.group(1)):02d}"

    return text


def yn_to_bool(value):
    if pd.isna(value):
        return np.nan

    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    text = str(value).strip().upper()

    if text in {
        "Y", "YES", "TRUE", "1", "1.0",
        "CORRECT", "PASS", "正确", "是", "对"
    }:
        return True

    if text in {
        "N", "NO", "FALSE", "0", "0.0",
        "INCORRECT", "FAIL", "错误", "否", "错"
    }:
        return False

    return np.nan


def safe_float(value):
    try:
        return float(value)
    except:
        return np.nan


# =========================================================
# 3. 读取文件
# =========================================================

test = read_csv_safely(TEST_FILE)
review = read_csv_safely(REVIEW_FILE)
recall = read_csv_safely(RECALL_FILE, required=False)
time_df = read_csv_safely(TIME_FILE)
cost_df = read_csv_safely(COST_FILE)


print("\n" + "=" * 70)
print("读取实验结果文件")
print("=" * 70)

print("✅ 正式测试结果.csv")
print("✅ 正式评估人工复核.csv")

if recall is not None:
    print("✅ Recall@5结果.csv")
else:
    print("⚠️ 未找到 Recall@5结果.csv，将使用已记录的 Recall@5 数值。")

print("✅ RAG_vs_CtrlF汇总.csv")
print("✅ 成本汇总.csv")


# =========================================================
# 4. 正式测试基础信息
# =========================================================

test_id_col = find_column(
    test,
    ["ID", "Question ID"]
)

test_type_col = find_column(
    test,
    ["Type", "Question Type"]
)

status_col = find_column(
    test,
    [
        "Predicted Status",
        "Status",
        "Prediction Status"
    ]
)


if test_id_col is None:
    raise ValueError("正式测试结果.csv 中找不到 ID 列。")


test["ID_clean"] = test[test_id_col].apply(clean_id)

test = test[
    test["ID_clean"].str.match(r"^Q\d{2}$", na=False)
].copy()

test = test.drop_duplicates(
    subset="ID_clean",
    keep="first"
)


answerable_test = test[
    test["ID_clean"].apply(
        lambda x: 1 <= int(x[1:]) <= 50
    )
].copy()

unanswerable_test = test[
    test["ID_clean"].apply(
        lambda x: 51 <= int(x[1:]) <= 60
    )
].copy()


# =========================================================
# 5. 人工复核：Answer Correct
# =========================================================

review_id_col = find_column(
    review,
    ["ID", "Question ID"]
)

answer_correct_col = find_column(
    review,
    [
        "Answer Correct",
        "Answer Correct?",
        "Answer Correctness",
        "回答正确",
        "答案正确"
    ]
)


if review_id_col is None:
    raise ValueError("正式评估人工复核.csv 中找不到 ID 列。")

if answer_correct_col is None:
    raise ValueError(
        "正式评估人工复核.csv 中找不到 Answer Correct 列。"
    )


review["ID_clean"] = review[
    review_id_col
].apply(clean_id)

review_answerable = review[
    review["ID_clean"].apply(
        lambda x:
        bool(re.fullmatch(r"Q\d{2}", x))
        and 1 <= int(x[1:]) <= 50
    )
].copy()

review_answerable = review_answerable.drop_duplicates(
    subset="ID_clean",
    keep="first"
)

review_answerable["Answer_Correct_bool"] = (
    review_answerable[answer_correct_col]
    .apply(yn_to_bool)
)


answer_correct_count = int(
    (review_answerable["Answer_Correct_bool"] == True).sum()
)

answer_wrong_count = int(
    (review_answerable["Answer_Correct_bool"] == False).sum()
)

answer_reviewed_count = (
    answer_correct_count
    + answer_wrong_count
)

answer_accuracy = (
    answer_correct_count / 50 * 100
)


# =========================================================
# 6. Citation correctness
# =========================================================

citation_col = None

# 优先匹配常见名称
citation_candidates = [
    "Citation Correct",
    "Citation Correct?",
    "Citation Valid",
    "Valid Citation",
    "Citation Accuracy",
    "Source Citation Correct",
    "引用正确",
    "引用有效"
]

citation_col = find_column(
    review,
    citation_candidates
)

# 如果上面没找到，再模糊搜索
if citation_col is None:
    for col in review.columns:
        n = normalise_name(col)

        if (
            "citation" in n
            and (
                "correct" in n
                or "valid" in n
            )
        ):
            citation_col = col
            break


citation_valid_count = np.nan
citation_reviewed_count = np.nan
citation_accuracy = np.nan

if citation_col is not None:

    review_answerable[
        "Citation_Correct_bool"
    ] = review_answerable[
        citation_col
    ].apply(yn_to_bool)

    citation_valid_count = int(
        (
            review_answerable[
                "Citation_Correct_bool"
            ] == True
        ).sum()
    )

    citation_invalid_count = int(
        (
            review_answerable[
                "Citation_Correct_bool"
            ] == False
        ).sum()
    )

    citation_reviewed_count = (
        citation_valid_count
        + citation_invalid_count
    )

    # 按计划书：valid-citation cases / 50
    citation_accuracy = (
        citation_valid_count / 50 * 100
    )

    print(
        f"\n检测到 Citation 列：{citation_col}"
    )

else:
    print(
        "\n⚠️ 没有自动找到 Citation Correct 列。"
        "\n最终汇总里 Citation Accuracy 会暂时留空。"
    )


# =========================================================
# 7. Unanswerable 正确拒答
# =========================================================

correct_abstention_count = 0

if status_col is not None:

    statuses = (
        unanswerable_test[status_col]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    correct_abstention_count = int(
        statuses.str.contains(
            "ABSTAIN",
            na=False
        ).sum()
    )

else:

    print(
        "\n⚠️ 没找到 Predicted Status 列，"
        "无法自动计算正确拒答。"
    )


abstention_accuracy = (
    correct_abstention_count
    / 10
    * 100
)


# =========================================================
# 8. Recall@5
# =========================================================

# 已经正式记录的结果
document_recall_count = 47
exact_evidence_recall_count = 42

# 如果 Recall@5结果.csv 存在，
# 尝试从文件自动重新计算
if recall is not None:

    doc_col = None
    evidence_col = None

    for col in recall.columns:
        n = normalise_name(col)

        if (
            "document" in n
            and "recall" in n
        ):
            doc_col = col

        if (
            "evidence" in n
            and "recall" in n
        ):
            evidence_col = col

    if doc_col is not None:

        parsed = recall[
            doc_col
        ].apply(yn_to_bool)

        count = int(
            (parsed == True).sum()
        )

        # 只有合理时才覆盖
        if 0 < count <= 50:
            document_recall_count = count

    if evidence_col is not None:

        parsed = recall[
            evidence_col
        ].apply(yn_to_bool)

        count = int(
            (parsed == True).sum()
        )

        if 0 < count <= 50:
            exact_evidence_recall_count = count


document_recall_percent = (
    document_recall_count / 50 * 100
)

exact_evidence_recall_percent = (
    exact_evidence_recall_count / 50 * 100
)


# =========================================================
# 9. 从汇总型 CSV 中读取 Metric
# =========================================================

def get_metric(df, metric_name):
    if df is None:
        return np.nan

    metric_col = find_column(
        df,
        ["Metric"]
    )

    value_col = find_column(
        df,
        ["Value"]
    )

    if (
        metric_col is None
        or value_col is None
    ):
        return np.nan

    target = normalise_name(metric_name)

    for _, row in df.iterrows():

        current = normalise_name(
            row[metric_col]
        )

        if current == target:
            return row[value_col]

    return np.nan


# =========================================================
# 10. 时间指标
# =========================================================

rag_mean_time = safe_float(
    get_metric(
        time_df,
        "RAG Mean Time"
    )
)

rag_median_time = safe_float(
    get_metric(
        time_df,
        "RAG Median Time"
    )
)

ctrlf_mean_time = safe_float(
    get_metric(
        time_df,
        "Ctrl-F Mean Time"
    )
)

ctrlf_median_time = safe_float(
    get_metric(
        time_df,
        "Ctrl-F Median Time"
    )
)

mean_time_saved = safe_float(
    get_metric(
        time_df,
        "Mean Time Saved"
    )
)

overall_time_saving = safe_float(
    get_metric(
        time_df,
        "Overall Mean-Time Saving"
    )
)

overall_speedup = safe_float(
    get_metric(
        time_df,
        "Mean-time Speed-up"
    )
)

ctrlf_timeouts = safe_float(
    get_metric(
        time_df,
        "Ctrl-F Timeouts"
    )
)

paired_questions = safe_float(
    get_metric(
        time_df,
        "Both-correct Paired Questions"
    )
)

paired_rag_mean = safe_float(
    get_metric(
        time_df,
        "Paired RAG Mean Time"
    )
)

paired_ctrlf_mean = safe_float(
    get_metric(
        time_df,
        "Paired Ctrl-F Mean Time"
    )
)

paired_mean_saved = safe_float(
    get_metric(
        time_df,
        "Paired Mean Time Saved"
    )
)

paired_saving = safe_float(
    get_metric(
        time_df,
        "Paired Time Saving"
    )
)

paired_speedup = safe_float(
    get_metric(
        time_df,
        "Paired Speed-up"
    )
)


# =========================================================
# 11. 成本指标
# =========================================================

total_api_cost = safe_float(
    get_metric(
        cost_df,
        "Total Online API Cost"
    )
)

avg_api_cost = safe_float(
    get_metric(
        cost_df,
        "Average Online API Cost per Question"
    )
)

answerable_cost = safe_float(
    get_metric(
        cost_df,
        "Answerable Mean API Cost"
    )
)

unanswerable_cost = safe_float(
    get_metric(
        cost_df,
        "Unanswerable Mean API Cost"
    )
)

corpus_chunks = safe_float(
    get_metric(
        cost_df,
        "Corpus Chunks"
    )
)

corpus_embedding_tokens = safe_float(
    get_metric(
        cost_df,
        "Corpus Embedding Tokens"
    )
)

corpus_embedding_cost = safe_float(
    get_metric(
        cost_df,
        "One-off Corpus Embedding Cost"
    )
)

amortisation_volume = safe_float(
    get_metric(
        cost_df,
        "Amortisation Query Volume"
    )
)

amortised_cost = safe_float(
    get_metric(
        cost_df,
        "Amortised Corpus Cost per Query"
    )
)

total_cost_amortised = safe_float(
    get_metric(
        cost_df,
        "Average Cost incl. Amortised Corpus"
    )
)


# =========================================================
# 12. 是否达到项目成功目标
# =========================================================

TARGET_PERCENT = 85.0
TARGET_CORRECT_COUNT = 43

target_met = (
    answer_correct_count
    >= TARGET_CORRECT_COUNT
)


# =========================================================
# 13. 构建最终结果表
# =========================================================

rows = []


def add(
    category,
    metric,
    value,
    denominator="",
    percentage="",
    unit="",
    note=""
):
    rows.append({
        "Category": category,
        "Metric": metric,
        "Value": value,
        "Denominator": denominator,
        "Percentage": percentage,
        "Unit": unit,
        "Note": note
    })


# ---------- Test design ----------

add(
    "Evaluation Design",
    "Answerable Questions",
    50,
    "",
    "",
    "questions"
)

add(
    "Evaluation Design",
    "Unanswerable Questions",
    10,
    "",
    "",
    "questions"
)

add(
    "Evaluation Design",
    "Total Formal Test Questions",
    60,
    "",
    "",
    "questions"
)


# ---------- Generation quality ----------

add(
    "Answer Quality",
    "Correct Answers",
    answer_correct_count,
    50,
    f"{answer_accuracy:.1f}%",
    "questions",
    "Human review against reference answers."
)

add(
    "Answer Quality",
    "Incorrect Answers",
    answer_wrong_count,
    50,
    f"{answer_wrong_count / 50 * 100:.1f}%",
    "questions"
)

add(
    "Answer Quality",
    "Success Target",
    TARGET_CORRECT_COUNT,
    50,
    "86.0%",
    "questions",
    "Predefined target corresponding to at least 85%."
)

add(
    "Answer Quality",
    "Success Target Met",
    "YES" if target_met else "NO",
    "",
    "",
    "",
    f"Actual result: {answer_correct_count}/50."
)


# ---------- Citation ----------

if not pd.isna(citation_valid_count):

    add(
        "Citation Quality",
        "Valid Citation Cases",
        citation_valid_count,
        50,
        f"{citation_accuracy:.1f}%",
        "questions"
    )

else:

    add(
        "Citation Quality",
        "Valid Citation Cases",
        "NOT AUTO-DETECTED",
        50,
        "",
        "",
        "Check the citation-review column manually."
    )


# ---------- Abstention ----------

add(
    "Abstention",
    "Correct Abstentions",
    correct_abstention_count,
    10,
    f"{abstention_accuracy:.1f}%",
    "questions"
)


# ---------- Retrieval ----------

add(
    "Retrieval",
    "Document Recall@5",
    document_recall_count,
    50,
    f"{document_recall_percent:.1f}%",
    "questions"
)

add(
    "Retrieval",
    "Exact Evidence Recall@5",
    exact_evidence_recall_count,
    50,
    f"{exact_evidence_recall_percent:.1f}%",
    "questions"
)


# ---------- Time ----------

add(
    "Efficiency",
    "RAG Mean Response Time",
    rag_mean_time,
    "",
    "",
    "seconds"
)

add(
    "Efficiency",
    "RAG Median Response Time",
    rag_median_time,
    "",
    "",
    "seconds"
)

add(
    "Efficiency",
    "Ctrl-F Mean Search Time",
    ctrlf_mean_time,
    "",
    "",
    "seconds"
)

add(
    "Efficiency",
    "Ctrl-F Median Search Time",
    ctrlf_median_time,
    "",
    "",
    "seconds"
)

add(
    "Efficiency",
    "Ctrl-F Timeouts",
    int(ctrlf_timeouts)
    if not np.isnan(ctrlf_timeouts)
    else np.nan,
    50,
    "",
    "questions"
)

add(
    "Efficiency",
    "Mean Time Saved",
    mean_time_saved,
    "",
    "",
    "seconds/question"
)

add(
    "Efficiency",
    "Overall Mean-Time Saving",
    overall_time_saving,
    "",
    f"{overall_time_saving:.2f}%"
    if not np.isnan(overall_time_saving)
    else "",
    "%"
)

add(
    "Efficiency",
    "Overall Speed-up",
    overall_speedup,
    "",
    "",
    "x"
)


# ---------- Paired ----------

add(
    "Paired Efficiency",
    "Both-correct Questions",
    int(paired_questions)
    if not np.isnan(paired_questions)
    else np.nan,
    50,
    "",
    "questions"
)

add(
    "Paired Efficiency",
    "Paired RAG Mean Time",
    paired_rag_mean,
    "",
    "",
    "seconds"
)

add(
    "Paired Efficiency",
    "Paired Ctrl-F Mean Time",
    paired_ctrlf_mean,
    "",
    "",
    "seconds"
)

add(
    "Paired Efficiency",
    "Paired Mean Time Saved",
    paired_mean_saved,
    "",
    "",
    "seconds"
)

add(
    "Paired Efficiency",
    "Paired Time Saving",
    paired_saving,
    "",
    f"{paired_saving:.2f}%"
    if not np.isnan(paired_saving)
    else "",
    "%"
)

add(
    "Paired Efficiency",
    "Paired Speed-up",
    paired_speedup,
    "",
    "",
    "x"
)


# ---------- Cost ----------

add(
    "Cost",
    "Total Online API Cost",
    total_api_cost,
    60,
    "",
    "USD"
)

add(
    "Cost",
    "Average Online API Cost per Question",
    avg_api_cost,
    "",
    "",
    "USD/question"
)

add(
    "Cost",
    "Answerable Mean API Cost",
    answerable_cost,
    "",
    "",
    "USD/question"
)

add(
    "Cost",
    "Unanswerable Mean API Cost",
    unanswerable_cost,
    "",
    "",
    "USD/question"
)

add(
    "Cost",
    "Corpus Chunks",
    int(corpus_chunks)
    if not np.isnan(corpus_chunks)
    else np.nan,
    "",
    "",
    "chunks"
)

add(
    "Cost",
    "Corpus Embedding Tokens",
    int(corpus_embedding_tokens)
    if not np.isnan(corpus_embedding_tokens)
    else np.nan,
    "",
    "",
    "tokens"
)

add(
    "Cost",
    "One-off Corpus Embedding Cost",
    corpus_embedding_cost,
    "",
    "",
    "USD"
)

add(
    "Cost",
    "Amortisation Query Volume",
    int(amortisation_volume)
    if not np.isnan(amortisation_volume)
    else np.nan,
    "",
    "",
    "queries"
)

add(
    "Cost",
    "Amortised Corpus Cost per Query",
    amortised_cost,
    "",
    "",
    "USD/query"
)

add(
    "Cost",
    "Average Cost incl. Amortised Corpus",
    total_cost_amortised,
    "",
    "",
    "USD/query"
)


# =========================================================
# 14. 保存
# =========================================================

summary = pd.DataFrame(rows)

summary.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)


# =========================================================
# 15. 打印最终报告核心数字
# =========================================================

print("\n")
print("=" * 72)
print("FINAL EXPERIMENT SUMMARY")
print("=" * 72)

print("\n--- Evaluation Set ---")
print("Answerable questions: 50")
print("Unanswerable questions: 10")
print("Total: 60")


print("\n--- Answer Quality ---")
print(
    f"Correct answers: "
    f"{answer_correct_count}/50 "
    f"= {answer_accuracy:.1f}%"
)

print(
    f"Target: ≥{TARGET_CORRECT_COUNT}/50 "
    f"(≥85%)"
)

print(
    "Target achieved:",
    "YES ✅" if target_met else "NO ❌"
)


if not pd.isna(citation_valid_count):
    print(
        f"Valid citation cases: "
        f"{citation_valid_count}/50 "
        f"= {citation_accuracy:.1f}%"
    )
else:
    print(
        "Valid citation cases: "
        "⚠️ Citation column not automatically detected"
    )


print(
    f"Correct abstentions: "
    f"{correct_abstention_count}/10 "
    f"= {abstention_accuracy:.1f}%"
)


print("\n--- Retrieval ---")

print(
    f"Document Recall@5: "
    f"{document_recall_count}/50 "
    f"= {document_recall_percent:.1f}%"
)

print(
    f"Exact Evidence Recall@5: "
    f"{exact_evidence_recall_count}/50 "
    f"= {exact_evidence_recall_percent:.1f}%"
)


print("\n--- Time Efficiency ---")

print(
    f"RAG mean response time: "
    f"{rag_mean_time:.2f} s"
)

print(
    f"RAG median response time: "
    f"{rag_median_time:.2f} s"
)

print(
    f"Ctrl-F mean search time: "
    f"{ctrlf_mean_time:.2f} s"
)

print(
    f"Ctrl-F median search time: "
    f"{ctrlf_median_time:.2f} s"
)

print(
    f"Mean time saved: "
    f"{mean_time_saved:.2f} s/question"
)

print(
    f"Overall time reduction: "
    f"{overall_time_saving:.2f}%"
)

print(
    f"Overall speed-up: "
    f"{overall_speedup:.2f}x"
)


print("\n--- Both-correct Paired Comparison ---")

print(
    f"Both correct: "
    f"{int(paired_questions)}/50"
)

print(
    f"Paired RAG mean: "
    f"{paired_rag_mean:.2f} s"
)

print(
    f"Paired Ctrl-F mean: "
    f"{paired_ctrlf_mean:.2f} s"
)

print(
    f"Paired mean saving: "
    f"{paired_mean_saved:.2f} s"
)

print(
    f"Paired time reduction: "
    f"{paired_saving:.2f}%"
)

print(
    f"Paired speed-up: "
    f"{paired_speedup:.2f}x"
)


print("\n--- Cost ---")

print(
    f"Total online API cost / 60 queries: "
    f"${total_api_cost:.6f}"
)

print(
    f"Average online API cost / question: "
    f"${avg_api_cost:.6f}"
)

print(
    f"One-off corpus embedding cost: "
    f"${corpus_embedding_cost:.6f}"
)

print(
    f"Average cost incl. amortised corpus: "
    f"${total_cost_amortised:.6f}/query"
)


print("\n")
print("=" * 72)
print("✅ 最终实验结果汇总完成")
print(f"结果文件：{OUTPUT_FILE.name}")
print("=" * 72)