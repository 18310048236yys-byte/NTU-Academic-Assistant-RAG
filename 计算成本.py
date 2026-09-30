from pathlib import Path
import pandas as pd
import numpy as np
import json
import re
import tiktoken


# =========================================================
# 1. 基本设置
# =========================================================

ROOT = Path(__file__).resolve().parent

TEST_FILE = ROOT / "正式测试结果.csv"
CHUNKS_FILE = ROOT / "分块结果.jsonl"

DETAIL_OUTPUT = ROOT / "正式测试成本明细.csv"
SUMMARY_OUTPUT = ROOT / "成本汇总.csv"


# =========================================================
# 2. 模型与价格
#    价格单位：USD / 1,000,000 tokens
#    Price checked: 2026-10-01
# =========================================================

GENERATION_MODEL = "openai/gpt-4.1-mini"
EMBEDDING_MODEL = "openai/text-embedding-3-small"

GPT_INPUT_PRICE_PER_M = 0.40
GPT_OUTPUT_PRICE_PER_M = 1.60
EMBEDDING_PRICE_PER_M = 0.02

PRICE_DATE = "2026-10-01"

# 用于展示一次性 corpus embedding 成本的摊销
AMORTIZATION_QUERY_VOLUME = 1000


# =========================================================
# 3. 读取 CSV
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


df = read_csv_safely(TEST_FILE)

print("\n==============================")
print("正式测试结果读取成功")
print("==============================")

print("\n检测到的列名：")
print(list(df.columns))


# =========================================================
# 4. 自动识别列
# =========================================================

def normalise_name(name):
    return re.sub(
        r"[\s_\-\(\)\[\]/]+",
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


id_col = find_column(
    df,
    [
        "ID",
        "Question ID",
        "题号"
    ]
)

question_col = find_column(
    df,
    [
        "Question",
        "问题"
    ]
)

prompt_col = find_column(
    df,
    [
        "Prompt Tokens",
        "Input Tokens",
        "Prompt Token",
        "Input Token"
    ]
)

completion_col = find_column(
    df,
    [
        "Completion Tokens",
        "Output Tokens",
        "Completion Token",
        "Output Token"
    ]
)

total_token_col = find_column(
    df,
    [
        "Total Tokens",
        "Total Token"
    ]
)


if id_col is None:
    raise ValueError("找不到 ID 列。")

if question_col is None:
    raise ValueError("找不到 Question 列。")

if prompt_col is None:
    raise ValueError(
        "找不到 Prompt Tokens / Input Tokens 列。\n"
        "请把控制台打印出的列名截图发给我。"
    )

if completion_col is None:
    raise ValueError(
        "找不到 Completion Tokens / Output Tokens 列。\n"
        "请把控制台打印出的列名截图发给我。"
    )


print("\n==============================")
print("自动识别结果")
print("==============================")

print("ID：", id_col)
print("Question：", question_col)
print("LLM Input Tokens：", prompt_col)
print("LLM Output Tokens：", completion_col)
print("Total Tokens：", total_token_col)


# =========================================================
# 5. 只保留 Q01-Q60
# =========================================================

def clean_id(value):
    value = str(value).strip().upper()

    match = re.search(r"Q(\d+)", value)

    if match:
        return f"Q{int(match.group(1)):02d}"

    return value


def valid_test_id(value):
    match = re.fullmatch(r"Q(\d{2})", value)

    if not match:
        return False

    number = int(match.group(1))

    return 1 <= number <= 60


df["ID_clean"] = df[id_col].apply(clean_id)

df = df[
    df["ID_clean"].apply(valid_test_id)
].copy()

df = df.drop_duplicates(
    subset="ID_clean",
    keep="first"
)

df = df.sort_values(
    "ID_clean"
).reset_index(drop=True)


print(f"\n正式测试题数：{len(df)}/60")


# =========================================================
# 6. Token 数值清洗
# =========================================================

df["LLM_Input_Tokens"] = pd.to_numeric(
    df[prompt_col],
    errors="coerce"
).fillna(0)

df["LLM_Output_Tokens"] = pd.to_numeric(
    df[completion_col],
    errors="coerce"
).fillna(0)


# =========================================================
# 7. 计算每个问题的 Query Embedding Tokens
#
# text-embedding-3-small 使用 cl100k_base。
# 正式评估时每个问题都会先做 query embedding，
# 包括最终选择 abstain 的问题。
# =========================================================

encoding = tiktoken.get_encoding("cl100k_base")


def count_embedding_tokens(text):
    if pd.isna(text):
        return 0

    return len(
        encoding.encode(str(text))
    )


df["Query_Embedding_Tokens"] = df[
    question_col
].apply(
    count_embedding_tokens
)


# =========================================================
# 8. 每题成本
# =========================================================

df["Query_Embedding_Cost_USD"] = (
    df["Query_Embedding_Tokens"]
    / 1_000_000
    * EMBEDDING_PRICE_PER_M
)

df["LLM_Input_Cost_USD"] = (
    df["LLM_Input_Tokens"]
    / 1_000_000
    * GPT_INPUT_PRICE_PER_M
)

df["LLM_Output_Cost_USD"] = (
    df["LLM_Output_Tokens"]
    / 1_000_000
    * GPT_OUTPUT_PRICE_PER_M
)

df["Online_API_Cost_USD"] = (
    df["Query_Embedding_Cost_USD"]
    + df["LLM_Input_Cost_USD"]
    + df["LLM_Output_Cost_USD"]
)


# =========================================================
# 9. 计算一次性 Corpus Embedding 成本
# =========================================================

corpus_embedding_tokens = 0
chunk_count = 0

if CHUNKS_FILE.exists():

    with open(
        CHUNKS_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            item = json.loads(line)

            chunk_count += 1

            # 优先使用分块阶段已经保存的 token_count
            if "token_count" in item:

                corpus_embedding_tokens += int(
                    item["token_count"]
                )

            else:
                # 如果没有 token_count，
                # 则重新对 chunk text 计数
                text = item.get(
                    "text",
                    ""
                )

                corpus_embedding_tokens += len(
                    encoding.encode(text)
                )

else:

    print(
        "\n⚠️ 找不到 分块结果.jsonl，"
        "将无法计算一次性 corpus embedding 成本。"
    )


corpus_embedding_cost = (
    corpus_embedding_tokens
    / 1_000_000
    * EMBEDDING_PRICE_PER_M
)

amortized_corpus_cost_per_query = (
    corpus_embedding_cost
    / AMORTIZATION_QUERY_VOLUME
)


# =========================================================
# 10. 正式测试成本统计
# =========================================================

test_query_count = len(df)

total_query_embedding_tokens = int(
    df["Query_Embedding_Tokens"].sum()
)

total_llm_input_tokens = int(
    df["LLM_Input_Tokens"].sum()
)

total_llm_output_tokens = int(
    df["LLM_Output_Tokens"].sum()
)

total_online_cost = df[
    "Online_API_Cost_USD"
].sum()

average_online_cost = (
    total_online_cost
    / test_query_count
    if test_query_count > 0
    else np.nan
)

average_cost_with_amortized_corpus = (
    average_online_cost
    + amortized_corpus_cost_per_query
)


# =========================================================
# 11. 可回答 / 不可回答分开统计
# =========================================================

type_col = find_column(
    df,
    [
        "Type",
        "Question Type"
    ]
)

answerable_cost = np.nan
unanswerable_cost = np.nan

if type_col is not None:

    type_clean = (
        df[type_col]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    answerable = df[
        type_clean == "answerable"
    ]

    unanswerable = df[
        type_clean == "unanswerable"
    ]

    if len(answerable) > 0:

        answerable_cost = answerable[
            "Online_API_Cost_USD"
        ].mean()

    if len(unanswerable) > 0:

        unanswerable_cost = unanswerable[
            "Online_API_Cost_USD"
        ].mean()


# =========================================================
# 12. 保存逐题成本
# =========================================================

detail_columns = [
    "ID_clean",
    question_col,
    "Query_Embedding_Tokens",
    "LLM_Input_Tokens",
    "LLM_Output_Tokens",
    "Query_Embedding_Cost_USD",
    "LLM_Input_Cost_USD",
    "LLM_Output_Cost_USD",
    "Online_API_Cost_USD"
]

if type_col is not None:
    detail_columns.insert(
        2,
        type_col
    )


detail = df[
    detail_columns
].copy()

detail = detail.rename(
    columns={
        "ID_clean": "ID"
    }
)

detail.to_csv(
    DETAIL_OUTPUT,
    index=False,
    encoding="utf-8-sig"
)


# =========================================================
# 13. 成本汇总表
# =========================================================

summary_rows = [

    [
        "Pricing Date",
        PRICE_DATE,
        ""
    ],

    [
        "Generation Model",
        GENERATION_MODEL,
        ""
    ],

    [
        "Embedding Model",
        EMBEDDING_MODEL,
        ""
    ],

    [
        "GPT Input Price",
        GPT_INPUT_PRICE_PER_M,
        "USD / 1M tokens"
    ],

    [
        "GPT Output Price",
        GPT_OUTPUT_PRICE_PER_M,
        "USD / 1M tokens"
    ],

    [
        "Embedding Price",
        EMBEDDING_PRICE_PER_M,
        "USD / 1M tokens"
    ],

    [
        "Formal Test Queries",
        test_query_count,
        "queries"
    ],

    [
        "Formal Test Query Embedding Tokens",
        total_query_embedding_tokens,
        "tokens"
    ],

    [
        "Formal Test LLM Input Tokens",
        total_llm_input_tokens,
        "tokens"
    ],

    [
        "Formal Test LLM Output Tokens",
        total_llm_output_tokens,
        "tokens"
    ],

    [
        "Total Online API Cost",
        total_online_cost,
        "USD"
    ],

    [
        "Average Online API Cost per Question",
        average_online_cost,
        "USD / question"
    ],

    [
        "Answerable Mean API Cost",
        answerable_cost,
        "USD / question"
    ],

    [
        "Unanswerable Mean API Cost",
        unanswerable_cost,
        "USD / question"
    ],

    [
        "Corpus Chunks",
        chunk_count,
        "chunks"
    ],

    [
        "Corpus Embedding Tokens",
        corpus_embedding_tokens,
        "tokens"
    ],

    [
        "One-off Corpus Embedding Cost",
        corpus_embedding_cost,
        "USD"
    ],

    [
        "Amortisation Query Volume",
        AMORTIZATION_QUERY_VOLUME,
        "queries"
    ],

    [
        "Amortised Corpus Cost per Query",
        amortized_corpus_cost_per_query,
        "USD / query"
    ],

    [
        "Average Cost incl. Amortised Corpus",
        average_cost_with_amortized_corpus,
        "USD / query"
    ]
]


summary = pd.DataFrame(
    summary_rows,
    columns=[
        "Metric",
        "Value",
        "Unit"
    ]
)


summary.to_csv(
    SUMMARY_OUTPUT,
    index=False,
    encoding="utf-8-sig"
)


# =========================================================
# 14. 打印最终结果
# =========================================================

print("\n")
print("=" * 70)
print("RAG COST ANALYSIS")
print("=" * 70)

print(
    f"\n价格日期：{PRICE_DATE}"
)

print(
    f"Generation：{GENERATION_MODEL}"
)

print(
    f"Embedding：{EMBEDDING_MODEL}"
)

print("\n--- 正式测试 Token Usage ---")

print(
    f"问题数：{test_query_count}"
)

print(
    f"Query embedding tokens："
    f"{total_query_embedding_tokens:,}"
)

print(
    f"LLM input tokens："
    f"{total_llm_input_tokens:,}"
)

print(
    f"LLM output tokens："
    f"{total_llm_output_tokens:,}"
)


print("\n--- Online API Cost ---")

print(
    f"60题总在线 API 成本："
    f"${total_online_cost:.6f}"
)

print(
    f"平均每题在线 API 成本："
    f"${average_online_cost:.6f}"
)


if not np.isnan(answerable_cost):

    print(
        f"可回答问题平均成本："
        f"${answerable_cost:.6f}"
    )


if not np.isnan(unanswerable_cost):

    print(
        f"不可回答问题平均成本："
        f"${unanswerable_cost:.6f}"
    )


print("\n--- Corpus Embedding ---")

print(
    f"Chunk 数：{chunk_count}"
)

print(
    f"Corpus embedding tokens："
    f"{corpus_embedding_tokens:,}"
)

print(
    f"一次性 corpus embedding 成本："
    f"${corpus_embedding_cost:.6f}"
)


print(
    f"\n若按 {AMORTIZATION_QUERY_VOLUME:,} "
    f"次查询摊销："
)

print(
    f"每题 corpus embedding 摊销成本："
    f"${amortized_corpus_cost_per_query:.8f}"
)

print(
    f"平均每题总成本（含摊销）："
    f"${average_cost_with_amortized_corpus:.6f}"
)


print("\n--- 成本范围说明 ---")

print(
    "包含：query embedding + "
    "GPT input tokens + GPT output tokens"
)

print(
    "一次性 corpus embedding 单独报告。"
)

print(
    "不包含：人工劳动、电脑电费、"
    "本地 FAISS、Streamlit 本地运行和 hosting 成本。"
)


print("\n")
print("=" * 70)
print("✅ 成本计算完成")
print(f"逐题成本：{DETAIL_OUTPUT.name}")
print(f"汇总结果：{SUMMARY_OUTPUT.name}")
print("=" * 70)