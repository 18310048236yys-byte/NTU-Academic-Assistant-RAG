import os
import re
import time
import json
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI


# ============================================================
# 1. 基本设置
# ============================================================

DEV_FILE = Path("正式测试集.csv")
RESULT_FILE = Path("正式测试结果.csv")

VECTOR_FOLDER = Path("向量库")
INDEX_FILE = VECTOR_FOLDER / "index.faiss"
METADATA_FILE = VECTOR_FOLDER / "metadata.json"

EMBEDDING_MODEL = "openai/text-embedding-3-small"
GENERATION_MODEL = "openai/gpt-4.1-mini"

TOP_K = 5
ABSTENTION_THRESHOLD = 0.35


# ============================================================
# 2. OpenRouter API
# ============================================================

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    raise ValueError(
        "没有读取到 OPENROUTER_API_KEY，请检查 .env 文件。"
    )

client = OpenAI(
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1"
)


# ============================================================
# 3. 加载 FAISS 和 Metadata
# ============================================================

index = faiss.read_index(
    str(INDEX_FILE)
)

with open(
    METADATA_FILE,
    "r",
    encoding="utf-8"
) as f:

    metadata = json.load(f)


if index.ntotal != len(metadata):
    raise ValueError(
        "FAISS 向量数量和 Metadata 数量不一致。"
    )


# ============================================================
# 4. Retrieval
# ============================================================

def retrieve(question):

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=question
    )

    query_embedding = np.array(
        [response.data[0].embedding],
        dtype="float32"
    )

    faiss.normalize_L2(
        query_embedding
    )

    RETRIEVAL_CANDIDATES = 10

    scores, indices = index.search(
        query_embedding,
        RETRIEVAL_CANDIDATES
    )

    results = []

    for score, idx in zip(
        scores[0],
        indices[0]
    ):

        if idx == -1:
            continue

        item = metadata[idx]

        results.append({
            "score": float(score),
            "document_id": item["document_id"],
            "filename": item["filename"],
            "page": item["page"],
            "text": item["text"]
        })
    # 如果问题明确提到了课程代码，
    # 优先保留包含相同课程代码的证据
    course_codes = re.findall(
        r"\bPE\d{4}\b",
        question.upper()
    )

    if course_codes:

        target_code = course_codes[0]

        for result in results:

            text_to_check = (
                    result["filename"] + " " + result["text"]
            ).upper()

            if target_code in text_to_check:
                result["course_match"] = 1
            else:
                result["course_match"] = 0

        results.sort(
            key=lambda x: (
                x["course_match"],
                x["score"]
            ),
            reverse=True
        )

    # 最终仍然只返回 Top 5
    return results[:TOP_K]


    return results


# ============================================================
# 5. Context
# ============================================================

def build_context(results):

    parts = []

    for number, result in enumerate(
        results,
        start=1
    ):

        parts.append(
            f"""
[SOURCE {number}]
Document ID: {result['document_id']}
Filename: {result['filename']}
PDF Page: {result['page']}

{result['text']}
"""
        )

    return "\n".join(parts)


# ============================================================
# 6. 生成回答
# ============================================================

def generate_answer(question, results):

    if not results:
        return (
            "I cannot answer this question based on the provided NTU documents.",
            "ABSTAIN",
            "",
            0,
            0,
            0
        )

    top_score = results[0]["score"]

    # similarity threshold 直接拒答
    if top_score < ABSTENTION_THRESHOLD:

        return (
            "I cannot answer this question based on the provided NTU documents.",
            "ABSTAIN",
            "",
            0,
            0,
            0
        )

    context = build_context(results)

    system_prompt = """
    You are the NTU Academic Assistant.

    Answer questions ONLY using the retrieved NTU document evidence.

    Rules:

    1. Do not use outside knowledge.
    2. Do not invent facts.
    3. Do not infer facts about one course, policy, or programme
       from information about another course, policy, or programme.
    4. If the question asks about a specific course such as PE6201,
       PE6202 or PE6203, the answer must be supported by evidence
       that explicitly refers to that course.
    5. If the retrieved evidence does not directly support the answer,
       respond exactly:

       I cannot answer this question based on the provided NTU documents.

    6. Cite evidence using [SOURCE 1], [SOURCE 2], etc.
    7. Cite only sources that directly support the answer.
    8. For a simple factual question, use ONE direct source whenever
       one source is sufficient.
    9. Do not add additional citations merely because other documents
       contain related information.
    10. Answer in the same language as the user's question.
    11. Keep the answer concise.
    """

    user_prompt = f"""
QUESTION:

{question}

RETRIEVED EVIDENCE:

{context}

Answer using only the evidence above.
"""

    response = client.chat.completions.create(
        model=GENERATION_MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        temperature=0
    )

    answer = (
        response
        .choices[0]
        .message
        .content
        .strip()
    )

    if answer.startswith(
        "I cannot answer this question"
    ):
        status = "ABSTAIN"
    else:
        status = "ANSWERED"

    # 提取 Citation
    cited_numbers = re.findall(
        r"\[SOURCE\s+(\d+)\]",
        answer
    )

    cited_documents = []

    for number in cited_numbers:

        idx = int(number) - 1

        if 0 <= idx < len(results):

            doc = results[idx]["document_id"]

            if doc not in cited_documents:
                cited_documents.append(doc)

    cited_documents_text = ",".join(
        cited_documents
    )

    # Token usage
    usage = getattr(
        response,
        "usage",
        None
    )

    if usage:

        input_tokens = getattr(
            usage,
            "prompt_tokens",
            0
        ) or 0

        output_tokens = getattr(
            usage,
            "completion_tokens",
            0
        ) or 0

        total_tokens = getattr(
            usage,
            "total_tokens",
            0
        ) or 0

    else:

        input_tokens = 0
        output_tokens = 0
        total_tokens = 0

    return (
        answer,
        status,
        cited_documents_text,
        input_tokens,
        output_tokens,
        total_tokens
    )


# ============================================================
# 7. 读取开发集
# ============================================================

dev_df = pd.read_csv(
    DEV_FILE
)

print("=" * 70)
print("开始运行开发集")
print("=" * 70)

print(
    f"\n开发题数量：{len(dev_df)}"
)

print(
    f"当前 Abstention Threshold："
    f"{ABSTENTION_THRESHOLD}\n"
)


# ============================================================
# 8. 逐题运行
# ============================================================

output_rows = []


for row_number, row in dev_df.iterrows():

    question_id = row["ID"]
    question = row["Question"]

    print("-" * 70)

    print(
        f"[{row_number + 1}/{len(dev_df)}] "
        f"{question_id}"
    )

    print(
        f"问题：{question}"
    )


    start_time = time.perf_counter()


    # Retrieval
    results = retrieve(
        question
    )


    top1_score = (
        results[0]["score"]
        if results
        else 0
    )

    top1_document = (
        results[0]["document_id"]
        if results
        else ""
    )

    top1_page = (
        results[0]["page"]
        if results
        else ""
    )


    # Generation
    (
        answer,
        predicted_status,
        cited_documents,
        input_tokens,
        output_tokens,
        total_tokens
    ) = generate_answer(
        question,
        results
    )


    response_seconds = (
        time.perf_counter()
        - start_time
    )


    print(
        f"Top-1：{top1_document}"
    )

    print(
        f"Score：{top1_score:.4f}"
    )

    print(
        f"状态：{predicted_status}"
    )

    print(
        f"回答：{answer}"
    )

    print()


    output_rows.append({

        "ID":
            question_id,

        "Question":
            question,

        "Type":
            row["Type"],

        "Reference Answer":
    row["Reference Answer"],

"Expected Document":
    row.get(
        "Expected Document",
        ""
    ),

"Expected Page":
    row.get(
        "Expected Page",
        ""
    ),

        "Top1 Score":
            round(
                top1_score,
                4
            ),

        "Top1 Document":
            top1_document,

        "Top1 Page":
            top1_page,

        "Predicted Status":
            predicted_status,

        "Model Answer":
            answer,

        "Cited Documents":
            cited_documents,

        "Response Seconds":
            round(
                response_seconds,
                2
            ),

        "Input Tokens":
            input_tokens,

        "Output Tokens":
            output_tokens,

        "Total Tokens":
            total_tokens
    })


# ============================================================
# 9. 保存结果
# ============================================================

result_df = pd.DataFrame(
    output_rows
)

result_df.to_csv(
    RESULT_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 10. 输出基本统计
# ============================================================

print("\n" + "=" * 70)
print("正式测试集运行完成")
print("=" * 70)

print(
    f"\n结果文件：{RESULT_FILE}"
)


answerable = result_df[
    result_df["Type"] == "answerable"
]

unanswerable = result_df[
    result_df["Type"] == "unanswerable"
]


answerable_answered = (
    answerable["Predicted Status"]
    == "ANSWERED"
).sum()


unanswerable_abstained = (
    unanswerable["Predicted Status"]
    == "ABSTAIN"
).sum()


print(
    f"\n可回答问题成功回答："
    f"{answerable_answered}/"
    f"{len(answerable)}"
)

print(
    f"不可回答问题成功拒答："
    f"{unanswerable_abstained}/"
    f"{len(unanswerable)}"
)


print("\n可回答问题 Top-1 Score：")

print(
    answerable[
        ["ID", "Top1 Score"]
    ].to_string(
        index=False
    )
)


print("\n不可回答问题 Top-1 Score：")

print(
    unanswerable[
        ["ID", "Top1 Score"]
    ].to_string(
        index=False
    )
)


print("\n✅ 正式测试集测试完成")