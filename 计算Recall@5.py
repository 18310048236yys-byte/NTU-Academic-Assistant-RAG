import os
import re
import json
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI


# ============================================================
# 1. 文件与模型设置
# ============================================================

TEST_FILE = Path("正式测试集.csv")
OUTPUT_FILE = Path("Recall@5结果.csv")

VECTOR_FOLDER = Path("向量库")
INDEX_FILE = VECTOR_FOLDER / "index.faiss"
METADATA_FILE = VECTOR_FOLDER / "metadata.json"

EMBEDDING_MODEL = "openai/text-embedding-3-small"

TOP_K = 5
RETRIEVAL_CANDIDATES = 10


# ============================================================
# 2. OpenRouter
# ============================================================

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    raise ValueError("没有读取到 OPENROUTER_API_KEY")

client = OpenAI(
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1"
)


# ============================================================
# 3. 加载向量库
# ============================================================

index = faiss.read_index(str(INDEX_FILE))

with open(
    METADATA_FILE,
    "r",
    encoding="utf-8"
) as f:
    metadata = json.load(f)


# ============================================================
# 4. Retrieval
#    注意：保持和冻结系统一致
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

    faiss.normalize_L2(query_embedding)

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

    # 与冻结系统一样：
    # 如果明确提到 PExxxx，优先匹配相同课程代码
    course_codes = re.findall(
        r"\bPE\d{4}\b",
        question.upper()
    )

    if course_codes:

        target_code = course_codes[0]

        for result in results:

            text_to_check = (
                result["filename"]
                + " "
                + result["text"]
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

    return results[:TOP_K]


# ============================================================
# 5. 读取正式测试集
# ============================================================

df = pd.read_csv(TEST_FILE)

# 只测 50 道 answerable
df = df[
    df["Type"] == "answerable"
].copy()


rows = []

document_hits = 0
page_hits = 0


# ============================================================
# 6. 逐题计算 Recall@5
# ============================================================

for i, row in df.iterrows():

    question_id = row["ID"]
    question = row["Question"]

    expected_document = str(
        row["Expected Document"]
    ).strip()

    expected_page = int(
        float(row["Expected Page"])
    )

    results = retrieve(question)

    top5_documents = [
        r["document_id"]
        for r in results
    ]

    top5_pages = [
        r["page"]
        for r in results
    ]

    # 文档级 Recall@5
    document_hit = (
        expected_document
        in top5_documents
    )

    # 更严格：
    # 正确 document + page 是否同时进入 Top-5
    page_hit = any(
        r["document_id"] == expected_document
        and r["page"] == expected_page
        for r in results
    )

    if document_hit:
        document_hits += 1

    if page_hit:
        page_hits += 1


    print(
        f"{question_id} | "
        f"Expected: {expected_document} p{expected_page} | "
        f"Doc Hit: {document_hit} | "
        f"Page Hit: {page_hit}"
    )


    rows.append({

        "ID":
            question_id,

        "Question":
            question,

        "Expected Document":
            expected_document,

        "Expected Page":
            expected_page,

        "Top5 Documents":
            ",".join(top5_documents),

        "Top5 Pages":
            ",".join(
                str(x)
                for x in top5_pages
            ),

        "Document Recall@5 Hit":
            "Y" if document_hit else "N",

        "Exact Evidence Recall@5 Hit":
            "Y" if page_hit else "N"
    })


# ============================================================
# 7. 保存
# ============================================================

result_df = pd.DataFrame(rows)

result_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 8. 总结
# ============================================================

total = len(df)

document_recall = (
    document_hits / total
    if total
    else 0
)

page_recall = (
    page_hits / total
    if total
    else 0
)


print("\n" + "=" * 70)

print(
    f"Document Recall@5: "
    f"{document_hits}/{total} "
    f"= {document_recall:.1%}"
)

print(
    f"Exact Evidence Recall@5: "
    f"{page_hits}/{total} "
    f"= {page_recall:.1%}"
)

print(
    f"\n结果已保存：{OUTPUT_FILE}"
)

print("=" * 70)