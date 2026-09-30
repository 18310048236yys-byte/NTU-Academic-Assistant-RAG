import os
import json
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI


# ============================================================
# 1. 基本设置
# ============================================================

VECTOR_FOLDER = Path("向量库")

INDEX_FILE = VECTOR_FOLDER / "index.faiss"
METADATA_FILE = VECTOR_FOLDER / "metadata.json"

EMBEDDING_MODEL = "openai/text-embedding-3-small"

TOP_K = 5


# ============================================================
# 2. 读取 OpenRouter API Key
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
# 3. 检查向量库文件
# ============================================================

if not INDEX_FILE.exists():
    raise FileNotFoundError(
        f"找不到 FAISS 文件：{INDEX_FILE}"
    )

if not METADATA_FILE.exists():
    raise FileNotFoundError(
        f"找不到 Metadata 文件：{METADATA_FILE}"
    )


# ============================================================
# 4. 加载 FAISS
# ============================================================

index = faiss.read_index(
    str(INDEX_FILE)
)


# ============================================================
# 5. 加载 Metadata
# ============================================================

with open(
    METADATA_FILE,
    "r",
    encoding="utf-8"
) as f:

    metadata = json.load(f)


print("=" * 75)
print("NTU RAG 检索测试")
print("=" * 75)

print(f"\nFAISS 向量数量：{index.ntotal}")
print(f"Metadata 数量：{len(metadata)}")
print(f"Top-K：{TOP_K}")


if index.ntotal != len(metadata):
    raise ValueError(
        "FAISS 向量数量和 Metadata 数量不一致。"
    )


# ============================================================
# 6. 检索函数
# ============================================================

def search(query, top_k=TOP_K):

    print("\n" + "=" * 75)
    print(f"问题：{query}")
    print("=" * 75)

    # --------------------------------------------
    # 生成用户问题的 embedding
    # --------------------------------------------

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=query
    )

    query_embedding = np.array(
        [response.data[0].embedding],
        dtype="float32"
    )

    # --------------------------------------------
    # L2 Normalize
    #
    # 和建立向量库时保持一致
    # --------------------------------------------

    faiss.normalize_L2(
        query_embedding
    )

    # --------------------------------------------
    # FAISS Search
    # --------------------------------------------

    scores, indices = index.search(
        query_embedding,
        top_k
    )

    # --------------------------------------------
    # 输出结果
    # --------------------------------------------

    print("\nTop 检索结果：\n")

    for rank, (score, idx) in enumerate(
        zip(scores[0], indices[0]),
        start=1
    ):

        if idx == -1:
            continue

        item = metadata[idx]

        print("-" * 75)

        print(f"排名：{rank}")

        print(
            f"Similarity Score：{score:.4f}"
        )

        print(
            f"Document：{item['document_id']}"
        )

        print(
            f"文件：{item['filename']}"
        )

        print(
            f"PDF 页码：{item['page']}"
        )

        print(
            f"Chunk ID：{item['chunk_id']}"
        )

        print("\n内容：")

        # 为了方便查看，只打印前 800 个字符
        text_preview = item["text"][:800]

        print(text_preview)

        print()


# ============================================================
# 7. 第一轮固定测试
# ============================================================

test_questions = [

    "What percentage of PE6201 is the End-of-Course Project?",

    "How many teaching hours is one Academic Unit at NTU?",

    "What is the minimum CGPA required to graduate from a coursework programme?",

    "What sanctions may the Board of Discipline impose on a student?",

    "Who does the NTU Open Access Policy apply to?"
]


for question in test_questions:

    search(question)


# ============================================================
# 8. 手动输入问题
# ============================================================

print("\n" + "=" * 75)
print("固定测试结束")
print("=" * 75)

print(
    "\n你现在可以自己输入问题。"
)

print(
    "输入 exit 退出程序。\n"
)


while True:

    user_question = input("请输入问题：").strip()

    if user_question.lower() == "exit":
        print("检索测试结束。")
        break

    if not user_question:
        continue

    search(user_question)