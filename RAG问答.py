import os
import json
import time
from pathlib import Path
import re
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
GENERATION_MODEL = "openai/gpt-4.1-mini"

TOP_K = 5

# 暂时使用的自动拒答阈值
# 之后会用 development questions 调整，而不是直接拿它做最终实验
ABSTENTION_THRESHOLD = 0.35


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
# 3. 加载 FAISS
# ============================================================

if not INDEX_FILE.exists():
    raise FileNotFoundError(
        f"找不到：{INDEX_FILE}"
    )

index = faiss.read_index(
    str(INDEX_FILE)
)


# ============================================================
# 4. 加载 Metadata
# ============================================================

if not METADATA_FILE.exists():
    raise FileNotFoundError(
        f"找不到：{METADATA_FILE}"
    )


with open(
    METADATA_FILE,
    "r",
    encoding="utf-8"
) as f:

    metadata = json.load(f)


if index.ntotal != len(metadata):

    raise ValueError(
        "FAISS 向量数量和 metadata 数量不一致。"
    )


print("=" * 75)
print("NTU Academic Assistant")
print("=" * 75)

print(f"\n向量数量：{index.ntotal}")
print(f"Generation model：{GENERATION_MODEL}")
print(f"Embedding model：{EMBEDDING_MODEL}")
print(f"Top-K：{TOP_K}")
print(f"临时 abstention threshold：{ABSTENTION_THRESHOLD}")


# ============================================================
# 5. Retrieval
# ============================================================

def retrieve(question, top_k=TOP_K):

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
            "chunk_id": item["chunk_id"],
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
# 6. 构造 Evidence
# ============================================================

def build_context(results):

    context_parts = []

    for number, result in enumerate(
        results,
        start=1
    ):

        context = f"""
[SOURCE {number}]
Document ID: {result['document_id']}
Filename: {result['filename']}
PDF Page: {result['page']}

{result['text']}
"""

        context_parts.append(context)

    return "\n".join(context_parts)


# ============================================================
# 7. 调用 GPT 生成 grounded answer
# ============================================================

def generate_answer(question, results):

    # 没有任何检索结果
    if not results:

        return (
            "I cannot answer this question based on "
            "the provided NTU documents.",
            True,
            None
        )


    top_score = results[0]["score"]


    # --------------------------------------------
    # 第一层 abstention：
    # similarity 太低直接拒答
    # --------------------------------------------

    if top_score < ABSTENTION_THRESHOLD:

        return (
            "I cannot answer this question based on "
            "the provided NTU documents.",
            True,
            None
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


Answer the question using only the retrieved evidence.
"""


    start_time = time.perf_counter()


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


    generation_time = (
        time.perf_counter() - start_time
    )


    answer = (
        response
        .choices[0]
        .message
        .content
        .strip()
    )


    abstained = answer.startswith(
        "I cannot answer this question"
    )


    # API token usage
    usage = getattr(
        response,
        "usage",
        None
    )


    usage_info = None

    if usage:

        usage_info = {
            "prompt_tokens":
                getattr(
                    usage,
                    "prompt_tokens",
                    None
                ),

            "completion_tokens":
                getattr(
                    usage,
                    "completion_tokens",
                    None
                ),

            "total_tokens":
                getattr(
                    usage,
                    "total_tokens",
                    None
                ),

            "generation_seconds":
                generation_time
        }


    return (
        answer,
        abstained,
        usage_info
    )


# ============================================================
# 8. 显示 Source
# ============================================================

def print_sources(answer, results):

    # 找出答案中真正出现过的 [SOURCE X]
    cited_numbers = re.findall(
        r"\[SOURCE\s+(\d+)\]",
        answer
    )

    cited_numbers = [
        int(number)
        for number in cited_numbers
    ]

    # 去重，同时保持原顺序
    cited_numbers = list(
        dict.fromkeys(cited_numbers)
    )

    if not cited_numbers:
        print("\n⚠️ 回答中没有检测到 Citation。")
        return

    print("\n实际引用来源：")

    for number in cited_numbers:

        # SOURCE 1 对应 results[0]
        index_number = number - 1

        if (
            index_number < 0
            or index_number >= len(results)
        ):
            continue

        result = results[index_number]

        print(
            f"[SOURCE {number}] "
            f"{result['document_id']} | "
            f"{result['filename']} | "
            f"PDF Page {result['page']} | "
            f"Score {result['score']:.4f}"
        )


# ============================================================
# 9. 完整问答函数
# ============================================================

def ask(question):

    total_start = time.perf_counter()


    # Retrieval
    results = retrieve(
        question
    )


    print("\n" + "=" * 75)
    print(f"问题：{question}")
    print("=" * 75)


    print("\n检索情况：")

    for rank, result in enumerate(
        results,
        start=1
    ):

        print(
            f"{rank}. "
            f"{result['document_id']} | "
            f"Page {result['page']} | "
            f"Score {result['score']:.4f}"
        )


    # Generation
    answer, abstained, usage = generate_answer(
        question,
        results
    )


    print("\n回答：\n")
    print(answer)


    # 如果不是拒答，显示来源
    if not abstained:
        print_sources(
            answer,
            results
        )


    total_time = (
        time.perf_counter() - total_start
    )


    print(
        f"\n总响应时间："
        f"{total_time:.2f} 秒"
    )


    if usage:

        print("\nToken usage：")

        print(
            f"Input tokens："
            f"{usage['prompt_tokens']}"
        )

        print(
            f"Output tokens："
            f"{usage['completion_tokens']}"
        )

        print(
            f"Total tokens："
            f"{usage['total_tokens']}"
        )

        print(
            f"Generation time："
            f"{usage['generation_seconds']:.2f} 秒"
        )


    if abstained:

        print(
            "\n状态：ABSTAIN"
        )

    else:

        print(
            "\n状态：ANSWERED"
        )


# ============================================================
# 10. 开始交互
# ============================================================

print("\n系统已启动。")

print(
    "输入问题开始问答；输入 exit 退出。\n"
)


while True:

    question = input(
        "请输入问题："
    ).strip()


    if question.lower() == "exit":

        print(
            "程序结束。"
        )

        break


    if not question:
        continue


    try:

        ask(
            question
        )

    except Exception as e:

        print(
            "\n❌ 出现错误："
        )

        print(e)