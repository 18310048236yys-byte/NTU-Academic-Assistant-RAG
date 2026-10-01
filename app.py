"""
NTU Academic Assistant - Streamlit User Interface

Purpose:
    Provides the interactive web interface for the NTU Academic Assistant
    RAG system. Users can submit natural-language questions and receive
    evidence-grounded answers with supporting source information.

Main workflow:
    1. Accept a question from the user through Streamlit.
    2. Pass the question to the RAG question-answering pipeline.
    3. Display the generated answer.
    4. Display supporting document/page citations when available.
    5. Show an abstention response when the corpus does not contain
       sufficient evidence.
    6. Display relevant runtime information such as response time and
       token usage where available.

Inputs:
    - Natural-language user question.
    - Existing FAISS vector index and document metadata.
    - OpenRouter API credentials loaded from the local environment.

Outputs:
    - Evidence-grounded answer or abstention message.
    - Supporting source information.
    - Retrieval / runtime information shown in the Streamlit interface.

Dependencies:
    - Streamlit
    - RAG问答.py
    - Local FAISS index and metadata
    - OpenRouter API

Security:
    API credentials are loaded from the local .env file and must not be
    hard-coded or committed to the GitHub repository.
"""



import os
import json
import re
import time
from pathlib import Path

import faiss
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI


# ============================================================
# 基本设置
# ============================================================

VECTOR_FOLDER = Path("向量库")

INDEX_FILE = VECTOR_FOLDER / "index.faiss"
METADATA_FILE = VECTOR_FOLDER / "metadata.json"

EMBEDDING_MODEL = "openai/text-embedding-3-small"
GENERATION_MODEL = "openai/gpt-4.1-mini"

TOP_K = 5
ABSTENTION_THRESHOLD = 0.35


# ============================================================
# 页面设置
# ============================================================

st.set_page_config(
    page_title="NTU Academic Assistant",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 NTU Academic Assistant")

st.caption(
    "RAG-based question answering for selected NTU course "
    "and academic-policy documents."
)

st.info(
    "Answers are generated only from the selected NTU document corpus. "
    "Please refer to official NTU sources for authoritative information."
)


# ============================================================
# API
# ============================================================

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    st.error("OPENROUTER_API_KEY was not found.")
    st.stop()


client = OpenAI(
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1"
)


# ============================================================
# 加载向量库
# ============================================================

@st.cache_resource
def load_vector_store():

    index = faiss.read_index(
        str(INDEX_FILE)
    )

    with open(
        METADATA_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        metadata = json.load(f)

    return index, metadata


index, metadata = load_vector_store()


# ============================================================
# Retrieval
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
# 构造 Context
# ============================================================

def build_context(results):

    context_parts = []

    for number, result in enumerate(
        results,
        start=1
    ):

        context_parts.append(
            f"""
[SOURCE {number}]
Document ID: {result['document_id']}
Filename: {result['filename']}
PDF Page: {result['page']}

{result['text']}
"""
        )

    return "\n".join(context_parts)


# ============================================================
# 生成回答
# ============================================================

def generate_answer(question, results):

    if not results:
        return (
            "I cannot answer this question based on "
            "the provided NTU documents.",
            True,
            None
        )

    top_score = results[0]["score"]

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


Answer using only the evidence above.
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

    elapsed = time.perf_counter() - start_time

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

    usage = getattr(response, "usage", None)

    usage_info = None

    if usage:
        usage_info = {
            "prompt_tokens": getattr(
                usage,
                "prompt_tokens",
                None
            ),
            "completion_tokens": getattr(
                usage,
                "completion_tokens",
                None
            ),
            "total_tokens": getattr(
                usage,
                "total_tokens",
                None
            ),
            "generation_seconds": elapsed
        }

    return answer, abstained, usage_info


# ============================================================
# 找出实际引用
# ============================================================

def get_cited_sources(answer, results):

    numbers = re.findall(
        r"\[SOURCE\s+(\d+)\]",
        answer
    )

    numbers = list(
        dict.fromkeys(
            int(number)
            for number in numbers
        )
    )

    cited = []

    for number in numbers:

        idx = number - 1

        if 0 <= idx < len(results):

            result = results[idx].copy()

            result["source_number"] = number

            cited.append(result)

    return cited


# ============================================================
# 用户输入
# ============================================================

question = st.text_input(
    "Ask a question",
    placeholder=(
        "e.g. What percentage of PE6201 is "
        "the End-of-Course Project?"
    )
)


if st.button(
    "Ask",
    type="primary"
):

    if not question.strip():

        st.warning(
            "Please enter a question."
        )

    else:

        total_start = time.perf_counter()

        with st.spinner(
            "Searching NTU documents..."
        ):

            results = retrieve(
                question
            )

            answer, abstained, usage = generate_answer(
                question,
                results
            )

        total_time = (
            time.perf_counter()
            - total_start
        )


        # ====================================================
        # 回答
        # ====================================================

        st.subheader("Answer")

        if abstained:

            st.warning(answer)

        else:

            st.write(answer)


        # ====================================================
        # Citation
        # ====================================================

        if not abstained:

            cited_sources = get_cited_sources(
                answer,
                results
            )

            st.subheader("Sources")

            if cited_sources:

                for source in cited_sources:

                    st.markdown(
                        f"""
**[SOURCE {source['source_number']}]**
{source['document_id']}  
`{source['filename']}`  
PDF Page {source['page']}
"""
                    )

            else:

                st.warning(
                    "No citation was detected in the answer."
                )


        # ====================================================
        # 技术信息
        # ====================================================

        with st.expander(
            "Retrieval details"
        ):

            for rank, result in enumerate(
                results,
                start=1
            ):

                st.write(
                    f"{rank}. "
                    f"{result['document_id']} | "
                    f"Page {result['page']} | "
                    f"Score {result['score']:.4f}"
                )

        st.caption(
            f"Total response time: "
            f"{total_time:.2f} seconds"
        )

        if usage:

            with st.expander(
                "Token usage"
            ):

                st.write(
                    f"Input tokens: "
                    f"{usage['prompt_tokens']}"
                )

                st.write(
                    f"Output tokens: "
                    f"{usage['completion_tokens']}"
                )

                st.write(
                    f"Total tokens: "
                    f"{usage['total_tokens']}"
                )