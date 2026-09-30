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

# 上一步生成的分块文件
CHUNK_FILE = Path("分块结果.jsonl")

# 自动创建一个“向量库”文件夹
VECTOR_FOLDER = Path("向量库")

# 输出文件
INDEX_FILE = VECTOR_FOLDER / "index.faiss"
METADATA_FILE = VECTOR_FOLDER / "metadata.json"
STATS_FILE = VECTOR_FOLDER / "构建统计.json"

# OpenRouter 上的 Embedding 模型
EMBEDDING_MODEL = "openai/text-embedding-3-small"

# 每次发送多少个 chunk
BATCH_SIZE = 32


# ============================================================
# 2. 读取 .env 中的 OpenRouter API Key
# ============================================================

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    raise ValueError(
        "没有读取到 OPENROUTER_API_KEY。\n"
        "请确认项目根目录存在 .env 文件，并包含：\n"
        "OPENROUTER_API_KEY=你的OpenRouterKey"
    )


print("✅ OpenRouter API Key 已读取")


# ============================================================
# 3. 创建 OpenRouter 客户端
# ============================================================

client = OpenAI(
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1"
)


# ============================================================
# 4. 检查分块文件
# ============================================================

if not CHUNK_FILE.exists():
    raise FileNotFoundError(
        f"找不到文件：{CHUNK_FILE}\n"
        "请确认“分块结果.jsonl”位于项目根目录。"
    )


# ============================================================
# 5. 读取全部 chunks
# ============================================================

chunks = []

with open(CHUNK_FILE, "r", encoding="utf-8") as f:

    for line in f:

        line = line.strip()

        if not line:
            continue

        chunk = json.loads(line)

        chunks.append(chunk)


if len(chunks) == 0:
    raise ValueError("分块结果.jsonl 中没有任何 chunk。")


print("\n" + "=" * 70)
print("NTU RAG 向量库构建")
print("=" * 70)

print(f"\n读取 chunks 数量：{len(chunks)}")
print(f"Embedding 模型：{EMBEDDING_MODEL}")


# ============================================================
# 6. 提取需要生成 embedding 的文本
# ============================================================

texts = [
    chunk["text"]
    for chunk in chunks
]


# 统计我们之前记录下来的 token 数
total_chunk_tokens = sum(
    chunk.get("token_count", 0)
    for chunk in chunks
)

print(f"Corpus chunk token 总数：{total_chunk_tokens}")


# ============================================================
# 7. 批量调用 OpenRouter Embedding API
# ============================================================

all_embeddings = []

print("\n开始生成 Embeddings...\n")


for start in range(0, len(texts), BATCH_SIZE):

    end = min(
        start + BATCH_SIZE,
        len(texts)
    )

    batch_texts = texts[start:end]

    print(
        f"正在处理 chunk "
        f"{start + 1} - {end} / {len(texts)}"
    )

    try:

        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch_texts
        )

    except Exception as e:

        print("\n❌ Embedding API 调用失败")
        print(f"错误发生在 chunk {start + 1} - {end}")
        print(f"错误信息：{e}")

        raise


    # OpenRouter 返回的 embedding
    batch_embeddings = [
        item.embedding
        for item in response.data
    ]

    all_embeddings.extend(
        batch_embeddings
    )


print("\n✅ 全部 Embeddings 生成完成")


# ============================================================
# 8. 检查数量
# ============================================================

if len(all_embeddings) != len(chunks):

    raise ValueError(
        "Embedding 数量与 chunk 数量不一致。\n"
        f"chunks：{len(chunks)}\n"
        f"embeddings：{len(all_embeddings)}"
    )


print(
    f"Embedding 数量：{len(all_embeddings)}"
)


# ============================================================
# 9. 转换为 NumPy 矩阵
# ============================================================

embedding_matrix = np.array(
    all_embeddings,
    dtype="float32"
)


print(
    f"Embedding matrix shape："
    f"{embedding_matrix.shape}"
)


# ============================================================
# 10. L2 Normalize
#
# 后面使用 Inner Product 搜索。
# 向量归一化后：
#
# Inner Product ≈ Cosine Similarity
#
# ============================================================

faiss.normalize_L2(
    embedding_matrix
)


# 自动获取 embedding 维度
dimension = embedding_matrix.shape[1]

print(
    f"Embedding dimension：{dimension}"
)


# ============================================================
# 11. 创建 FAISS Index
# ============================================================

index = faiss.IndexFlatIP(
    dimension
)


# 将全部 embedding 加入 FAISS
index.add(
    embedding_matrix
)


print(
    f"✅ FAISS 向量数量：{index.ntotal}"
)


# ============================================================
# 12. 创建“向量库”文件夹
# ============================================================

VECTOR_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 13. 保存 FAISS Index
# ============================================================

faiss.write_index(
    index,
    str(INDEX_FILE)
)


print(
    f"✅ 已保存 FAISS：{INDEX_FILE}"
)


# ============================================================
# 14. 保存 Metadata
#
# FAISS 本身只保存向量。
#
# 所以需要另外保存：
# D几
# PDF文件
# 页码
# chunk内容
#
# 后面 Citation 就依赖这里。
# ============================================================

metadata = []


for chunk in chunks:

    metadata.append({

        "chunk_id":
            chunk["chunk_id"],

        "document_id":
            chunk["document_id"],

        "filename":
            chunk["filename"],

        "page":
            chunk["page"],

        "chunk_index_on_page":
            chunk["chunk_index_on_page"],

        "token_count":
            chunk["token_count"],

        "text":
            chunk["text"]
    })


with open(
    METADATA_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        metadata,
        f,
        ensure_ascii=False,
        indent=2
    )


print(
    f"✅ 已保存 Metadata：{METADATA_FILE}"
)


# ============================================================
# 15. 保存构建统计
#
# 后面写报告和计算成本的时候会用。
# ============================================================

stats = {

    "embedding_provider":
        "OpenRouter",

    "embedding_model":
        EMBEDDING_MODEL,

    "number_of_documents":
        10,

    "number_of_chunks":
        len(chunks),

    "total_chunk_tokens":
        total_chunk_tokens,

    "embedding_dimension":
        dimension,

    "chunk_size":
        500,

    "chunk_overlap":
        100
}


with open(
    STATS_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        stats,
        f,
        ensure_ascii=False,
        indent=2
    )


print(
    f"✅ 已保存构建统计：{STATS_FILE}"
)


# ============================================================
# 16. 最终检查
# ============================================================

print("\n" + "=" * 70)
print("向量库构建结果")
print("=" * 70)

print(
    f"\nCorpus 文档数量：10"
)

print(
    f"Chunk 数量：{len(chunks)}"
)

print(
    f"Embedding 数量：{len(all_embeddings)}"
)

print(
    f"FAISS 向量数量：{index.ntotal}"
)

print(
    f"Embedding dimension：{dimension}"
)

print(
    f"Embedding model：{EMBEDDING_MODEL}"
)

print("\n生成文件：")

print(
    f"1. {INDEX_FILE}"
)

print(
    f"2. {METADATA_FILE}"
)

print(
    f"3. {STATS_FILE}"
)


# ============================================================
# 17. 判断是否成功
# ============================================================

if index.ntotal == len(chunks):

    print("\n✅ 向量库构建成功！")

else:

    print("\n❌ 向量数量异常，请检查。")


print("=" * 70)