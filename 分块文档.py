from pathlib import Path
from pypdf import PdfReader
import tiktoken
import json
import re


# =========================
# 基本设置
# =========================

PDF_FOLDER = Path("所需文件")

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

OUTPUT_FILE = Path("分块结果.jsonl")

# OpenAI embedding 模型使用的 tokenizer
encoding = tiktoken.get_encoding("cl100k_base")


def get_document_number(path):
    """
    从 D1、D2、...、D10 中提取编号，
    用于正确排序文件。
    """
    match = re.match(r"D(\d+)", path.name)

    if match:
        return int(match.group(1))

    return 999


def get_document_id(filename):
    """
    从文件名提取 D1、D2、...、D10。
    """
    match = re.match(r"(D\d+)", filename)

    if match:
        return match.group(1)

    return "UNKNOWN"


def split_text_into_chunks(text):
    """
    将文字按照 500 tokens 切分，
    相邻 chunk 保留 100 tokens overlap。
    """

    tokens = encoding.encode(text)

    chunks = []

    start = 0
    chunk_index = 1

    while start < len(tokens):

        end = start + CHUNK_SIZE

        chunk_tokens = tokens[start:end]

        chunk_text = encoding.decode(chunk_tokens)

        chunks.append({
            "chunk_index": chunk_index,
            "text": chunk_text,
            "token_count": len(chunk_tokens)
        })

        # 已经到达文本结尾
        if end >= len(tokens):
            break

        start = end - CHUNK_OVERLAP

        chunk_index += 1

    return chunks


# =========================
# 找到全部 PDF
# =========================

pdf_files = list(PDF_FOLDER.glob("D*.pdf"))
pdf_files.sort(key=get_document_number)


print("=" * 70)
print("NTU RAG 文档分块")
print("=" * 70)

print(f"\n找到 PDF 数量：{len(pdf_files)}")
print(f"Chunk size：{CHUNK_SIZE} tokens")
print(f"Overlap：{CHUNK_OVERLAP} tokens\n")


all_chunks = []

global_chunk_id = 1


# =========================
# 读取并分块
# =========================

for pdf_path in pdf_files:

    print("-" * 70)
    print(f"正在处理：{pdf_path.name}")

    document_id = get_document_id(pdf_path.name)

    reader = PdfReader(pdf_path)

    document_chunk_count = 0

    for page_number, page in enumerate(reader.pages, start=1):

        text = page.extract_text() or ""

        # 空白页直接跳过
        if not text.strip():
            continue

        page_chunks = split_text_into_chunks(text)

        for chunk in page_chunks:

            chunk_data = {
                "chunk_id": global_chunk_id,
                "document_id": document_id,
                "filename": pdf_path.name,
                "page": page_number,
                "chunk_index_on_page": chunk["chunk_index"],
                "token_count": chunk["token_count"],
                "text": chunk["text"]
            }

            all_chunks.append(chunk_data)

            global_chunk_id += 1
            document_chunk_count += 1

    print(f"✅ 生成 chunks：{document_chunk_count}")


# =========================
# 保存结果
# =========================

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:

    for chunk in all_chunks:

        f.write(
            json.dumps(
                chunk,
                ensure_ascii=False
            )
            + "\n"
        )


# =========================
# 输出统计
# =========================

print("\n" + "=" * 70)

print(f"✅ 全部文档处理完成")
print(f"✅ 总 chunks 数量：{len(all_chunks)}")
print(f"✅ 已保存至：{OUTPUT_FILE}")

print("=" * 70)


# =========================
# 显示前三个 chunk
# =========================

print("\n前 3 个 chunk 示例：\n")

for chunk in all_chunks[:3]:

    print("-" * 70)

    print(f"Chunk ID：{chunk['chunk_id']}")
    print(f"Document：{chunk['document_id']}")
    print(f"文件：{chunk['filename']}")
    print(f"PDF 页码：{chunk['page']}")
    print(f"Token 数：{chunk['token_count']}")

    print("\n内容：")

    print(chunk["text"][:500])

    print()