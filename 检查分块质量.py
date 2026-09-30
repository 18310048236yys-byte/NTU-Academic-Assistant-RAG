import json
from pathlib import Path


CHUNK_FILE = Path("分块结果.jsonl")

total_chunks = 0
problem_chunks = 0
problem_characters = 0

examples = []


with open(CHUNK_FILE, "r", encoding="utf-8") as f:

    for line in f:

        chunk = json.loads(line)

        total_chunks += 1

        text = chunk["text"]

        count = text.count("�")

        if count > 0:

            problem_chunks += 1
            problem_characters += count

            if len(examples) < 10:
                examples.append({
                    "chunk_id": chunk["chunk_id"],
                    "document_id": chunk["document_id"],
                    "filename": chunk["filename"],
                    "page": chunk["page"],
                    "count": count,
                    "text": text[:300]
                })


print("=" * 70)
print("分块文字质量检查")
print("=" * 70)

print(f"\n总 chunks：{total_chunks}")
print(f"含乱码 chunks：{problem_chunks}")
print(f"乱码字符总数：{problem_characters}")

if total_chunks > 0:

    ratio = problem_chunks / total_chunks * 100

    print(f"受影响 chunk 比例：{ratio:.2f}%")


print("\n" + "=" * 70)

if problem_chunks == 0:

    print("✅ 没有发现乱码，可以进入 Embedding 阶段。")

else:

    print("⚠️ 发现乱码，下面显示部分例子：\n")

    for example in examples:

        print("-" * 70)

        print(f"Chunk ID：{example['chunk_id']}")
        print(f"Document：{example['document_id']}")
        print(f"文件：{example['filename']}")
        print(f"页码：{example['page']}")
        print(f"乱码数量：{example['count']}")

        print("\n文字：")
        print(example["text"])
        print()