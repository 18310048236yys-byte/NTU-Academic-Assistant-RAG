from pathlib import Path
from pypdf import PdfReader
import re


# 你的PDF文件所在的文件夹
PDF_FOLDER = Path("所需文件")


def get_document_number(path):
    """
    从 D1、D2、...、D10 中提取数字，
    这样打印时会按照 D1-D10 排序。
    """
    match = re.match(r"D(\d+)", path.name)

    if match:
        return int(match.group(1))

    return 999


# 找到“所需文件”文件夹中所有以 D 开头的 PDF
pdf_files = list(PDF_FOLDER.glob("D*.pdf"))

# 按 D1、D2、...、D10 排序
pdf_files.sort(key=get_document_number)


print("=" * 60)
print("NTU RAG 文档读取测试")
print("=" * 60)

print(f"\n找到 PDF 数量：{len(pdf_files)}\n")


for pdf_path in pdf_files:

    print("-" * 60)
    print(f"正在读取：{pdf_path.name}")

    try:
        reader = PdfReader(pdf_path)

        page_count = len(reader.pages)

        total_characters = 0
        empty_pages = []

        first_text = ""

        for page_number, page in enumerate(reader.pages, start=1):

            text = page.extract_text() or ""

            total_characters += len(text)

            if not text.strip():
                empty_pages.append(page_number)

            # 保存第一页部分文字用于检查
            if page_number == 1:
                first_text = text[:300]

        print(f"页数：{page_count}")
        print(f"提取字符数：{total_characters}")

        if empty_pages:
            print(f"⚠️ 没有提取到文字的页面：{empty_pages}")
        else:
            print("✅ 所有页面均提取到文字")

        print("\n第一页文字预览：")
        print(first_text.replace("\n", " "))

        print("\n✅ 读取成功")

    except Exception as e:

        print("❌ 读取失败")
        print(f"错误信息：{e}")


print("\n" + "=" * 60)

if len(pdf_files) == 10:
    print("✅ 成功找到全部 10 份 corpus 文档")
else:
    print(f"⚠️ 应该有 10 份 PDF，但目前找到 {len(pdf_files)} 份")

print("=" * 60)