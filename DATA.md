# Data Documentation

## 1. Overview

The NTU Academic Assistant uses a small, curated document corpus containing official or course-provided NTU academic information.

The corpus was designed to support questions about:

- course information;
- assessment requirements and timelines;
- programme information;
- academic policies;
- academic integrity;
- graduate academic regulations.

The final corpus contains **10 PDF documents**.

After text extraction and chunking, the corpus produced:

- **10 source documents**
- **72 text chunks**
- **26,269 embedding tokens**

No personal student records or confidential personal information are included in the corpus.

---

## 2. Source Documents

| ID | Document | Type | Purpose in the Corpus |
|---|---|---|---|
| D1 | PE6201 Course Outline | Course document | Provides PE6201 course structure, learning outcomes, assessments and course requirements |
| D2 | PE6201 Assessment Timeline | Course document | Provides PE6201 assessment dates, submission requirements and project milestones |
| D3 | PE6202 Course Outline | Course document | Provides PE6202 course information, schedule and assessment details |
| D4 | PE6203 Course Outline | Course document | Provides PE6203 course information, learning outcomes, schedule and assessment details |
| D5 | MSc Enterprise AI Trimester 1 Course Synopsis | Programme document | Provides programme-level course information and course relationships |
| D6 | NTU Statute 6 — Student Discipline | University policy | Provides information related to student discipline and university regulations |
| D7 | NTU Open Access Policy | University policy | Provides information about NTU open-access requirements |
| D8 | NTU Anti-Plagiarism Guidance | Academic guidance | Provides guidance on plagiarism detection and interpretation |
| D9 | NTU Academic Integrity | Academic guidance | Provides definitions and examples related to academic integrity and plagiarism |
| D10 | NTU Graduate Academic Handbook | Graduate academic handbook | Provides graduate academic regulations, credit transfer, course registration, grading and academic procedures |

These documents were selected because they represent the main information categories required by the prototype: course information, assessment information, programme information and academic-policy information.

---

## 3. Data Selection

The corpus was intentionally limited to a defined set of NTU-related documents so that system outputs could be evaluated against known source material.

The document set supports questions such as:

- What are the requirements for a course?
- When is an assessment due?
- What is the assessment structure?
- What does an academic policy state?
- What are the relevant graduate academic regulations?
- What information is explicitly available in the provided NTU documents?

Questions requiring information outside this corpus are handled through the system's abstention mechanism.

---

## 4. Data Processing

The PDF documents are processed using the following pipeline:

```text
PDF Documents
    ↓
Page-level Text Extraction
    ↓
Token-based Chunking
    ↓
Metadata Preservation
    ↓
Embedding Generation
    ↓
FAISS Vector Index
```

PDF text is extracted using:

```text
pypdf
```

Documents are processed page by page so that source-page information can be retained for citations.

---

## 5. Chunking Strategy

The extracted text is divided into overlapping chunks using:

```text
Chunk size: 500 tokens
Chunk overlap: 100 tokens
```

The final corpus contains:

```text
72 chunks
```

Each chunk stores metadata including:

- chunk ID;
- document ID;
- source filename;
- PDF page number;
- chunk number;
- token count;
- extracted text.

The overlap helps preserve information that may otherwise be split at chunk boundaries.

Chunks are kept within individual PDF pages to maintain page-level traceability.

---

## 6. Embedding Data

Each chunk is converted into an embedding using:

```text
openai/text-embedding-3-small
```

The corpus contains approximately:

```text
26,269 embedding tokens
```

The embedding vectors are converted to `float32`, L2-normalized and stored in a local FAISS `IndexFlatIP` index.

The resulting vector index is used for semantic retrieval.

For each query, the system retrieves the Top-5 most similar chunks.

---

## 7. Data Privacy and Responsible Use

The prototype corpus excludes personal and confidential student information.

The system is designed for academic-information retrieval rather than retrieval of personal records.

Examples of information intentionally not included include:

- student matriculation numbers;
- personal academic advisers;
- scholarship application status;
- individual academic records;
- other private student information.

Questions requiring such information should therefore trigger abstention rather than unsupported generation.

API credentials are stored separately in a local `.env` file and are not included in the repository.

---

## 8. Source Document Distribution

The source PDFs are not included in the public GitHub repository.

This is intentional because some documents are course-provided or may have distribution restrictions.

The repository instead provides this data manifest so that the corpus composition and processing method are transparent.

Where required for assessment, the original source documents can be provided through the authorised course submission channel.

---

## 9. Reproducibility

The main data-processing scripts included in the repository allow the corpus to be rebuilt when the source documents are available.

The processing workflow includes:

```text
读取文档.py
    ↓
分块文档.py
    ↓
检查分块质量.py
    ↓
建立向量库.py
```

The generated data artefacts include:

```text
分块结果.jsonl
向量库/index.faiss
向量库/metadata.json
向量库/构建统计.json
```

The vector-index directory is excluded from the public repository because it can be regenerated from the source corpus.

---

## 10. Data Scope

The corpus represents a controlled prototype dataset rather than the complete set of NTU academic information.

This controlled scope makes it possible to:

- trace answers back to known documents;
- evaluate retrieval systematically;
- test citation behaviour;
- evaluate abstention on information outside the corpus;
- compare RAG retrieval with manual document search.

The system is therefore evaluated according to whether it correctly uses the information contained in this defined corpus.