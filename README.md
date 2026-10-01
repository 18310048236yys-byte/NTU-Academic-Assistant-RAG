## Documentation

- [Product Documentation](PRODUCT.md)
- [Data Documentation](DATA.md)
- [Evaluation Documentation](EVALS.md)
# NTU Academic Assistant

## A RAG-based Question Answering System for Course and Academic Information

This project was developed for the PE6201 Emerging AI Technologies End-of-Course Project at Nanyang Technological University (NTU).

The project implements a Retrieval-Augmented Generation (RAG) system that helps students retrieve information from NTU course documents, programme materials, academic handbooks, and university policy documents.

The system retrieves relevant evidence from a local document corpus, generates an answer grounded in the retrieved evidence, provides source citations, and abstains when the available documents do not contain sufficient information.

---

## 1. Problem

NTU students often need to search across multiple PDF documents to find information about:

- course requirements;
- assessment structure;
- assignment deadlines;
- academic policies;
- programme regulations;
- academic integrity requirements.

Traditional keyword search such as Ctrl-F can be inefficient when the wording of the user's question differs from the wording in the source document, when information is distributed across multiple documents, or when the user does not know which document contains the answer.

This project investigates whether a RAG-based academic assistant can provide accurate, evidence-grounded answers while reducing the time required to search academic documents.

---

## 2. System Architecture

The system follows the following pipeline:

```text
PDF Documents
    ↓
Text Extraction
    ↓
Token-based Chunking
    ↓
Embedding Generation
    ↓
FAISS Vector Index
    ↓
Top-5 Retrieval
    ↓
Evidence-grounded LLM Generation
    ↓
Citation / Abstention
    ↓
Streamlit User Interface
```

The main components are:

- PDF extraction: `pypdf`
- Tokenisation: `tiktoken`
- Chunk size: 500 tokens
- Chunk overlap: 100 tokens
- Embedding model: `openai/text-embedding-3-small`
- Vector store: FAISS
- Retrieval method: normalized vector similarity search
- Retrieval depth: Top-5
- Generation model: `openai/gpt-4.1-mini`
- API provider: OpenRouter
- User interface: Streamlit
- Citation metadata: document ID, filename, and PDF page
- Abstention mechanism: explicit refusal when evidence is insufficient

---

## 3. Document Corpus

The prototype uses 10 NTU-related PDF documents.

The corpus includes:

1. PE6201 Course Outline
2. PE6201 Assessment Timeline
3. PE6202 Course Outline
4. PE6203 Course Outline
5. MSc Enterprise AI Trimester 1 Course Synopsis
6. NTU Student Discipline
7. NTU Open Access Policy
8. NTU Anti-Plagiarism Guidance
9. NTU Academic Integrity
10. NTU Graduate Academic Handbook

The documents were processed into a total of:

**72 text chunks**

Each chunk preserves metadata including:

- chunk ID;
- document ID;
- filename;
- PDF page number;
- token count;
- extracted text.

The prototype excludes personal and confidential information from the document corpus.

---

## 4. Document Processing

PDF text is extracted using `pypdf`.

Documents are processed page by page so that page-level citation information can be preserved.

The extracted text is divided into overlapping chunks using:

```text
Chunk size: 500 tokens
Chunk overlap: 100 tokens
```

Chunks do not span across PDF pages.

The final processed corpus contains:

```text
72 chunks
```

---

## 5. Embeddings and Vector Search

Each text chunk is embedded using:

```text
openai/text-embedding-3-small
```

The embedding vectors are converted to `float32` and L2-normalized.

The normalized embeddings are stored in a FAISS:

```text
IndexFlatIP
```

This allows inner-product search over normalized vectors to approximate cosine similarity.

For each user query, the system:

1. generates a query embedding;
2. normalizes the query vector;
3. searches the FAISS index;
4. retrieves the Top-5 most relevant chunks;
5. passes the retrieved evidence to the generation model.

---

## 6. Answer Generation

The system uses:

```text
openai/gpt-4.1-mini
```

through the OpenRouter API.

The generation prompt instructs the model to:

- answer only from the retrieved NTU evidence;
- avoid unsupported claims;
- directly answer the user's question;
- cite supporting sources;
- use the same language as the user where appropriate;
- abstain when the provided evidence is insufficient.

The application therefore combines retrieval with constrained generation rather than relying on the language model's general knowledge.

---

## 7. Abstention Mechanism

The system includes an explicit abstention mechanism.

When the available NTU documents do not provide sufficient evidence, the system returns:

```text
I cannot answer this question based on the provided NTU documents.
```

Examples of questions that should be rejected include questions about:

- current exchange rates;
- current stock prices;
- personal GPA;
- personal scholarship status;
- real-time campus shuttle information;
- other information that is not present in the document corpus.

The purpose of this mechanism is to reduce hallucination and prevent the system from presenting unsupported information as fact.

---

## 8. Streamlit Application

A Streamlit interface is provided for interactive use.

The interface allows users to:

- enter a natural-language question;
- receive an evidence-grounded answer;
- view cited sources;
- inspect retrieval results;
- view response time;
- inspect token usage.

The application can be started using:

```bash
streamlit run app.py
```

---

## 9. Development and System Freezing

A separate development set was created before formal evaluation.

The development set was used to inspect:

- answerable questions;
- unanswerable questions;
- retrieval scores;
- abstention behaviour;
- answer quality;
- citation behaviour.

After development and tuning were completed, the system configuration was frozen.

The formal evaluation was then performed without changing the main retrieval and generation configuration.

This separation reduces the risk of tuning the system directly on the final evaluation set.

---

## 10. Formal Evaluation Design

The formal evaluation set contained:

```text
50 answerable questions
10 unanswerable questions
60 questions in total
```

The 50 answerable questions included:

- reference answers;
- expected supporting documents;
- expected PDF pages.

The 10 unanswerable questions were designed to test whether the system correctly abstains when the answer is not available in the corpus.

The evaluation examined:

- answer correctness;
- citation validity;
- abstention accuracy;
- Document Recall@5;
- Exact Evidence Recall@5;
- RAG response time;
- manual Ctrl-F search time;
- paired time efficiency;
- API cost per question.

---

## 11. Answer Quality Results

The final answer-quality results were:

| Metric | Result |
|---|---:|
| Correct answers | 43 / 50 |
| Answer accuracy | 86.0% |
| Predefined target | At least 85% |
| Target achieved | Yes |
| Valid citation cases | 39 / 50 |
| Citation validity | 78.0% |
| Correct abstentions | 10 / 10 |
| Abstention accuracy | 100.0% |

The system achieved the predefined answer-accuracy target.

However, citation validity was lower than answer correctness. This indicates that the system can sometimes produce a correct answer while selecting unnecessary or less directly supporting citations.

---

## 12. Retrieval Results

Retrieval quality was evaluated using two metrics.

### Document Recall@5

Document Recall@5 measures whether at least one of the Top-5 retrieved chunks comes from the expected supporting document.

Result:

```text
47 / 50 = 94.0%
```

### Exact Evidence Recall@5

Exact Evidence Recall@5 measures whether the Top-5 retrieval results contain the expected supporting evidence at the required document/page level.

Result:

```text
42 / 50 = 84.0%
```

The difference between the two metrics suggests that the system often retrieves the correct document but does not always retrieve the exact supporting passage required for the question.

---

## 13. RAG vs Ctrl-F Baseline

The same 50 answerable questions were used for a manual Ctrl-F search comparison.

A 180-second timeout limit was used for the manual baseline.

### Overall Time Comparison

| Metric | RAG | Ctrl-F |
|---|---:|---:|
| Mean time | 2.44 s | 60.97 s |
| Median time | 2.24 s | 57.28 s |

Additional results:

```text
Mean time saved: 58.53 seconds per question
Overall mean-time reduction: 96.00%
Overall speed-up: 25.01×
Ctrl-F timeouts: 1 / 50
```

---

## 14. Both-Correct Paired Comparison

A paired comparison was also performed for questions where both approaches were treated as successful.

The results were:

```text
Both-correct questions: 43 / 50

RAG mean time: 2.37 seconds
Ctrl-F mean time: 55.95 seconds

Mean time saved: 53.58 seconds
Paired time reduction: 95.76%
Paired speed-up: 23.61×
```

The paired analysis provides a more conservative comparison because it focuses on cases where both methods successfully completed the task.

---

## 15. API Cost Analysis

API cost was estimated from measured token usage and listed provider rates dated 1 October 2026.

The formal evaluation used:

```text
Query embedding tokens: 883
LLM input tokens: 133,019
LLM output tokens: 3,509
```

The resulting estimated cost was:

| Metric | Result |
|---|---:|
| Total online API cost for 60 questions | $0.058840 |
| Average online API cost per question | $0.000981 |
| One-off corpus embedding cost | $0.000525 |

The corpus embedding stage processed:

```text
72 chunks
26,269 embedding tokens
```

For reporting purposes, the one-off corpus embedding cost was also amortised over 1,000 queries.

The resulting average cost including amortised corpus embedding remained approximately:

```text
$0.000981 per query
```

The cost estimate includes:

- query embedding;
- LLM input tokens;
- LLM output tokens.

The estimate excludes:

- human labour;
- local computing costs;
- local FAISS operations;
- electricity;
- hosting costs.

The cost figures should therefore be interpreted as estimated API costs based on measured token usage and listed provider rates, rather than as a complete operational cost model.

---

## 16. Key Findings

The final evaluation produced four main findings.

First, the system achieved an answer accuracy of:

```text
86.0%
```

which met the predefined target of at least 85%.

Second, the system achieved:

```text
100.0% abstention accuracy
```

on the 10 formally defined unanswerable questions.

Third, the RAG system substantially reduced document-search time compared with the Ctrl-F baseline.

The overall comparison showed:

```text
RAG mean response time: 2.44 seconds
Ctrl-F mean search time: 60.97 seconds
Overall speed-up: 25.01×
```

Fourth, the system was inexpensive to operate at the prototype scale:

```text
Average online API cost: approximately $0.000981 per question
```

---

## 17. Main Limitations

The prototype has several limitations.

### Citation selection

Citation validity was:

```text
39 / 50 = 78.0%
```

which was lower than answer correctness.

Some answers were correct but included unnecessary or less directly supporting sources.

Improving citation selection is therefore one of the clearest opportunities for future work.

### Exact evidence retrieval

Document Recall@5 reached 94%, while Exact Evidence Recall@5 was 84%.

This suggests that retrieving the correct document is easier than retrieving the exact supporting passage.

### Small corpus

The prototype uses only 10 selected NTU documents.

The evaluation therefore does not establish reliability across all NTU academic information.

### Document freshness

University policies, deadlines, courses, and academic requirements may change.

The system does not automatically detect or update outdated documents.

### Human baseline

The Ctrl-F baseline represents a limited manual-search experiment and should not be interpreted as a universal estimate of human search performance.

### External API dependency

The generation and embedding models are accessed through an external API.

The system therefore depends on:

- API availability;
- API pricing;
- model behaviour;
- external data-processing considerations.

---

## 18. Responsible Use

The application is a prototype academic-information assistant.

It is not intended to replace official NTU systems or authoritative university information.

Important information such as:

- academic deadlines;
- graduation requirements;
- disciplinary matters;
- assessment requirements;
- programme rules;

should still be verified against official NTU sources.

Retrieved documents are treated as evidence rather than instructions.

When evidence is insufficient, the system is designed to abstain instead of generating unsupported information.

---

## 19. Possible Future Improvements

Possible future extensions include:

- hybrid keyword and vector retrieval;
- retrieval reranking;
- query rewriting;
- better chunk deduplication;
- improved citation-selection logic;
- citation verification after generation;
- automatic document-version tracking;
- larger document corpora;
- evaluation with more users;
- multiple-human-searcher Ctrl-F baselines;
- automated monitoring of updated NTU documents.

These extensions were not added to the formally evaluated version because the project focused first on evaluating a frozen basic RAG pipeline.

---

## 20. Project Structure

```text
6201结课作业/
│
├── 所需文件/
│   ├── D1_PE6201_Course_Outline.pdf
│   ├── D2_PE6201_Assessment_Timeline.pdf
│   ├── D3_PE6202_Course_Outline.pdf
│   ├── D4_PE6203_Course_Outline.pdf
│   ├── D5_MSEAI_Trimester1_Course_Synopsis.pdf
│   ├── D6_NTU_Statute6_Student_Discipline_2024.pdf
│   ├── D7_NTU_Open_Access_Policy_2024.pdf
│   ├── D8_NTU_Anti_Plagiarism_Guidance.pdf
│   ├── D9_NTU_Academic_Integrity.pdf
│   └── D10_Graduate_Academic_Handbook.pdf
│
├── 向量库/
│   ├── index.faiss
│   ├── metadata.json
│   └── 构建统计.json
│
├── 读取文档.py
├── 分块文档.py
├── 检查分块质量.py
├── 建立向量库.py
├── 检索测试.py
├── RAG问答.py
├── app.py
│
├── 开发集.csv
├── 正式测试集.csv
├── 正式测试结果.csv
├── 正式评估人工复核.csv
├── Recall@5结果.csv
├── CtrlF基线结果.csv
├── RAG_vs_CtrlF比较结果.csv
├── RAG_vs_CtrlF汇总.csv
├── 正式测试成本明细.csv
├── 成本汇总.csv
├── 最终实验结果汇总.csv
│
├── 分块结果.jsonl
├── README.md
├── requirements.txt
├── .gitignore
└── .env
```

The `.env` file is local only and must not be committed to GitHub.

---

## 21. Installation

Create a Python environment and install the required packages.

```bash
pip install -r requirements.txt
```

The project requires an OpenRouter API key.

Create a local `.env` file in the project root:

```text
OPENROUTER_API_KEY=your_api_key_here
```

Do not commit the `.env` file or API keys to a public repository.

---

## 22. Running the Application

If the FAISS index has already been created, start the Streamlit application using:

```bash
streamlit run app.py
```

The browser interface will then allow the user to submit questions to the NTU Academic Assistant.

---

## 23. Rebuilding the RAG Pipeline

To rebuild the system from the source documents, run the preprocessing and indexing scripts in sequence.

The overall workflow is:

```text
1. Read the PDF documents
2. Extract the text
3. Create overlapping chunks
4. Inspect chunk quality
5. Generate embeddings
6. Build the FAISS index
7. Test retrieval
8. Run the RAG application
```

The relevant Python scripts are included in the project repository.

---

## 24. Security

API credentials must never be stored directly in source code.

The OpenRouter API key is loaded from:

```text
.env
```

The following files and folders should be excluded from GitHub through `.gitignore`:

```text
.env
.venv/
__pycache__/
*.pyc
.DS_Store
.idea/
```

If an API key is accidentally exposed publicly, it should be revoked immediately and replaced with a new key.

---

## 25. Final Result Summary

The formally evaluated prototype achieved:

```text
Answer accuracy:
43/50 = 86.0%

Valid citation cases:
39/50 = 78.0%

Correct abstentions:
10/10 = 100.0%

Document Recall@5:
47/50 = 94.0%

Exact Evidence Recall@5:
42/50 = 84.0%

RAG mean response time:
2.44 seconds

Ctrl-F mean search time:
60.97 seconds

Overall time reduction:
96.00%

Overall speed-up:
25.01×

Both-correct paired speed-up:
23.61×

Total API cost for 60 formal questions:
$0.058840

Average API cost per question:
$0.000981

One-off corpus embedding cost:
$0.000525
```

The results show that the prototype met its predefined answer-accuracy target and substantially reduced information-search time while maintaining a very low API cost.

The main remaining weaknesses are citation precision and exact evidence retrieval, which provide clear directions for future improvement.