# Product Documentation

## 1. Product Overview

**NTU Academic Assistant** is a Retrieval-Augmented Generation (RAG) prototype designed to help NTU postgraduate students retrieve course and academic information from a defined document corpus.

The system accepts a natural-language question, retrieves relevant evidence from NTU-related PDF documents, generates a concise answer grounded in that evidence, and provides source citations.

If the available documents do not provide sufficient support, the system abstains instead of generating an unsupported answer.

---

## 2. Persona

### Primary Persona

**NTU postgraduate student**

Typical needs include:

- checking course requirements;
- finding assessment deadlines;
- understanding assessment structures;
- locating academic-policy information;
- checking graduate academic regulations;
- finding information across multiple PDF documents.

### User Problem

Academic information may be distributed across multiple course outlines, assessment timelines, handbooks and policy documents.

Traditional Ctrl-F search requires users to:

- identify the likely source document;
- choose appropriate keywords;
- inspect multiple search results;
- manually combine information when the answer is distributed across documents.

The NTU Academic Assistant provides a natural-language interface over this document corpus.

---

## 3. Input

The primary system input is:

```text
A natural-language question from the user
```

Example:

```text
What is the deadline for appealing against academic termination and how many appeals are allowed per candidature?
```

The system is designed for questions that can be answered from the defined NTU document corpus.

The input can include:

- course-information questions;
- assessment questions;
- academic-policy questions;
- programme-information questions;
- graduate academic-regulation questions.

---

## 4. Output

The system produces one of two main outputs.

### Supported Question

For a question supported by the corpus, the system returns:

```text
Evidence-grounded answer
+
Source citation(s)
```

The citation information is derived from retained document and PDF-page metadata.

### Unsupported Question

If the corpus does not contain sufficient evidence, the system returns an abstention response:

```text
I cannot answer this question based on the provided NTU documents.
```

This prevents the application from intentionally relying on unrestricted model knowledge for unsupported questions.

---

## 5. High-Level Product Architecture

The high-level product flow is:

```text
User Question
      ↓
Query Embedding
      ↓
FAISS Vector Search
      ↓
Top-5 Relevant Chunks
      ↓
Retrieved Evidence
      ↓
GPT-4.1-mini
      ↓
Answer + Citation
      OR
Abstention
      ↓
Streamlit Interface
```

The knowledge-base construction process is:

```text
10 NTU-related PDF Documents
      ↓
pypdf Text Extraction
      ↓
Page-level Processing
      ↓
500-token Chunks
100-token Overlap
      ↓
text-embedding-3-small
      ↓
L2 Normalization
      ↓
FAISS IndexFlatIP
```

---

## 6. Main Technical Components

| Component | Implementation |
|---|---|
| PDF extraction | `pypdf` |
| Tokenisation | `tiktoken` |
| Chunk size | 500 tokens |
| Chunk overlap | 100 tokens |
| Number of source PDFs | 10 |
| Number of chunks | 72 |
| Embedding model | `openai/text-embedding-3-small` |
| Vector search | FAISS `IndexFlatIP` |
| Retrieval depth | Top-5 |
| Generation model | `openai/gpt-4.1-mini` |
| API provider | OpenRouter |
| User interface | Streamlit |
| Evidence metadata | Document ID, filename and PDF page |
| Unsupported-query handling | Explicit abstention |

---

## 7. Product Logic

For each user question, the system performs the following steps:

```text
1. Receive the user question

2. Generate a query embedding

3. Normalize the embedding

4. Search the FAISS vector index

5. Retrieve the Top-5 most similar chunks

6. Assess whether the retrieved evidence is sufficient

7. If evidence is insufficient:
   Return an abstention

8. If evidence is sufficient:
   Send the question and retrieved evidence to GPT-4.1-mini

9. Generate an evidence-grounded answer

10. Display the answer and supporting source information
```

---

## 8. Why RAG

The project uses RAG rather than a standalone language model because the task depends on institution-specific information.

A standalone language model may not contain:

- the latest course structure;
- project-specific assessment details;
- the exact contents of the selected academic-policy documents;
- the specific document versions used in the prototype.

RAG allows the generation model to answer using retrieved project-specific evidence.

This also makes the answer more traceable because the application can show the source documents used in generation.

---

## 9. Why Local FAISS

A local FAISS vector index was selected because the prototype uses a compact corpus of only 72 chunks.

Advantages for this prototype include:

- simple implementation;
- low operating cost;
- fast local vector search;
- no additional managed-vector-database service;
- easy reproducibility.

A managed vector database may become more appropriate if the system is expanded to:

- much larger document collections;
- multiple simultaneous users;
- shared enterprise access;
- centralized administration;
- production-scale monitoring.

---

## 10. Metrics Targeted

The project defined answer quality as a primary success measure.

The predefined answer-accuracy target was:

```text
At least 85%
```

The evaluation also measured several supporting dimensions:

```text
Answer Accuracy
Citation Validity
Correct Abstention
Document Recall@5
Exact Evidence Recall@5
Response Time
Ctrl-F Baseline Time
Overall Speed-up
Both-Correct Paired Speed-up
API Cost per Question
```

---

## 11. Metrics Reached

The formal evaluation produced the following results:

| Metric | Result |
|---|---:|
| Answer Accuracy | **43/50 = 86.0%** |
| Predefined Answer-Accuracy Target | **≥85%** |
| Target Achieved | **Yes** |
| Valid Citation Cases | **39/50 = 78.0%** |
| Correct Abstentions | **10/10 = 100.0%** |
| Document Recall@5 | **47/50 = 94.0%** |
| Exact Evidence Recall@5 | **42/50 = 84.0%** |
| RAG Mean Response Time | **2.44 seconds** |
| Ctrl-F Mean Search Time | **60.97 seconds** |
| Overall Speed-up | **25.01×** |
| Both-Correct Paired Speed-up | **23.61×** |
| Total Online API Cost / 60 Questions | **$0.058840** |
| Average Online API Cost | **$0.000981/question** |
| One-off Corpus Embedding Cost | **$0.000525** |

---

## 12. Business Value

The prototype reduces the effort required to locate information across multiple academic documents.

Under the formal evaluation procedure:

```text
RAG mean response time:
2.44 seconds

Manual Ctrl-F mean search time:
60.97 seconds
```

This resulted in a measured:

```text
25.01× overall speed-up
```

The system therefore provides a faster route from a natural-language question to an evidence-backed answer within the tested document corpus.

The low measured API cost also supports the feasibility of a lightweight academic-information assistant.

---

## 13. Responsible Product Behaviour

The product includes several controls intended to improve responsible use:

- answers are grounded in retrieved evidence;
- source information is displayed;
- unsupported questions trigger abstention;
- personal and confidential student records are excluded from the corpus;
- API credentials are stored outside the source code;
- official NTU documents remain authoritative for consequential academic decisions.

The system is designed as an academic-information assistant, not as a replacement for official NTU systems or official university advice.

---

## 14. Current Product Scope

The current prototype is intentionally bounded.

It is designed around:

```text
10 NTU-related PDF documents
72 chunks
50 answerable formal questions
10 unanswerable formal questions
```

This controlled scope allows retrieval, citation and abstention behaviour to be evaluated against known evidence.

The current system demonstrates the complete end-to-end product flow from natural-language input to evidence-grounded output.

---

## 15. Future Product Direction

The current results provide a baseline for future improvements.

Potential extensions include:

- retrieval reranking;
- hybrid keyword and vector retrieval;
- query rewriting;
- citation verification;
- improved evidence selection;
- automatic document-version tracking;
- controlled index refresh;
- larger document collections;
- multi-user evaluation.

Future changes should be evaluated against the frozen baseline and should demonstrate measurable improvements in evidence quality, answer quality, latency or cost before additional architectural complexity is introduced.