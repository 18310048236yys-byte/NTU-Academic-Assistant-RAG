# Evaluation Documentation

## 1. Overview

The NTU Academic Assistant was evaluated using a structured development and formal testing process.

The evaluation was designed to measure:

- answer correctness;
- citation validity;
- abstention performance;
- retrieval quality;
- response efficiency;
- comparison with manual Ctrl-F search;
- API cost per question.

A separate development set was used before formal evaluation. After development and tuning, the system configuration was frozen before the final test set was run.

---

## 2. Evaluation Structure

The evaluation process was divided into two stages:

```text
Development Set
    ↓
System Tuning
    ↓
Configuration Frozen
    ↓
Formal Evaluation
```

This separation was used to reduce direct tuning on the final evaluation questions.

---

## 3. Development Set

The development set contained:

```text
10 answerable questions
5 unanswerable questions
```

The development set was used to inspect:

- retrieval behaviour;
- answer quality;
- abstention behaviour;
- retrieval-score patterns;
- citation behaviour.

The development questions were kept separate from the final formal evaluation set.

The system configuration was frozen before formal testing.

---

## 4. Formal Evaluation Set

The formal evaluation contained:

```text
50 answerable questions
10 unanswerable questions
60 questions in total
```

The answerable questions were created with:

- a reference answer;
- an expected supporting document;
- an expected supporting PDF page.

The unanswerable questions were designed to test whether the system would abstain when the required information was not present in the defined document corpus.

Examples of unsupported question categories included:

- personal student information;
- real-time campus information;
- current financial-market information;
- other information outside the provided NTU documents.

---

## 5. Formal Evaluation Files

The main evaluation files are:

| File | Purpose |
|---|---|
| `开发集.csv` | Development questions used before the system was frozen |
| `开发集结果.csv` | Outputs from the development evaluation |
| `正式测试集.csv` | Final 50 answerable + 10 unanswerable questions |
| `正式测试结果.csv` | Model outputs, retrieval information, latency and token usage |
| `正式评估人工复核.csv` | Human review of answer correctness and citation validity |
| `Recall@5结果.csv` | Retrieval evaluation results |
| `CtrlF基线结果.csv` | Manual Ctrl-F baseline results |
| `RAG_vs_CtrlF比较结果.csv` | Question-level RAG vs Ctrl-F comparison |
| `RAG_vs_CtrlF汇总.csv` | Aggregate efficiency comparison |
| `正式测试成本明细.csv` | Per-question API cost estimates |
| `成本汇总.csv` | Aggregate cost results |
| `最终实验结果汇总.csv` | Consolidated final experiment metrics |

These files are retained so that aggregate metrics can be traced back to question-level results.

---

## 6. Answer Correctness Evaluation

Answer correctness was assessed manually against the reference answers and the supporting NTU documents.

A response was counted as correct when it:

- addressed the question;
- matched the required factual content;
- did not introduce unsupported claims that changed the meaning of the answer.

The final result was:

```text
43 / 50 correct
= 86.0% answer accuracy
```

The predefined success target was:

```text
at least 85% answer accuracy
```

The final system therefore met the predefined answer-accuracy target.

---

## 7. Citation Validity

Citation validity was evaluated separately from answer correctness.

This separation is important because a response can contain the correct factual answer while citing a source that does not directly support the answer.

The final result was:

```text
39 / 50 valid citation cases
= 78.0%
```

This metric is reported separately from answer accuracy so that answer generation and evidence attribution can be evaluated independently.

---

## 8. Abstention Evaluation

The formal test set contained 10 questions whose answers were not available in the defined corpus.

A correct result required the system to abstain rather than generate an unsupported answer.

The system achieved:

```text
10 / 10 correct abstentions
= 100.0%
```

The standard abstention response was:

```text
I cannot answer this question based on the provided NTU documents.
```

The abstention test therefore checks whether the system can distinguish between supported academic-information questions and questions outside the available evidence.

---

## 9. Retrieval Evaluation

Retrieval was evaluated using two Recall@5 measures.

### Document Recall@5

Document Recall@5 checks whether at least one of the Top-5 retrieved chunks comes from the expected supporting document.

Result:

```text
47 / 50
= 94.0%
```

### Exact Evidence Recall@5

Exact Evidence Recall@5 checks whether the Top-5 retrieval results include the expected supporting evidence at the required document/page level.

Result:

```text
42 / 50
= 84.0%
```

The two measures are reported separately because retrieving the correct document does not always mean retrieving the exact supporting passage required for the answer.

---

## 10. Manual Ctrl-F Baseline

The same 50 answerable questions were used for a manual Ctrl-F comparison.

The purpose of the baseline was to compare the RAG system against a simple document-search workflow using the same source material.

The timing procedure used:

```text
Question displayed
    ↓
Manual document search using Ctrl-F
    ↓
Answer and source identified
    ↓
Timer stopped
```

A maximum time limit of:

```text
180 seconds per question
```

was used.

The baseline produced:

```text
Ctrl-F mean search time: 60.97 seconds
Ctrl-F median search time: 57.28 seconds
Timeouts: 1 / 50
```

The experiment used one manual searcher.

---

## 11. RAG Response-Time Evaluation

RAG latency was measured using the same 50 answerable questions.

The results were:

```text
RAG mean response time: 2.44 seconds
RAG median response time: 2.24 seconds
```

Compared with the manual Ctrl-F baseline:

```text
Mean time saved: 58.53 seconds per question
Overall mean-time reduction: 96.00%
Overall speed-up: 25.01×
```

---

## 12. Both-Correct Paired Comparison

A paired comparison was also performed for questions where both approaches were treated as successful.

The paired set contained:

```text
43 questions
```

Results:

```text
RAG mean time: 2.37 seconds
Ctrl-F mean time: 55.95 seconds

Mean time saved: 53.58 seconds
Paired time reduction: 95.76%
Paired speed-up: 23.61×
```

This paired comparison helps separate efficiency from answer-success differences by comparing questions successfully completed by both approaches.

---

## 13. API Cost Evaluation

Cost was estimated from measured token usage in the formal evaluation.

The calculation included:

- query-embedding tokens;
- LLM input tokens;
- LLM output tokens.

The formal test used:

```text
Query embedding tokens: 883
LLM input tokens: 133,019
LLM output tokens: 3,509
```

Using the recorded provider rates dated 1 October 2026, the estimated online API cost was:

```text
Total online API cost for 60 questions:
$0.058840

Average online API cost per question:
$0.000981
```

The one-off corpus embedding cost was:

```text
$0.000525
```

The corpus embedding stage used:

```text
26,269 tokens
72 chunks
```

The cost estimate does not include:

- human labour;
- local computing;
- electricity;
- hosting;
- other operational overhead.

---

## 14. Metrics Targeted and Metrics Reached

| Metric | Target / Purpose | Final Result |
|---|---|---:|
| Answer Accuracy | At least 85% | **86.0%** |
| Citation Validity | Measure evidence attribution | **78.0%** |
| Correct Abstention | Test unsupported-question handling | **100.0%** |
| Document Recall@5 | Measure document-level retrieval | **94.0%** |
| Exact Evidence Recall@5 | Measure document/page evidence retrieval | **84.0%** |
| Search Efficiency | Faster than manual Ctrl-F | **25.01× overall speed-up** |
| Both-Correct Efficiency | Compare successful cases | **23.61× speed-up** |
| API Cost | Measure prototype cost-to-serve | **$0.000981/question** |

---

## 15. Evaluation Interpretation

The evaluation indicates that the system met its predefined answer-accuracy target while providing strong document-level retrieval and reliable abstention on the formal unsupported-question set.

The difference between:

```text
Document Recall@5: 94.0%
Exact Evidence Recall@5: 84.0%
Citation Validity: 78.0%
```

shows why retrieval and citation should be measured separately from answer correctness.

The correct document is often retrieved, but evidence selection and citation attribution remain distinct stages of the pipeline.

The efficiency experiment also shows a substantial measured reduction in search time under the tested procedure.

Because the Ctrl-F baseline used a single searcher, the measured speed-up is interpreted as a prototype-level comparison rather than a universal estimate of human search performance.

---

## 16. Evaluation Scope

The evaluation was designed for the defined prototype corpus.

The results therefore describe performance on:

- the 10-document NTU corpus;
- the frozen formal test set;
- the implemented retrieval and generation configuration.

The evaluation does not claim university-wide coverage.

Future evaluation could extend the current design through:

- additional paraphrased questions;
- more near-scope unanswerable questions;
- additional human reviewers;
- multiple manual-search participants;
- larger document collections;
- comparative retrieval experiments.

---

## 17. Reproducibility

The evaluation scripts and result files are retained in the repository so that the evaluation workflow can be inspected.

The overall process is:

```text
Development Set
    ↓
Tune Prototype
    ↓
Freeze Configuration
    ↓
Run 60-question Formal Test
    ↓
Human Review
    ↓
Recall@5 Evaluation
    ↓
Ctrl-F Baseline
    ↓
Efficiency Comparison
    ↓
API Cost Analysis
    ↓
Final Metric Summary
```

The reported metrics are based on the saved formal evaluation outputs rather than on manually reconstructed summary values.