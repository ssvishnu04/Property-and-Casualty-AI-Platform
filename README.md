# Property & Casualty Insurance & Reinsurance AI Risk Platform

A **production-style AI system** for real-time claim risk scoring, reinsurance analytics, and GenAI-powered explanations using ML + RAG architecture.

---

## Overview

This project simulates a real-world **P&C insurance AI platform** used by adjusters, analysts, and risk teams.

It combines:

* Machine Learning (risk & fraud prediction)
* Reinsurance analytics (ceded loss, retention)
* Generative AI (RAG-based explanations)
* MLOps (audit logs, drift monitoring, CI/CD)

---

## Key Features

### Real-Time Claim Scoring

* Severity prediction (Low / Medium / High)
* Fraud risk probability
* Recommended reserve
* Triage priority

### Reinsurance Analytics

* Retention breach detection
* Ceded loss calculation
* Recovery ratio estimation

### GenAI (RAG)

* Context-aware explanations using LLM
* Combines structured + unstructured data

### Monitoring & Audit

* Prediction audit logs
* Feature & prediction drift detection

---

## Architecture

### High-Level Flow

```
Streamlit UI → FastAPI → ML + RAG → Response
```

### Data Pipeline

```
Bronze → Silver → Gold → ML + GenAI
```

---

## Tech Stack

| Layer      | Tools                       |
| ---------- | --------------------------- |
| Frontend   | Streamlit                   |
| Backend    | FastAPI                     |
| ML         | Scikit-learn                |
| GenAI      | LangChain + Groq            |
| Vector DB  | FAISS                       |
| Embeddings | Hugging Face                |
| Deployment | Docker, Hugging Face Spaces |
| CI/CD      | GitHub Actions              |

---


## Project Structure

```
api/            → FastAPI backend
app/            → Streamlit frontend
src/            → Data + ML + RAG pipelines
data/           → Bronze / Silver / Gold layers
```

---

## Deployment

* FastAPI deployed on Hugging Face Spaces (Docker)
* Streamlit deployed on Streamlit Cloud
* CI/CD via GitHub Actions

---

## Business Impact

* Faster claim triage
* Improved fraud detection
* Transparent AI decisions
* Better reinsurance risk visibility

---

## 👨‍💻 Author

Vishnu Yadavalli
Senior ML / Data Engineering Leader

---
