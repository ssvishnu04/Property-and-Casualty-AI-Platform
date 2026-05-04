# Property & Casualty Insurance & Reinsurance AI Risk Platform

A **production-style AI platform** for real-time claim risk scoring, reinsurance analytics, and GenAI-powered explanations using **Machine Learning + RAG architecture**.

---
## Live Demo

**Streamlit App (Frontend UI)**  
https://propertycasualty-reinsurance-risk-platform.streamlit.app  
---

## Application Screenshots

### Real-Time Claim Scoring
![Real Time](data/images/ApplicationScreen.jpg)

### Prediction Audit Dashboard
![Audit Logs](data/images/PredictionAudit.jpg)

### GenAI Explanation (RAG)
![RAG](data/images/RagExplanation.jpg)

### Model Monitoring
![Monitoring](data/images/Monitoring_Drift.jpg)

---
## Overview

This project simulates a **real-world AI platform used in Property & Casualty insurance** by:

- Claims adjusters  
- Fraud analysts  
- Risk & reinsurance teams 

It combines:

* Machine Learning (risk & fraud prediction)
* Reinsurance analytics (ceded loss, retention)
* Generative AI (RAG-based explanations)
* MLOps (audit logs, drift monitoring, CI/CD)

---

## Key Features

### Real-Time Claim Scoring

* Severity prediction (Low / Medium / High)
* Fraud probability scoring
* Recommended reserve
* Triage priority (Urgent / Standard)

### Reinsurance Analytics

* Retention breach detection
* Ceded loss calculation
* Recovery ratio estimation

### GenAI (RAG Explanation)

* Context-aware explanations using LLM
* Combines structured claim data + unstructured documents (notes , emails , Accord XML)
* Improves trust and explainability

### Monitoring & Audit (MLOps)

* Prediction audit logs
* Feature & prediction drift detection
* Model Performance tracking

---
## Example Use Case
A claims adjuster submits a new claim:

- Loss Type: **Hail**
- State: **TX**
- Claim Amount: **$425,000**

### System Output:
- Severity → **High**
- Fraud Risk → **Low**
- Recommended Reserve → **$434K**
- Triage Priority → **Urgent**
- Reinsurance → **Retention breached → Ceded loss triggered**
- GenAI → **Explains risk drivers + recommended actions**

**Outcome:** Faster triage, reduced manual review, better financial decisions

---

## Architecture

### High-Level Flow

```
Streamlit UI (User)
→ FastAPI (Backend)
→ ML Models (Risk + Fraud + Reserve)
→ RAG Pipeline (LangChain + FAISS)
→ Response (Prediction + Explanation)
```

### Data Pipeline (Lakehouse Style)

```
Bronze Layer (Ingestion)
→ Silver Layer (Cleaning + Standardization)
→ Gold Layer (Feature Engineering)
→ ML + GenAI
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
| Data Pipeline| Pandas                    |
| Deployment | Docker, Hugging Face Spaces |
| CI/CD      | GitHub Actions              |

---


## Project Structure

```
api/            → FastAPI backend (inference + APIs)
app/            → Streamlit frontend (UI dashboards)
src/            → Data pipelines, ML models, RAG logic
data/           → Bronze / Silver / Gold layers
tests/          → Unit tests
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
