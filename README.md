### Capstone Project ### 
# Zepto Analytics & Support Assistant

An end-to-end Machine Learning pipeline and GenAI RAG Support Assistant built for Zepto operations.

## Repository Structure

Masai-Capstone-project/
├── analytics/
│   ├── 01_eda.py
│   ├── 02_modeling.py
│   ├── titanic.csv
│   ├── chart_decision_tree.png
│   ├── chart_residuals.png
│   └── best_titanic_pipeline.joblib
├── support_assistant/
│   ├── docs/
│   │   ├── doc_01.txt
│   │   ├── doc_02.txt
│   │   ├── doc_03.txt
│   │   ├── doc_04.txt
│   │   ├── doc_05.txt
│   │   ├── doc_06.txt
│   │   ├── doc_07.txt
│   │   └── doc_08.txt
│   ├── Dockerfile
│   ├── graph.py
│   ├── main.py
│   ├── prompts.py
│   ├── requirements.txt
│   ├── vectorstore.py
│   └── README.md
├── .gitignore
└── README.md

# Project Overview # 
Analytics & Predictive Modeling (/analytics)
EDA (01_eda.py): Automated data cleaning, categorical encoding, and feature correlation analysis.
Modeling (02_modeling.py): Classification pipeline (Logistic Regression, Decision Trees, Random Forests) with SMOTE balancing, hyperparameter tuning (GridSearchCV), and regression task with residual evaluation.
Artifacts: Exported inference pipeline (best_titanic_pipeline.joblib) and diagnostic plots.

# Support Assistant RAG Pipeline (/support assistant) # 
Vector Store: Local persistent ChromaDB collection using sentence-transformers/all-MiniLM-L6-v2 embeddings over 8 Zepto policy documents (doc_01–doc_08).
Lang Graph Flow: 3-node graph (classify intent → retrieve_and_answer / direct answer) with conditional routing.
Offline Mock Mode (MOCK_LLM=1): Deterministic keyword classification and context-grounded canned answers running fully offline.
Fast API Service: Exposes POST /ask endpoint returning structured Pydantic JSON (answer, sources, confidence).
Containerization: Docker file configured to run Fast API on port 7860.

# Quick Start # 
1. Run Analytics Pipeline
python analytics/01_eda.py
python analytics/02_modeling.py

2. Run Support Assistant API
pip install -r support_assistant/requirements.txt
python -m uvicorn support_assistant.main:app --host 0.0.0.0 --port 7860

3. Run with Docker
cd support assistant
docker build -t zepto-support-assistant .
docker run -p 7860:7860 zepto-support-assistant

# Tech Stack #
ML & Analytics: Python, Pandas, Scikit-learn, Joblib, Matplotlib

RAG & Orchestration: LangGraph, ChromaDB, Sentence-Transformers, Pydantic

API & Deployment: FastAPI, Uvicorn, Docker
