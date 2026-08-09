Zepto Analytics & GenAI Customer Support Assistant
An end-to-end Machine Learning data pipeline and GenAI RAG Support Assistant built for Zepto operations. This repository contains data processing workflows, machine learning models for customer behavior analysis, and a production-ready RAG service for automated customer support.

📁 Repository Structure
Plaintext
Masai-Capstone-project/
├── analytics/                      # Module 1 & 2: Data Pipeline, EDA & ML Modeling
│   ├── 01_eda.py                   # Exploratory Data Analysis script
│   ├── 02_modeling.py              # ML classification & regression pipeline
│   ├── titanic.csv                 # Core dataset
│   ├── chart_decision_tree.png     # Decision Tree visualization artifact
│   ├── chart_residuals.png         # Regression residual plot artifact
│   └── best_titanic_pipeline.joblib# Serialized model pipeline artifact
│
├── support_assistant/              # Module 3: RAG GenAI Support Assistant
│   ├── docs/                       # Zepto policy document corpus (doc_01 - doc_08)
│   ├── Dockerfile                  # Production container definition (Port 7860)
│   ├── graph.py                    # LangGraph orchestration & intent router
│   ├── main.py                     # FastAPI server exposing POST /ask
│   ├── prompts.py                  # Structured RAG prompt skeleton
│   ├── requirements.txt            # Module 3 dependencies
│   ├── vectorstore.py              # SentenceTransformers & ChromaDB vectorstore
│   └── README.md                   # Detailed Module 3 documentation & API benchmarks
│
├── .gitignore                      # Git exclusion rules
└── README.md                       # Project root documentation
📊 Modules Overview
1. Module 1 & 2 — Data Pipeline, EDA & Predictive Modeling (/analytics)
This module covers the end-to-end data processing and predictive modeling pipeline.

Key Implementation Highlights:
Exploratory Data Analysis (01_eda.py): Automated data cleaning, missing value imputation, categorical encoding, and feature correlation analysis.

Classification Pipeline (02_modeling.py): Trains and compares Logistic Regression, Decision Trees, and Random Forests using scikit-learn Pipelines. Includes handling of class imbalance via SMOTE and class_weight='balanced'.

Hyperparameter Tuning: Automated grid search via GridSearchCV evaluating models with 5-fold cross-validation based on F1-score.

Regression Side-Task: Linear regression model predicting customer transaction metrics with residual analysis.

Artifact Export: Exports the fitted preprocessing and inference pipeline into best_titanic_pipeline.joblib alongside visual diagnostic charts.

2. Module 3 — Support Assistant (/support_assistant)
Module 3 is a GenAI-powered automated support assistant built with LangGraph, ChromaDB, Sentence-Transformers, and FastAPI.

Architecture Overview:
Plaintext
[User Query] ──> Local Vector Store (ChromaDB + all-MiniLM-L6-v2)
                      │
                      ▼
            1. Intent Classification (LangGraph Node)
                      │
        ┌─────────────┴─────────────┐
        │ policy_question           │ general_question
        ▼                           ▼
 2. Retrieve & Answer         3. Direct Answer
 (ChromaDB Cosine Search)      (Canned / Direct Response)
        │                           │
        └─────────────┬─────────────┘
                      ▼
           [Pydantic JSON Response]
Key Implementation Highlights:
Document Ingestion: Embeds 8 official Zepto policy documents (docs/doc_01.txt to doc_08.txt) locally using sentence-transformers/all-MiniLM-L6-v2 into a persistent ChromaDB vector store.

LangGraph Orchestration (graph.py): StateGraph workflow featuring 3 nodes (classify_intent, retrieve_and_answer, direct_answer) and a conditional routing edge.

Offline Mock Baseline (MOCK_LLM=1): Fully deterministic, offline mode running keyword-heuristic intent classification and canned context-grounded responses without external API calls or network requirements.

Structured Output Enforcement: Pydantic schema (AssistantResponse) guaranteeing structured outputs (answer, sources, confidence) with built-in retry-on-failure logic for optional LLM integrations (MOCK_LLM=0).

FastAPI Service (main.py): Web API exposing a POST /ask endpoint.

Containerization: Dockerfile configured to serve the FastAPI application via uvicorn on port 7860.

🚀 Getting Started
Prerequisites
Python 3.10+

Git

Docker Desktop (Optional, for containerized execution)

💻 Local Setup & Execution
1. Clone the Repository
PowerShell
git clone https://github.com/Pranav-git08/Masai-Capstone-project.git
cd Masai-Capstone-project
2. Run Analytics Pipeline (Modules 1 & 2)
PowerShell
# Run Exploratory Data Analysis
python analytics/01_eda.py

# Run Model Training & Export Artifacts
python analytics/02_modeling.py
3. Run Support Assistant API (Module 3)
PowerShell
# Install dependencies
pip install -r support_assistant/requirements.txt

# Set Python Path to root
$env:PYTHONPATH="."

# Launch FastAPI Server
python -m uvicorn support_assistant.main:app --host 0.0.0.0 --port 7860 --reload
Interactive API documentation will be available at: http://localhost:7860/docs

🧪 Sample API Usage (Module 3)
Policy Question (Triggers Retrieval)
PowerShell
Invoke-RestMethod -Uri "http://localhost:7860/ask" -Method Post -ContentType "application/json" -Body '{"query": "What is the delivery fee for orders below 149?"}' | ConvertTo-Json
Response:

JSON
{
  "answer": "Based on the retrieved context: Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes of order confirmation...",
  "sources": ["doc_01", "doc_05", "doc_03"],
  "confidence": 1.0
}
General Question (Direct Fallback)
PowerShell
Invoke-RestMethod -Uri "http://localhost:7860/ask" -Method Post -ContentType "application/json" -Body '{"query": "Tell me a joke"}' | ConvertTo-Json
Response:

JSON
{
  "answer": "I can only answer questions about Zepto policies right now.",
  "sources": [],
  "confidence": 1.0
}
🐳 Running with Docker (Module 3)
PowerShell
cd support_assistant
docker build -t zepto-support-assistant .
docker run -p 7860:7860 zepto-support-assistant
🛠️ Tech Stack
Languages & Libraries: Python, Pandas, NumPy, Scikit-learn, Joblib, Matplotlib

Orchestration & GenAI: LangGraph, ChromaDB, Sentence-Transformers, Pydantic, Groq API (Optional)

API & Deployment: FastAPI, Uvicorn, Docker
