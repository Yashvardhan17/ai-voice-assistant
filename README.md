# AI Voice Assistant with Intent Detection

Local synthetic telecom demonstration using Python, scikit-learn, Faster-Whisper, LangGraph, SQLAlchemy/SQLite, Sentence-Transformers/FAISS, Streamlit, pyttsx3, and optional MLflow.

## Windows quick start (PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt
python -m ml.train
python -m pytest -q
uvicorn backend.main:app --reload
```

In **another** activated PowerShell window in the repository root:

```powershell
streamlit run frontend/app.py
```

Open http://localhost:8501 and http://localhost:8000/docs.

Initial first use of speech recognition downloads a Faster-Whisper tiny model; initial RAG query downloads an all-MiniLM-L6-v2 Sentence-Transformer model. Internet access is needed for those one-time downloads.

## Test scenarios

- Customer 1 (Central): "My internet is not working. Create a complaint." → no active outage; ticket created.
- Customer 2 (North): "Check whether there is an outage in my area." → active outage; no ticket.
- Customer 1: "What is my outstanding balance?" → synthetic account balance.
- "How do I restart the router?" → retrieved fictional policy text.

## Caveats

This is a development starter, not a production deployment. Intent examples are small/synthetic; confidence can be unreliable and test-set scores can be misleading. The LangGraph routing and tools are currently deterministic (not an autonomous multi-agent planner). RAG responses are extractive, not LLM-generated. The optional LLM API integration, wider database schema, robust audio evaluation, prompt/version tracing, and cloud deployment still require implementation. Do not claim those features as implemented. No real customer data is used.
