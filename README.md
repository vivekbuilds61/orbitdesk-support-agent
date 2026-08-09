# OrbitDesk Support Agent

A local **AI-powered customer support agent** built using **LangGraph, Retrieval-Augmented Generation (RAG), Sentence Transformers, and Hugging Face Transformers**.

The system retrieves relevant information from the OrbitDesk knowledge base and resolved support cases, generates a grounded response, verifies the response, and returns a structured JSON result with source citations.

---

## 1. Project Overview

The OrbitDesk Support Agent processes customer support questions through a LangGraph workflow:

```text
User Question
      |
      v
   Triage
      |
      v
  Retrieval
      |
      v
  Generation
      |
      v
 Verification
      |
      v
  Finalize
```

The system is designed to:

* Answer questions using only the supplied knowledge base.
* Retrieve relevant knowledge-base passages.
* Retrieve information from resolved support cases.
* Handle ambiguous questions through clarification.
* Detect unsupported or unsafe requests.
* Escalate cases when required.
* Verify generated answers against retrieved sources.
* Return structured JSON output.

---

## 2. Main Features

### RAG Pipeline

The system uses Retrieval-Augmented Generation to ground answers in the OrbitDesk documentation.

### LangGraph Workflow

The application uses a graph-based workflow:

```text
triage
   |
   +----> clarification
   |
   +----> out_of_scope
   |
   v
retrieve
   |
   v
generate
   |
   v
verify
   |
   +----> retry generation
   |
   +----> safe failure
   |
   v
finalize
```

### Source Citations

Generated responses include the source IDs used to answer the question.

Example:

```json
"sources": [
  {
    "source_id": "KB-003",
    "passage": "Workspace Settings and Timezones..."
  }
]
```

### Verification

The verification stage checks whether the generated answer is grounded in the retrieved documentation.

### Safety Rules

The system detects unsupported requests such as:

* Refund requests
* Legal advice
* Financial advice
* Medical advice
* Requests to reveal secrets
* Requests to ignore system documentation
* Unsupported actions

---

## 3. Technologies

* Python 3.10
* LangGraph
* Hugging Face Transformers
* Sentence Transformers
* PyTorch
* pytest
* JSON
* PowerShell / Windows

---

## 4. Models

### Embedding Model

```text
sentence-transformers/all-MiniLM-L6-v2
```

Used for semantic retrieval and intent classification.

### Generation Model

```text
google/flan-t5-base
```

Used to generate grounded support responses.

The models are downloaded from Hugging Face on the first run and cached locally.

---

## 5. Project Structure

```text
orbitdesk/
│
├── data/
│   ├── knowledge_base/
│   │   ├── 01_product_overview.md
│   │   ├── 02_roles_and_permissions.md
│   │   ├── 03_workspace_settings_and_timezones.md
│   │   ├── 04_scheduled_exports.md
│   │   ├── 05_api_credentials.md
│   │   ├── 06_connections_and_refreshes.md
│   │   ├── 07_delivery_destinations.md
│   │   ├── 08_escalation_and_diagnostics.md
│   │   ├── 09_audit_logs.md
│   │   └── 10_security_and_safe_responses.md
│   │
│   ├── output_schema.json
│   ├── resolved_cases.json
│   └── sample_questions.json
│
├── diagram/
│   ├── graph_diagram.png
│   └── render_diagram.py
│
├── src/
│   ├── __init__.py
│   ├── cli.py
│   ├── corpus.py
│   ├── graph.py
│   ├── models.py
│   ├── retrieval.py
│   ├── state.py
│   │
│   └── nodes/
│       ├── __init__.py
│       ├── finalize.py
│       ├── generate.py
│       ├── retrieve.py
│       ├── templates.py
│       ├── triage.py
│       └── verify.py
│
├── tests/
│   ├── __init__.py
│   └── test_graph_routing.py
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 6. Installation

### Step 1: Clone the repository

```powershell
git clone https://github.com/vivekbuilds61/orbitdesk-support-agent.git
```

### Step 2: Enter the project

```powershell
cd orbitdesk-support-agent
```

### Step 3: Create a virtual environment

```powershell
python -m venv .venv
```

### Step 4: Activate the virtual environment

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

You should see:

```text
(.venv) PS C:\...\orbitdesk-support-agent>
```

### Step 5: Install dependencies

```powershell
pip install -r requirements.txt
```

---

## 7. Running the Application

### Basic question

```powershell
python -m src.cli --question "How do Admins change the workspace timezone?"
```

### Run with trace

```powershell
python -m src.cli --question "How does changing the workspace timezone affect scheduled exports?" --trace
```

The trace shows the executed workflow:

```text
triage
retrieve
generate
verify
finalize
```

---

## 8. Example Questions

### Workspace timezone

```powershell
python -m src.cli --question "How do Admins change the workspace timezone?"
```

### Scheduled exports

```powershell
python -m src.cli --question "How does changing the workspace timezone affect scheduled exports?"
```

### API credentials

```powershell
python -m src.cli --question "How do I create an API credential?"
```

### Troubleshooting

```powershell
python -m src.cli --question "What should I check if a scheduled export fails?"
```

---

## 9. Running Tests

Run all tests:

```powershell
pytest
```

The project includes automated tests for graph routing, including:

* Answerable questions
* Clarification questions
* Escalation cases
* Out-of-scope requests
* Permission-related questions

Example expected result:

```text
9 passed
```

---

## 10. Output Format

The CLI returns structured JSON.

Example:

```json
[
  {
    "question": "How do Admins change the workspace timezone?",
    "classification": "answerable",
    "answer": "Admins can change workspace settings from Settings > Workspace.",
    "sources": [
      {
        "source_id": "KB-003",
        "passage": "Workspace Settings and Timezones..."
      }
    ],
    "confidence": 1.0,
    "requires_human": false,
    "reason": "Best KB match score..."
  }
]
```

---

## 11. Classification Types

### answerable

The knowledge base contains enough information to answer the question.

### requires_clarification

The question is too vague and additional information is required.

Example:

```text
Our data sync is not working. Can you tell me how to fix it?
```

### requires_escalation

The issue meets the escalation conditions defined by the support documentation.

### out_of_scope

The request asks for an unsupported action or information outside the system's scope.

---

## 12. Retrieval Pipeline

The retrieval process works as follows:

```text
Question
   |
   v
Embedding
   |
   v
Knowledge Base + Resolved Cases
   |
   v
Cosine Similarity
   |
   v
Keyword / Semantic Ranking
   |
   v
Top Relevant Passages
```

The retrieval system combines semantic similarity with keyword matching and diversifies results by source.

---

## 13. Corpus Construction

The corpus is constructed from:

### Knowledge Base

Markdown documents stored in:

```text
data/knowledge_base/
```

### Resolved Cases

Support cases stored in:

```text
data/resolved_cases.json
```

Each retrieved chunk contains:

```text
source_id
passage_id
text
kind
status
```

---

## 14. Generation

The generation node creates a response using only retrieved passages.

The prompt instructs the model to:

* Stay grounded in the supplied documentation.
* Avoid unsupported claims.
* Avoid performing unsupported actions.
* Use information from multiple passages when necessary.
* Admit when documentation does not contain enough information.

---

## 15. Verification

The verification node checks the generated response for:

* Grounding
* Source overlap
* Unsupported claims
* Required citations
* Safety restrictions

If verification fails, the system can retry generation.

The maximum generation attempts are:

```text
2
```

This provides:

```text
Initial generation
       |
       v
Verification
       |
       +---- Pass ----> Finalize
       |
       +---- Fail ----> Retry
                         |
                         v
                    Verification
```

---

## 16. Safety

The system does not attempt to perform unsupported actions.

Examples include:

```text
Issue a refund
Reveal an API credential secret
Provide legal advice
Provide financial advice
Provide medical advice
Contact a customer on behalf of the user
```

Credential secrets are never intended to be stored in source control or exposed in support responses.

---

## 17. Environment Variables

The project supports selecting between real and stub model backends.

### Real backend

```powershell
$env:ORBITDESK_MODEL_BACKEND="real"
```

### Stub backend

```powershell
$env:ORBITDESK_MODEL_BACKEND="stub"
```

The default backend is:

```text
real
```

The stub backend is useful for running routing tests without loading Hugging Face models.

---

## 18. GitHub

Repository:

```text
https://github.com/vivekbuilds61/orbitdesk-support-agent
```

The repository contains:

* Source code
* Knowledge base
* Tests
* Architecture diagram
* Requirements
* Documentation

The virtual environment is excluded using `.gitignore`.

---

## 19. Development

After modifying the code:

```powershell
git status
```

Stage changes:

```powershell
git add .
```

Commit:

```powershell
git commit -m "Update OrbitDesk support agent"
```

Push:

```powershell
git push origin main
```

---

## 20. Known Limitation

The generation model used by the project is:

```text
google/flan-t5-base
```

Because it is a relatively small local instruction model, some responses may be more extractive than a larger instruction-tuned model.

The system prioritizes **grounded and verifiable responses** over unsupported generation.

---

## 21. Future Improvements

Possible future enhancements include:

* FAISS or another vector database
* BM25 + dense hybrid retrieval
* Larger instruction-tuned generation model
* Streaming responses
* Web-based interface
* Conversation memory
* Evaluation metrics
* API deployment using FastAPI
* Docker deployment
* GPU acceleration

---

## 22. Author

**Vivek Jalakote**

Artificial Intelligence & Machine Learning

GitHub:

https://github.com/vivekbuilds61

---

## License

This project was developed as an AI/ML support-agent implementation and assessment project.
