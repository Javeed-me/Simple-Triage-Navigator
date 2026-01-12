**Symptoms Triage Navigator**

A hybrid AI-based healthcare triage demo that combines:
- A **rule-based medical triage engine** (for safety and determinism)
- A **local open-source LLM via Ollama** (for better explanations and follow-up questions)
 
 >>> It does **not** provide medical diagnosis or medical advice.



**Prerequisites**

Make sure you have:

- Python 3.9+
- Ollama installed  
  (https://ollama.com/download)



**How to Run**

1. Open the file `appoll.py`

2. Install the required Python libraries:
```bash
pip install streamlit requests urllib3

