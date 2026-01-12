"""
Streamlit Triage Navigator — Robust Ollama Integration

Features:
- Persistent HTTP session with retries (fast failover)
- Cached health check for Ollama at startup (so UI knows availability fast)
- Short timeouts and exponential backoff on retries
- Deterministic fallback functions if Ollama unavailable
- Minimal LLM calls (follow-up + explanation rewrite)
- Clean Streamlit chat-like UI + structured JSON output

Requirements:
pip install streamlit requests urllib3
Ollama (optional): run a model locally to enable LLM features

By default this tries Ollama at http://localhost:11434.
Change OLLAMA_URL / OLLAMA_MODEL below if needed.
"""

import streamlit as st
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
from datetime import datetime
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time

# -----------------------
# Config: Ollama settings
# -----------------------
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "deepseek-r1:1.5b"  # change to the model name you run locally
OLLAMA_TIMEOUT = 4  # seconds for each request attempt (short to avoid UI hangs)
HEALTHCHECK_TIMEOUT = 1.5  # quick health check

# -----------------------
# Cached session w/ retries
# -----------------------
@st.cache_resource
def get_http_session():
    session = requests.Session()
    # Retry strategy: small number of retries with backoff for transient issues
    retries = Retry(
        total=2,
        backoff_factor=0.5,
        status_forcelist=(500, 502, 503, 504),
        allowed_methods=("POST", "GET")
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

session = get_http_session()

# -----------------------
# Fast Ollama health check 
# -----------------------
def check_ollama_available(force=False) -> bool:
    """
    Returns True if Ollama reachable. Cached in session_state with timestamp.
    Avoid repeated slow checks by caching for ~60 seconds.
    """
    now = time.time()
    cache = st.session_state.get("_ollama_health_cache", {})
    last = cache.get("last_checked", 0)
    if not force and (now - last) < 60 and "available" in cache:
        return cache["available"]

    try:
        # lightweight GET/PING to base URL (some Ollama setups may not respond on '/')
        # use POST health-check endpoint by calling /models or a tiny request
        # Here we attempt a tiny prompt call with tiny timeout just to confirm API reachable.
        resp = session.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": "ping", "stream": False},
            timeout=HEALTHCHECK_TIMEOUT
        )
        available = resp.status_code == 200
    except Exception:
        available = False

    st.session_state["_ollama_health_cache"] = {"available": available, "last_checked": now}
    return available

# Determine availability once at startup for fast UI rendering
if "_ollama_status" not in st.session_state:
    st.session_state["_ollama_status"] = check_ollama_available()

# -----------------------
# Ollama caller with fast failover
# -----------------------
def call_ollama(prompt: str, model: str = OLLAMA_MODEL, timeout: int = OLLAMA_TIMEOUT) -> Optional[str]:
    """
    Calls Ollama with short timeout. Returns response string on success, else None.
    Uses persistent session with limited retries. Catches exceptions quickly.
    """
    try:
        payload = {"model": model, "prompt": prompt, "stream": False}
        resp = session.post(OLLAMA_URL, json=payload, timeout=timeout)
        if resp.status_code == 200:
            # Expected format: {"response": "..."} but be tolerant
            data = resp.json()
            # some Ollama builds might use different key names; try common ones
            text = data.get("response") or data.get("text") or data.get("output") or ""
            if isinstance(text, list):
                text = " ".join(text)
            return text.strip()
        else:
            return None
    except Exception as e:
        # record last error to show in UI if needed
        st.session_state["_last_ollama_error"] = str(e)
        # update cached health to false quickly
        st.session_state["_ollama_status"] = False
        st.session_state["_ollama_health_cache"] = {"available": False, "last_checked": time.time()}
        return None

# -----------------------
# Deterministic fallbacks (instant)
# -----------------------
def fallback_followup_question(symptoms: str) -> str:
    s = symptoms.lower()
    if any(k in s for k in ["chest", "pressure", "tightness", "crushing"]):
        return "Is the chest pain severe, sudden, or radiating to the arm/jaw?"
    if any(k in s for k in ["breath", "breathing", "shortness of breath", "cant breathe"]):
        return "Are you finding it difficult to breathe even at rest?"
    if "fever" in s:
        return "How high is your fever and how long have you had it?"
    return "On a scale from mild to severe, how would you rate your main symptom?"

def fallback_rewrite_explanation(base_expl: str, category: str, red_flags: List[str]) -> str:
    lines = []
    if category == "Emergency":
        lines.append("There are warning signs that require immediate medical attention.")
        if red_flags:
            lines.append("Warning signs detected: " + ", ".join(red_flags) + ".")
    elif category == "Urgent":
        lines.append("Your symptoms are concerning and should be checked soon, ideally within 24 hours.")
    elif category == "Routine":
        lines.append("Symptoms look important but not immediately life-threatening; consider booking a clinic visit.")
    else:
        lines.append("Symptoms appear mild. Home care (rest, fluids) may help; monitor closely.")
    lines.append("This is not a diagnosis. Please consult a medical professional if concerned.")
    return " ".join(lines)

# -----------------------
# Rule-based triage engine 
# -----------------------

@dataclass
class SymptomRule:
    symptom_keyword: str
    emergency_indicators: List[str]
    urgent_indicators: List[str]
    home_care_tip: str
    emergency_score: int
    urgent_score: int
    home_score: int

KNOWLEDGE_BASE = [
    SymptomRule("chest pain",
                emergency_indicators=["severe", "pressure", "crushing", "radiating", "shortness of breath", "sweating"],
                urgent_indicators=["mild but persistent", "worse with effort"],
                home_care_tip="Avoid heavy activity and monitor symptoms.",
                emergency_score=3, urgent_score=2, home_score=1),
    SymptomRule("fever",
                emergency_indicators=["rigors", "confusion", "cannot drink", "breathing fast", "rash"],
                urgent_indicators=["above 103", "more than 3 days", "very weak"],
                home_care_tip="Stay hydrated and rest.",
                emergency_score=3, urgent_score=2, home_score=1),
    SymptomRule("headache",
                emergency_indicators=["sudden worst headache", "vision loss", "weakness", "confusion"],
                urgent_indicators=["worsening", "fever", "neck stiffness"],
                home_care_tip="Rest in a quiet space.",
                emergency_score=3, urgent_score=2, home_score=1),
]

RED_FLAG_PHRASES = [
    "difficulty breathing", "trouble breathing", "shortness of breath",
    "severe chest pain", "unconscious", "passed out", "not waking up",
    "heavy bleeding", "blood in vomit", "black stool"
]

@dataclass
class TriageResult:
    category: str
    score_total: int
    red_flags_detected: List[str]
    matched_symptoms: List[str]
    explanation: str
    next_steps: List[str]
    safety_disclaimer: str

def detect_red_flags(text: str) -> List[str]:
    t = text.lower()
    return [p for p in RED_FLAG_PHRASES if p in t]

def score_symptoms(text: str) -> Dict[str,int]:
    t = text.lower()
    scores = {"emergency":0, "urgent":0, "home":0}
    for rule in KNOWLEDGE_BASE:
        if rule.symptom_keyword in t:
            scores["home"] += rule.home_score
            for w in rule.emergency_indicators:
                if w in t:
                    scores["emergency"] += rule.emergency_score
            for w in rule.urgent_indicators:
                if w in t:
                    scores["urgent"] += rule.urgent_score
    return scores

def classify_urgency(scores: Dict[str,int], red_flags: List[str]) -> str:
    if red_flags:
        return "Emergency"
    if scores["emergency"] >= scores["urgent"] and scores["emergency"] >= 3:
        return "Emergency"
    if scores["urgent"] >= 2:
        return "Urgent"
    if scores["emergency"] == 0 and scores["urgent"] == 0:
        return "Home care"
    return "Routine"

def generate_next_steps(category: str) -> List[str]:
    if category == "Emergency":
        return ["Go to nearest emergency department immediately.", "If unable to travel, call emergency services."]
    if category == "Urgent":
        return ["See a clinician within 24 hours.", "If worse, go to emergency care."]
    if category == "Routine":
        return ["Book a routine clinic visit in the next few days.", "Monitor symptoms and record changes."]
    return ["Try home care (rest, fluids). Seek care if worse."]

# -----------------------
# Triage wrapper that uses LLM 
# -----------------------
def triage(user_text: str, prefer_llm: bool = True) -> TriageResult:
    red_flags = detect_red_flags(user_text)
    scores = score_symptoms(user_text)
    category = classify_urgency(scores, red_flags)
    base_expl = f"Based on rule scores: emergency={scores['emergency']}, urgent={scores['urgent']}."
    matched = [r.symptom_keyword for r in KNOWLEDGE_BASE if r.symptom_keyword in user_text.lower()]
    next_steps = generate_next_steps(category)
    disclaimer = "This is NOT a medical diagnosis. Please consult a medical professional."

    # Decide whether to call LLM:
    use_llm = False
    if prefer_llm and st.session_state.get("_ollama_status", False):
        # do a fast single LLM call but with short timeout via call_ollama
        prompt = f"""
You are a safe assistant. Rewrite this explanation in plain safe language, without diagnosing:
{base_expl}
Category: {category}
Red flags: {red_flags}
Instructions: Keep it short, simple, and include a final "Please consult a medical professional."
"""
        llm_resp = call_ollama(prompt)
        if llm_resp:
            explanation = llm_resp
            use_llm = True
        else:
            explanation = fallback_rewrite_explanation(base_expl, category, red_flags)
            # mark Ollama as unavailable now (fast fallback)
            st.session_state["_ollama_status"] = False
    else:
        explanation = fallback_rewrite_explanation(base_expl, category, red_flags)

    return TriageResult(category=category,
                        score_total=scores["emergency"]+scores["urgent"]+scores["home"],
                        red_flags_detected=red_flags,
                        matched_symptoms=matched,
                        explanation=explanation,
                        next_steps=next_steps,
                        safety_disclaimer=disclaimer)

# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="Triage Navigator (Ollama Integration)", page_icon="", layout="centered")
st.title("Symptoms Care Navigator")
st.caption("Demo triage tool — NOT medical advice or diagnosis.")

# show Ollama status clearly
ollama_ok = st.session_state.get("_ollama_status", False)
if ollama_ok:
    st.success("LLM: Ollama available (local). Responses may be enhanced by LLM.")
else:
    st.info("LLM: Not available — using deterministic fallback responses.")

# session initialization
if "chat" not in st.session_state:
    st.session_state.chat = []
if "symptoms_saved" not in st.session_state:
    st.session_state.symptoms_saved = ""
if "asked_followup" not in st.session_state:
    st.session_state.asked_followup = False

def add_chat(role: str, text: str):
    st.session_state.chat.append({"role": role, "text": text})

# seed assistant message
if not st.session_state.chat:
    add_chat("assistant", "Hello — tell me your main symptom(s). I will ask a short follow-up and then suggest next steps.")

# render chat
for m in st.session_state.chat:
    with st.chat_message(m["role"]):
        st.markdown(m["text"])

# get user input
inp = st.chat_input("Describe your symptoms here...")
if inp:
    add_chat("user", inp)

    if not st.session_state.symptoms_saved:
        # first message: store symptom and ask follow-up
        st.session_state.symptoms_saved = inp

        # Attempt LLM followup generation quickly, else fallback
        followup_prompt = f"Generate one brief, safe follow-up question for: {inp}. Keep it short and non-medical."
        llm_q = None
        if st.session_state.get("_ollama_status", False):
            llm_q = call_ollama(followup_prompt)
            if not llm_q:
                st.session_state["_ollama_status"] = False  # mark it down if failed

        followup_q = llm_q or fallback_followup_question(inp)
        add_chat("assistant", followup_q)
        st.rerun()  # jump UI to show follow-up

    else:
        # answer to follow-up arrived; combine and triage
        combined = st.session_state.symptoms_saved + " " + inp

        result = triage(combined, prefer_llm=True)

        # prepare human-friendly summary
        lines = [
            f"**Triage result:** {result.category}",
            "",
            f"**Explanation:** {result.explanation}",
            "",
            "**Next steps:**"
        ]
        for s in result.next_steps:
            lines.append(f"- {s}")
        lines.append("")
        lines.append(f"**Disclaimer:** {result.safety_disclaimer}")

        add_chat("assistant", "\n".join(lines))

        # show structured JSON
        st.markdown("---")
        st.subheader("🔧 Structured Output (for integration)")
        st.json({
            "timestamp": datetime.utcnow().isoformat(),
            "input_text": combined,
            "triage_result": asdict(result)
        })

        # reset conversation
        st.session_state.symptoms_saved = ""
        st.session_state.asked_followup = False
        st.rerun()
