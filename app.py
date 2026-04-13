# ===============================
# Program Fit Explorer (Streamlit)
# Minimal update focusing on:
# 1) Add `background_type` (domain bucket) right after `academic_background`
# 2) Pie chart for decision breakdown + horizontal bar for background types (sorted desc)
# 3) Richer core-skills extraction (more than one)
#
# Install (same env where you run this):
#   pip install streamlit pandas llama-cpp-python plotly
# Run:
#   streamlit run streamlit_app.py
# ===============================

import json
import re
import random
from pathlib import Path
from datetime import datetime
from io import StringIO

import pandas as pd
import streamlit as st
from llama_cpp import Llama

# Charts
try:
    import plotly.express as px
    HAS_PLOTLY = True
except Exception:
    HAS_PLOTLY = False

# ===============================
# GLOBAL CONFIG
# ===============================
CTX_LEN = 4096
N_THREADS = 0
N_GPU_LAYERS = 9999
SEED = 42
random.seed(SEED)

TEMPERATURE = 0.5
TOP_P = 0.9
MAX_NEW_TOKENS = 220

REQUIRED_COLUMNS = [
    "academic_background",
    "academic_interests",
    "professional_interests",
    "previous_work_experience",
]

MAX_WORDS = 65
MIN_SENTS, MAX_SENTS = 1, 3

JSON_OBJ = re.compile(r"\{.*\}", re.DOTALL)
FIRST_PERSON_RE = re.compile(r"\b(I|I'm|I’ve|I’d|my|me)\b", re.IGNORECASE)

STOPWORDS = {
    "the","and","of","in","for","to","a","an","on","with","at","by",
    "from","into","about","as","is","are","this","that","these","those",
    "my","our","your","their","through","using","use","based"
}

TONE_HINTS = [
    "sound cautiously optimistic but honest about gaps",
    "sound very practical and straightforward",
    "sound quietly confident but not overselling yourself",
    "sound a bit cautious, focusing on what you still need to learn",
    "sound reflective, connecting the program to your long-term goals",
]

OPENING_HINTS = [
    "You can start from a project, a class, or a goal that feels most relevant.",
    "You can start from a specific academic interest or course that connects to the program.",
    "You can start from your professional or internship experiences.",
    "You can start from a gap you notice in your skills and how that affects your decision.",
]

BANNED_PREFIXES = [
    "strict json", "convert this", "rewrite the explanation", "original explanation",
]

# ===============================
# DOMAINS / LABELS
# ===============================
DOMAIN_BUCKET_LABELS = {
    "arts_design_media": "Arts, Design & Media",
    "business": "Business",
    "computer_it": "Computer & IT",
    "data": "Data",
    "education": "Education",
    "engineering": "Engineering",
    "environmental": "Environmental & Sustainability",
    "health_life_sciences": "Health & Life Sciences",
    "interdisciplinary_bioinfo": "Interdisciplinary (Bioinformatics)",
    "math_stats": "Mathematics & Statistics",
    "natural_sciences": "Natural Sciences",
    "other_humanities": "Other (History, Philosophy)",
    "public_policy": "Public Policy & Government",
    "social_sciences": "Social Sciences",
    "sports_performance": "Sports & Human Performance",
    "urban_arch_planning": "Urban, Architecture & Planning",
    "other": "Other / Interdisciplinary",
    "unknown": "Unknown",
}
DOMAIN_CODES_FOR_LLM = [k for k in DOMAIN_BUCKET_LABELS.keys() if k not in ("unknown",)]
DEGREE_OPTIONS = ["Bachelors/undergraduate", "Masters", "PhD"]

# ===============================
# SESSION MODEL
# ===============================
_LLM_CACHE = {}

def get_llm(model_path: str) -> Llama:
    model_path = (model_path or "").strip()
    if not model_path:
        raise ValueError("Model path is empty.")
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")
    key = (model_path, CTX_LEN, N_THREADS, N_GPU_LAYERS, SEED)
    if key in _LLM_CACHE:
        return _LLM_CACHE[key]
    llm_obj = Llama(
        model_path=model_path,
        n_ctx=CTX_LEN,
        n_threads=N_THREADS,
        n_gpu_layers=N_GPU_LAYERS,
        verbose=False,
        seed=SEED,
    )
    _LLM_CACHE[key] = llm_obj
    return llm_obj

def chat_once(system_prompt: str, user_prompt: str,
              temperature=TEMPERATURE, top_p=TOP_P, max_tokens=MAX_NEW_TOKENS) -> str:
    llm_obj = st.session_state.get("llm", None)
    if llm_obj is None:
        raise RuntimeError("Model is not loaded. Click ‘Run evaluation’ first.")
    out = llm_obj.create_chat_completion(
        messages=[
            {"role":"system","content":system_prompt},
            {"role":"user","content":user_prompt},
        ],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    return out["choices"][0]["message"]["content"]

# ===============================
# HELPERS
# ===============================
def tokenize(text: str):
    tokens = re.split(r"[^a-zA-Z]+", (text or "").lower())
    return [t for t in tokens if len(t) > 2 and t not in STOPWORDS]

def trim_words(s: str, max_words=MAX_WORDS) -> str:
    w = (s or "").split()
    if not w:
        return ""
    s2 = " ".join(w[:max_words + 10])
    if not s2.endswith((".", "!", "?")):
        s2 += "."
    return s2

def extract_degree_type(desc: str) -> str:
    text = (desc or "").lower()
    if any(tok in text for tok in ["phd","ph.d","doctor of philosophy","doctoral program","doctoral degree"]):
        return "PhD"
    if any(tok in text for tok in ["master of science","master of arts","master’s","master's","ms in","m.s. in","msc","graduate program","graduate degree"]):
        return "Masters"
    if any(tok in text for tok in ["bachelor of","undergraduate program","undergraduate degree","b.sc","b.s.","ba in","b.a. in"]):
        return "Bachelors/undergraduate"
    return "Masters"

def classify_domain_from_text(text: str) -> str:
    t = (text or "").lower()
    if any(k in t for k in ["data science","data analytics","analytics","data engineering","machine learning","artificial intelligence","ai","business analytics"]):
        return "data"
    if any(k in t for k in ["computer science","software engineering","information systems","information technology","it","cybersecurity"]):
        return "computer_it"
    if any(k in t for k in ["statistics","statistical","biostatistics","mathematics","applied math","math and statistics"]):
        return "math_stats"
    if any(k in t for k in ["civil engineering","mechanical engineering","electrical engineering","industrial engineering","chemical engineering","engineering program"]):
        return "engineering"
    if "economics" in t or "econometric" in t:
        return "business"
    if any(k in t for k in ["finance","financial engineering","quantitative finance","financial analysis"]):
        return "business"
    if any(k in t for k in ["business administration","business management","mba","marketing","accounting","supply chain management","operations management"]):
        return "business"
    if any(k in t for k in ["public policy","public administration","urban planning","governance"]):
        return "public_policy"
    if any(k in t for k in ["health informatics","nursing","public health","biomedical","clinical","healthcare"]):
        return "health_life_sciences"
    if any(k in t for k in ["psychology","sociology","international relations","political science","social science"]):
        return "social_sciences"
    if any(k in t for k in ["graphic design","journalism","media studies","visual arts"]):
        return "arts_design_media"
    if "environmental science" in t or "sustainability" in t:
        return "environmental"
    if "sports science" in t or "sport science" in t:
        return "sports_performance"
    if "architecture" in t or "urban planning" in t:
        return "urban_arch_planning"
    if any(k in t for k in ["history","philosophy","literature"]):
        return "other_humanities"
    return "other" if t.strip() else "unknown"

def regex_program_name(desc: str) -> str | None:
    text = desc.strip()
    m = re.search(r"(Master of [A-Za-z ]+ in [^,\.]+)", text)
    if m: return m.group(1).strip()
    m2 = re.search(r"(Bachelor of [A-Za-z ]+ in [^,\.]+)", text)
    if m2: return m2.group(1).strip()
    m3 = re.search(r"(Doctor of Philosophy in [^,\.]+)", text)
    if m3: return m3.group(1).strip()
    m4 = re.search(r"In ([^.]+?program)", text)
    if m4: return m4.group(1).strip()
    return None

# ===== Expanded rule-based core skills & focus areas =====
CORE_SKILL_PATTERNS = [
    (["statistics","statistical"], "statistics"),
    (["probability"], "probability"),
    (["linear algebra"], "linear algebra"),
    (["calculus"], "calculus"),
    ([" python ","python,"], "programming in Python"),
    ([" r "," r,"], "programming in R"),
    (["sql"], "SQL / data querying"),
    (["machine learning","ml"], "machine learning"),
    (["predictive modeling","predictive modelling"], "predictive modeling"),
    (["data analysis","data analytic"], "data analysis"),
    (["data visualization","visualization"], "data visualization"),
    (["big data","large datasets"], "working with large datasets"),
    (["database","data management","etl"], "data management & ETL"),
    (["cloud","aws","azure","gcp"], "cloud platforms for data/ML"),
    (["research methods","research design","experimental design"], "research & experimental design"),
    (["communication of complex","communicate complex","present complex"], "communication of analytical findings"),
    (["optimization","operations research"], "optimization / OR"),
]

FOCUS_STRONG_PATTERNS = [
    (["design and implement analytics projects","designing and implementing analytics projects"], "designing & implementing analytics projects"),
    (["solve complex organizational problems","complex organizational problems"], "solving complex organizational problems"),
    (["decision-making skills","decision making skills","data-driven decisions"], "decision-making with data"),
    (["project management"], "project management for analytics"),
    (["knowledge discovery"], "knowledge discovery"),
    (["manage and disseminate","management and dissemination"], "managing & disseminating analytical knowledge"),
    (["communication of complex data","communicate complex data"], "communicating complex data"),
]

def rule_based_core_skills(program_description: str, max_items: int = 8) -> list[str]:
    text = " " + (program_description or "").lower() + " "
    found = []
    for triggers, label in CORE_SKILL_PATTERNS:
        if any(t in text for t in triggers):
            found.append(label)
    found = list(dict.fromkeys(found))
    return found[:max_items]

def rule_based_focus_areas(program_description: str, max_items: int = 5) -> list[str]:
    text = " " + (program_description or "").lower() + " "
    found = []
    for triggers, label in FOCUS_STRONG_PATTERNS:
        if any(t in text for t in triggers):
            found.append(label)
    found = list(dict.fromkeys(found))
    return found[:max_items]

# ===== LLM fallback extractors =====
def extract_list_fallback(program_description: str, instruction: str, low=3, high=6) -> list[str]:
    sys_p = (
        "You extract short items from a graduate program description. "
        "Return STRICT JSON as an array of concise strings (no sentences)."
    )
    usr_p = (
        f"{instruction}\n\n"
        f"Program description:\n{program_description}\n\n"
        f"Return STRICT JSON array with {low}-{high} items, e.g. [\"item1\",\"item2\"]."
    )
    txt = chat_once(sys_p, usr_p, temperature=0.0, top_p=1.0, max_tokens=220)
    m = JSON_OBJ.search(txt or "")
    cand = m.group(0) if m else (txt or "")
    try:
        arr = json.loads(cand)
        if isinstance(arr, list):
            items = [str(x).strip() for x in arr if str(x).strip()]
            return items[:high]
    except Exception:
        pass
    return []

def ensure_core_prereqs(meta: dict, program_description: str) -> list[str]:
    core = meta.get("core_prerequisites", []) or []
    if core:
        return core
    rb = rule_based_core_skills(program_description)
    if rb:
        return rb
    fb = extract_list_fallback(
        program_description,
        "List the core skills or prior knowledge students should have (e.g., statistics, probability, linear algebra, Python/R, SQL, data visualization).",
        low=4, high=8,
    )
    return fb

def ensure_focus_areas(meta: dict, program_description: str) -> list[str]:
    fa = meta.get("focus_areas", []) or []
    if fa:
        return fa
    rb = rule_based_focus_areas(program_description)
    if rb:
        return rb
    fb = extract_list_fallback(
        program_description,
        "List the program's key focus areas/themes (concise noun phrases).",
        low=3, high=5,
    )
    return fb

# ===== Program metadata =====
def parse_program_metadata_json(text: str):
    m = JSON_OBJ.search(text or "")
    cand = m.group(0) if m else (text or "")
    try:
        data = json.loads(cand)
        if not isinstance(data, dict): return None
        def norm_list(x):
            if isinstance(x, list):
                return [str(v).strip() for v in x if str(v).strip()]
            if isinstance(x, str) and x.strip():
                return [p.strip() for p in x.split(",") if p.strip()]
            return []
        name = str(data.get("program_name","")).strip()
        domain = str(data.get("program_domain","")).strip()
        if domain not in DOMAIN_BUCKET_LABELS:
            domain = "other"
        return {
            "program_name": name or "this graduate program",
            "program_domain": domain,
            "focus_areas": norm_list(data.get("focus_areas", [])),
            "core_prerequisites": norm_list(data.get("core_prerequisites", [])),
            "preferred_backgrounds": norm_list(data.get("preferred_backgrounds", [])),
        }
    except Exception:
        return None

def extract_program_metadata(program_description: str):
    system_p = (
        "You are analyzing a graduate program description. "
        "Return STRICT JSON with keys: program_name, program_domain, focus_areas, core_prerequisites, preferred_backgrounds. "
        "program_domain must be one of: " + ", ".join([f'"{d}"' for d in DOMAIN_CODES_FOR_LLM]) + "."
    )
    user_p = (
        "Extract concise metadata. Prefer short phrases over sentences.\n\n"
        f"{program_description}\n\n"
        "Return STRICT JSON with that schema only."
    )
    txt = chat_once(system_p, user_p, temperature=0.0, top_p=1.0, max_tokens=256)
    meta = parse_program_metadata_json(txt)

    if meta is None:
        rep_txt = chat_once(
            "You convert to strict JSON with keys {program_name, program_domain, focus_areas, core_prerequisites, preferred_backgrounds} only.",
            "Convert this to STRICT JSON with only those keys:\n\n" + (txt or ""),
            temperature=0.0, top_p=1.0, max_tokens=220,
        )
        meta = parse_program_metadata_json(rep_txt)

    if meta is None:
        fallback_name = regex_program_name(program_description) or "this graduate program"
        meta = {
            "program_name": fallback_name,
            "program_domain": "other",
            "focus_areas": [],
            "core_prerequisites": [],
            "preferred_backgrounds": [],
        }

    # Improve program_name via regex if it's generic
    if meta["program_name"] == "this graduate program":
        rn = regex_program_name(program_description)
        if rn:
            meta["program_name"] = rn

    # Domain override by rule if LLM gave weak domain
    combined_text = meta["program_name"] + " " + program_description
    rule_domain = classify_domain_from_text(combined_text)
    if meta["program_domain"] in ("other","unknown") and rule_domain not in ("unknown",):
        meta["program_domain"] = rule_domain

    # Ensure core skills and focus areas filled (richer now)
    meta["core_prerequisites"] = ensure_core_prereqs(meta, program_description)
    meta["focus_areas"] = ensure_focus_areas(meta, program_description)
    return meta

# ===============================
# DECISION GENERATION (unchanged logic)
# ===============================
def parse_decision_json_simple(text: str):
    m = JSON_OBJ.search(text or "")
    cand = m.group(0) if m else (text or "")
    try:
        data = json.loads(cand)
        if not isinstance(data, dict): return None
        if "decision" not in data or "explanation" not in data: return None
        d = str(data["decision"]).strip().lower()
        decision = "Yes" if d.startswith("y") else ("No" if d.startswith("n") else str(data["decision"]))
        explanation = str(data["explanation"]).strip()
        return {"decision": decision, "explanation": explanation}
    except Exception:
        return None

def build_program_summary(program_meta: dict, degree_type: str) -> str:
    parts = []
    name = program_meta.get("program_name","this graduate program")
    domain_code = program_meta.get("program_domain","other")
    domain_label = DOMAIN_BUCKET_LABELS.get(domain_code, "Other / Interdisciplinary")
    core = program_meta.get("core_prerequisites",[])
    focus = program_meta.get("focus_areas",[])
    parts.append(f"Name: {name}")
    parts.append(f"Degree type: {degree_type}")
    parts.append(f"Domain: {domain_label}")
    if focus:
        parts.append("Key themes: " + ", ".join(focus))
    if core:
        parts.append("Key skills or knowledge expected: " + ", ".join(core))
    return " | ".join(parts)

def build_decision_system_prompt(program_meta: dict, degree_type: str) -> str:
    name = program_meta.get("program_name","this graduate program")
    domain_code = program_meta.get("program_domain","other")
    domain_label = DOMAIN_BUCKET_LABELS.get(domain_code, "Other / Interdisciplinary")
    return (
        f"You are the internal voice of a prospective student deciding whether to apply to "
        f"\"{name}\" (degree type: {degree_type}, domain: {domain_label}).\n"
        "You must return STRICT JSON with keys {decision, explanation}.\n"
        "decision: 'Yes' or 'No'.\n"
        "explanation: 1–3 sentences, first-person (I/me/my), around 55 words and ≤65 words.\n"
        "Avoid copying exact wording from instructions or summary; vary openings across students."
    )

def build_decision_user_prompt(program_description: str, program_meta: dict,
                               degree_type: str, profile: dict) -> str:
    summary = build_program_summary(program_meta, degree_type)
    tone = random.choice(TONE_HINTS)
    opening = random.choice(OPENING_HINTS)
    return (
        "Decide if this student would realistically want to apply and whether they seem reasonably prepared.\n"
        f"Tone: {tone}.\n"
        f"{opening}\n\n"
        "Program summary:\n"
        f"{summary}\n\n"
        "Program description (for context):\n"
        f"{program_description}\n\n"
        "Student profile (JSON):\n"
        f"{json.dumps(profile, ensure_ascii=False)}\n\n"
        "Return STRICT JSON with this structure:\n"
        "{\n"
        '  "decision": "Yes" or "No",\n'
        '  "explanation": "1–3 sentences, first-person, ≤65 words, varied phrasing"\n'
        "}"
    )

def remove_prompt_artifacts(text: str) -> str:
    if not text: return text
    t = text.strip()
    parts = re.split(r'([.!?])', t)
    cleaned_chunks = []
    for i in range(0, len(parts), 2):
        chunk = parts[i].strip()
        if not chunk: continue
        lower = chunk.lower()
        if any(lower.startswith(bp) for bp in BANNED_PREFIXES):
            continue
        cleaned_chunks.append(chunk)
        if i + 1 < len(parts) and parts[i+1] in ".!?":
            cleaned_chunks.append(parts[i+1])
    cleaned = "".join(cleaned_chunks).strip()
    return cleaned or t

def ensure_first_person(s: str) -> str:
    if FIRST_PERSON_RE.search(s or ""):
        return s
    t = re.sub(r"^The student\b", "I", s or "", flags=re.IGNORECASE)
    if t != (s or ""):
        return t
    return "From my perspective, " + (s or "").lstrip()

def fix_grammar_minimal(s: str) -> str:
    if not s: return s
    t = s.strip()
    t = re.sub(r"\b[Ii]\s+my\b", "In my", t)
    t = re.sub(r"\bi['’`]?m\b", "I'm", t, flags=re.IGNORECASE)
    t = re.sub(r"\bi\b", "I", t)
    t = re.sub(r"\bwith a experience\b", "with experience", t, flags=re.IGNORECASE)
    t = re.sub(r"\ba experience\b", "an experience", t, flags=re.IGNORECASE)
    t = re.sub(r"\binterested for\b", "interested in", t, flags=re.IGNORECASE)
    t = re.sub(r"\s+([,.!?])", r"\1", t)
    t = re.sub(r"([.!?])([A-Za-z])", r"\1 \2", t)
    t = re.sub(r"\s+", " ", t).strip()
    if not t.endswith((".", "!", "?")):
        t += "."
    return t

def limit_sentences(s: str) -> str:
    text = (s or "").strip()
    if not text: return text
    parts = re.split(r'([.!?])', text)
    out_parts, count = [], 0
    for i in range(0, len(parts), 2):
        chunk = parts[i].strip()
        if not chunk: continue
        if count >= MAX_SENTS: break
        out_parts.append(chunk)
        if i + 1 < len(parts) and parts[i+1] in ".!?":
            out_parts.append(parts[i+1])
        else:
            out_parts.append(".")
        count += 1
    out = "".join(out_parts).strip()
    return trim_words(out, MAX_WORDS)

def maybe_rewrite_opening_for_variety(s: str, prob: float = 0.8) -> str:
    text = s.lstrip()
    lower = text.lower()
    candidates = [
        "my background in ", "with a background in ", "with my background in ",
        "given my background in ", "while my background is ", "while my background in ", "as a ",
    ]
    if any(lower.startswith(c) for c in candidates) and random.random() < prob:
        alt_openers = [
            "Studying ", "Working in ", "Spending time in ",
            "My studies in ", "Over the past few years, ",
            "Looking ahead, ", "Because I’ve spent time in ", "From my perspective, ",
        ]
        alt = random.choice(alt_openers)
        match = next(c for c in candidates if lower.startswith(c))
        new = alt + text[len(match):]
        leading_ws = s[:len(s) - len(s.lstrip())]
        return leading_ws + new
    return s

def postprocess_explanation(expl: str) -> str:
    t = (expl or "").strip()
    t = remove_prompt_artifacts(t)
    t = ensure_first_person(t)
    t = fix_grammar_minimal(t)
    t = limit_sentences(t)
    t = maybe_rewrite_opening_for_variety(t)
    return t

def generate_decision(program_description: str, program_meta: dict,
                      degree_type: str, profile: dict) -> dict:
    sys_p = build_decision_system_prompt(program_meta, degree_type)
    usr_p = build_decision_user_prompt(program_description, program_meta, degree_type, profile)
    txt = chat_once(sys_p, usr_p)
    data = parse_decision_json_simple(txt)
    if data is None:
        simple_expl = chat_once(
            "You are the student's internal voice. Output ONLY 1–3 first-person sentences about whether they would apply.",
            "Student profile:\n" + json.dumps(profile, ensure_ascii=False) +
            "\nProgram description:\n" + program_description,
            temperature=0.6, top_p=0.9, max_tokens=140,
        )
        expl = postprocess_explanation(simple_expl)
        return {"decision": "No", "explanation": expl}
    return data

# ===============================
# SIMPLE FIT / CONFIDENCE
# ===============================
def inferred_rigor_from_background(bg: str) -> float:
    text = (bg or "").lower()
    high = [
        "mathematics","math","statistics","biostatistics",
        "computer science","cs","software","engineering",
        "physics","data science","data analytics",
        "economics","econometric","quantitative","machine learning",
        "operations research","actuarial","information technology","it",
        "bioinformatics"
    ]
    medium = [
        "business","management","psychology","biology",
        "public policy","sociology","marketing","accounting",
        "chemistry","environmental","health","political science",
        "finance","supply chain","education","urban planning"
    ]
    if any(w in text for w in high): return 0.8
    if any(w in text for w in medium): return 0.5
    if text.strip(): return 0.2
    return 0.0

def prereq_skill_fit(profile: dict, program_meta: dict) -> float:
    text_all = " ".join(str(profile.get(k,"")) for k in profile.keys())
    tokens_profile = set(tokenize(text_all))
    core = [c for c in (program_meta.get("core_prerequisites",[]) or []) if c]
    hits = 0
    for c in core:
        tokens_c = set(tokenize(c))
        if tokens_c and tokens_profile.intersection(tokens_c):
            hits += 1
    base = hits / len(core) if core else 0.0
    rigor = inferred_rigor_from_background(profile.get("academic_background",""))
    return max(0.0, min(1.0, 0.6*base + 0.4*rigor))

def interest_fit(profile: dict, program_meta: dict) -> float:
    ai = (profile.get("academic_interests","") or "")
    pi = (profile.get("professional_interests","") or "")
    tokens_int = set(tokenize(ai + " " + pi))
    if not tokens_int: return 0.0
    kw = []
    kw += tokenize(program_meta.get("program_name",""))
    kw += tokenize(DOMAIN_BUCKET_LABELS.get(program_meta.get("program_domain",""),""))
    for fa in program_meta.get("focus_areas", []):
        kw += tokenize(fa)
    kw = [k for k in kw if k]
    if not kw: return 0.0
    kw_unique = list(dict.fromkeys(kw))
    hits = sum(1 for k in kw_unique if k in tokens_int)
    return max(0.0, min(1.0, hits / len(kw_unique)))

def experience_score(profile: dict) -> float:
    pw = str(profile.get("previous_work_experience","") or "").strip().lower()
    if pw in {"yes","y","true","1"}: return 1.0
    if len(pw) >= 3: return 0.7
    return 0.0

def program_alignment_score(skill_fit: float, interest_fit_value: float, exp_score: float) -> float:
    return max(0.0, min(1.0, 0.5*skill_fit + 0.3*interest_fit_value + 0.2*exp_score))

def model_confidence_from_alignment(decision: str, skill_fit_value: float, interest_fit_value: float) -> float:
    avg_fit = 0.6*skill_fit_value + 0.4*interest_fit_value
    return (0.5 + 0.4*avg_fit) if decision == "Yes" else (0.5 + 0.4*(1.0-avg_fit))

def blend_and_interval(model_score: float, align_score: float) -> tuple[float, float]:
    blend = 0.5*model_score + 0.5*align_score
    d = abs(model_score - align_score)
    width = max(0.10, min(0.30, 0.25 - 0.10 * (1 - d)))
    return blend, width

def percent_interval_str(center: float, width: float) -> str:
    lo = max(0.0, center - width/2)
    hi = min(1.0, center + width/2)
    return f"{int(round(lo*100))}–{int(round(hi*100))}%"

def calibrated_decision(initial_decision: str, skill_fit_value: float,
                        interest_fit_value: float, alignment: float) -> str:
    if alignment < 0.20 and skill_fit_value < 0.30: return "No"
    if alignment > 0.60 and interest_fit_value > 0.35: return "Yes"
    return initial_decision

# ===============================
# BACKGROUND BUCKET
# ===============================
def background_bucket(bg: str) -> str:
    text = (bg or "").lower()
    if any(k in text for k in ["data science","data analytics","data engineering","analytics","machine learning","artificial intelligence","ai"]):
        return "data"
    if any(k in text for k in ["computer science","software engineering","information systems","information technology","cybersecurity","it"]):
        return "computer_it"
    if any(k in text for k in ["statistics","statistical","biostatistics","mathematics","applied math"]):
        return "math_stats"
    if any(k in text for k in ["civil engineering","mechanical engineering","electrical engineering","industrial engineering","chemical engineering","engineering"]):
        return "engineering"
    if "economics" in text or "econometric" in text:
        return "business"
    if "finance" in text or "financial" in text:
        return "business"
    if any(k in text for k in ["business","management","mba","marketing","accounting","supply chain"]):
        return "business"
    if any(k in text for k in ["public policy","public administration","urban planning","governance"]):
        return "public_policy"
    if any(k in text for k in ["health informatics","nursing","public health","biomedical","clinical","healthcare"]):
        return "health_life_sciences"
    if any(k in text for k in ["psychology","sociology","international relations","political science","social science"]):
        return "social_sciences"
    if any(k in text for k in ["graphic design","journalism","media","visual arts"]):
        return "arts_design_media"
    if "environmental science" in text or "sustainability" in text:
        return "environmental"
    if "sports science" in text or "sport science" in text:
        return "sports_performance"
    if "architecture" in text or "urban planning" in text:
        return "urban_arch_planning"
    if "bioinformatics" in text:
        return "interdisciplinary_bioinfo"
    if any(k in text for k in ["biology","chemistry","physics","agriculture","geography"]):
        return "natural_sciences"
    if any(k in text for k in ["history","philosophy","literature"]):
        return "other_humanities"
    return "other" if text.strip() else "unknown"

# ===============================
# EVALUATION PIPELINE
# ===============================
def evaluate_profiles(df: pd.DataFrame, program_description: str,
                      program_meta: dict | None = None,
                      degree_type: str | None = None):
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}\nFound columns: {list(df.columns)}")

    if program_meta is None:
        program_meta = extract_program_metadata(program_description)
    if degree_type is None:
        degree_type = extract_degree_type(program_description)

    rows = []
    for i, row in df.iterrows():
        profile = {
            "academic_background": str(row["academic_background"]),
            "academic_interests": str(row["academic_interests"]),
            "professional_interests": str(row["professional_interests"]),
            "previous_work_experience": str(row["previous_work_experience"]),
        }
        data = generate_decision(program_description, program_meta, degree_type, profile)
        model_decision = "Yes" if data.get("decision","Yes").strip().lower().startswith("y") else "No"
        expl = postprocess_explanation(data.get("explanation",""))

        skill = prereq_skill_fit(profile, program_meta)
        intr = interest_fit(profile, program_meta)
        exp_score = experience_score(profile)
        align = program_alignment_score(skill, intr, exp_score)
        final_decision = calibrated_decision(model_decision, skill, intr, align)

        model_score_raw = model_confidence_from_alignment(final_decision, skill, intr)
        conf_center, width = blend_and_interval(model_score_raw, align)

        rows.append({
            "row_index": i,
            "decision": final_decision,
            "explanation": expl,
            "model_confidence": int(round(model_score_raw*100)),
            "program_alignment": int(round(align*100)),
            "certainty_range": percent_interval_str(conf_center, width),
            "confidence_center": conf_center,
            "confidence_width": width,
        })

    out_df = pd.DataFrame(rows).set_index("row_index").sort_index()
    final_df = df.join(out_df, how="left")

    # ===== (1) Insert background_type immediately after academic_background =====
    bt_series = final_df["academic_background"].apply(background_bucket).map(
        lambda c: DOMAIN_BUCKET_LABELS.get(c, "Other / Interdisciplinary")
    )
    cols = list(final_df.columns)
    if "background_type" not in cols:
        insert_pos = cols.index("academic_background") + 1 if "academic_background" in cols else 1
        cols = cols[:insert_pos] + ["background_type"] + cols[insert_pos:]
    final_df = final_df.reindex(columns=cols)
    final_df["background_type"] = bt_series

    return final_df, program_meta, degree_type

# ===============================
# STREAMLIT UI
# ===============================
st.set_page_config(page_title="Program Fit Explorer", layout="wide")
st.sidebar.title("Program Fit Explorer")

model_path = st.sidebar.text_input("Model path (.gguf)", value="", help="Full path to your local GGUF model.")
uploaded_program = st.sidebar.file_uploader("Upload Program Description (TXT)", type=["txt"])
uploaded_csv = st.sidebar.file_uploader("Upload Student Profiles (CSV)", type=["csv"])
run_button = st.sidebar.button("▶️ Run evaluation")
st.sidebar.markdown("---")
st.sidebar.write("Required CSV columns:")
st.sidebar.code(", ".join(REQUIRED_COLUMNS), language="text")

# Session state init
for key, default in [
    ("results_df", None),
    ("program_meta", None),
    ("degree_type", None),
    ("original_df", None),
    ("program_text", None),
    ("llm", None),
    ("model_path", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

if run_button:
    try:
        if not model_path:
            st.error("Please provide a model path.")
        elif uploaded_program is None or uploaded_csv is None:
            st.error("Please upload both program description (TXT) and student profiles (CSV).")
        else:
            with st.spinner("Loading model…"):
                model = get_llm(model_path)
                st.session_state.llm = model
                st.session_state.model_path = model_path

            program_text = uploaded_program.read().decode("utf-8", errors="ignore").strip()
            df = pd.read_csv(uploaded_csv)

            if df.empty:
                st.error("CSV appears empty.")
            elif len(program_text) < 20:
                st.error("Program description is too short.")
            else:
                with st.spinner("Evaluating student profiles…"):
                    results_df, program_meta, degree_type = evaluate_profiles(df, program_text)

                st.session_state.results_df = results_df
                st.session_state.program_meta = program_meta
                st.session_state.degree_type = degree_type
                st.session_state.original_df = df
                st.session_state.program_text = program_text

                st.success("Evaluation complete.")
    except Exception as e:
        st.error(f"Error during evaluation: {e}")

results_df = st.session_state.results_df
program_meta = st.session_state.program_meta
degree_type = st.session_state.degree_type

st.title("Program Fit Dashboard")

if results_df is None:
    st.info("Upload files and click **Run evaluation** to see results.")
else:
    tab1, tab2, tab3 = st.tabs(["Program overview", "Metrics Breakdown", "Results table"])

    # ---- TAB 1 ----
    with tab1:
        st.subheader("Program Summary (editable)")

        if program_meta is None:
            st.info("No program metadata available yet.")
        else:
            name = program_meta.get("program_name","this graduate program")
            domain_code = program_meta.get("program_domain","other")
            focus = program_meta.get("focus_areas",[])
            core = program_meta.get("core_prerequisites",[])
            preferred = program_meta.get("preferred_backgrounds",[])

            domain_label_current = DOMAIN_BUCKET_LABELS.get(domain_code, "Other / Interdisciplinary")
            core_text = "\n".join(core)
            focus_text = "\n".join(focus)

            with st.form("meta_form"):
                col_a, col_b = st.columns(2)
                with col_a:
                    prog_name_edit = st.text_input("Program name", value=name)
                    degree_edit = st.selectbox(
                        "Degree type",
                        options=DEGREE_OPTIONS,
                        index=DEGREE_OPTIONS.index(degree_type) if degree_type in DEGREE_OPTIONS else 1
                    )
                with col_b:
                    domain_label_options = [DOMAIN_BUCKET_LABELS[c] for c in DOMAIN_BUCKET_LABELS if c != "unknown"]
                    domain_label_edit = st.selectbox(
                        "Program domain",
                        options=domain_label_options,
                        index=domain_label_options.index(domain_label_current)
                        if domain_label_current in domain_label_options
                        else domain_label_options.index("Other / Interdisciplinary"),
                    )

                st.markdown("**Core skills / knowledge expected** (one per line):")
                core_edit = st.text_area("core_prerequisites", value=core_text, height=120, label_visibility="collapsed")

                st.markdown("**Focus areas** (one per line):")
                focus_edit = st.text_area("focus_areas", value=focus_text, height=120, label_visibility="collapsed")

                submitted = st.form_submit_button("Apply metadata changes & recompute")

            if submitted:
                if st.session_state.llm is None:
                    mp = st.session_state.get("model_path", "")
                    if mp:
                        with st.spinner("Reloading model…"):
                            st.session_state.llm = get_llm(mp)
                    else:
                        st.error("Model is not loaded. Please click ‘Run evaluation’ first.")
                        st.stop()

                try:
                    inv_domain = {v: k for k, v in DOMAIN_BUCKET_LABELS.items()}
                    new_domain_code = inv_domain.get(domain_label_edit, "other")
                    new_core = [line.strip() for line in core_edit.splitlines() if line.strip()]
                    new_focus = [line.strip() for line in focus_edit.splitlines() if line.strip()]

                    new_meta = {
                        "program_name": prog_name_edit.strip() or "this graduate program",
                        "program_domain": new_domain_code,
                        "focus_areas": new_focus,
                        "core_prerequisites": new_core,
                        "preferred_backgrounds": preferred,
                    }

                    if st.session_state.original_df is None or st.session_state.program_text is None:
                        st.warning("Original data not found; please run evaluation again.")
                    else:
                        with st.spinner("Recomputing with updated metadata…"):
                            new_results_df, new_meta_out, new_degree_out = evaluate_profiles(
                                st.session_state.original_df,
                                st.session_state.program_text,
                                program_meta=new_meta,
                                degree_type=degree_edit,
                            )
                        st.session_state.results_df = new_results_df
                        st.session_state.program_meta = new_meta_out
                        st.session_state.degree_type = new_degree_out
                        results_df = new_results_df
                        program_meta = new_meta_out
                        degree_type = new_degree_out
                        st.success("Metadata updated and results recomputed.")
                except Exception as e:
                    st.error(f"Error updating metadata: {e}")

            col_s1, col_s2 = st.columns([2,1])
            with col_s1:
                st.markdown(f"**Program name:** {program_meta.get('program_name','this graduate program')}")
                st.markdown(f"**Degree type:** {degree_type}")
                d_label = DOMAIN_BUCKET_LABELS.get(program_meta.get("program_domain","other"),"Other / Interdisciplinary")
                st.markdown(f"**Domain:** {d_label}")
            with col_s2:
                st.markdown(f"**# students evaluated:** {len(results_df)}")

            st.markdown("**Core skills / knowledge expected:**")
            core_now = program_meta.get("core_prerequisites",[])
            if core_now:
                for c in core_now:
                    st.markdown(f"- {c}")
            else:
                st.markdown("_Not clearly specified in metadata._")

            st.markdown("**Focus areas:**")
            focus_now = program_meta.get("focus_areas",[])
            if focus_now:
                for f in focus_now:
                    st.markdown(f"- {f}")
            else:
                st.markdown("_Not clearly specified in metadata._")

    # ---- TAB 2 ----
    with tab2:
        st.subheader("High-level metrics")

        total = len(results_df)
        yes_count = (results_df["decision"] == "Yes").sum()
        no_count = (results_df["decision"] == "No").sum()
        yes_rate = (yes_count / total * 100) if total else 0
        avg_align = results_df["program_alignment"].mean()
        avg_conf = results_df["model_confidence"].mean()
        avg_align_yes = results_df.loc[results_df["decision"]=="Yes", "program_alignment"].mean()
        avg_align_no = results_df.loc[results_df["decision"]=="No", "program_alignment"].mean()

        c1, c2, c3 = st.columns(3)
        c1.metric("Total students", total)
        c2.metric("Yes decisions", f"{yes_count} ({yes_rate:.1f}%)")
        c3.metric("Average alignment", f"{avg_align:.1f}%")

        c4, c5, c6 = st.columns(3)
        c4.metric("Average model confidence", f"{avg_conf:.1f}%")
        c5.metric("Avg alignment (Yes)", f"{avg_align_yes:.1f}%" if pd.notna(avg_align_yes) else "—")
        c6.metric("Avg alignment (No)", f"{avg_align_no:.1f}%" if pd.notna(avg_align_no) else "—")

        # ===== (2a) Decision breakdown as PIE CHART =====
        st.markdown("### Decision breakdown")
        if HAS_PLOTLY:
            pie_df = pd.DataFrame({"decision": ["Yes","No"], "count": [yes_count, no_count]})
            fig_pie = px.pie(pie_df, names="decision", values="count", hole=0.3, title="Yes / No")
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.warning("Plotly is not installed; showing table instead. Install with: pip install plotly")
            st.dataframe(pd.DataFrame({"decision": ["Yes","No"], "count": [yes_count, no_count]}), use_container_width=True)

        # ===== (2b) Background types as HORIZONTAL BAR (sorted by count, highest on top) =====
        st.markdown("### By academic background type")
        seg_df = results_df.copy()
        agg_bg = (
            seg_df.groupby("background_type", dropna=False)
            .agg(
                students=("decision","count"),
                yes_rate=("decision", lambda x: (x=="Yes").mean()*100 if len(x) else 0),
                avg_alignment=("program_alignment","mean"),
                avg_confidence=("model_confidence","mean"),
            )
            .reset_index()
        )
        agg_bg = agg_bg.sort_values("students", ascending=False)

        if HAS_PLOTLY:
            fig_bar = px.bar(
                agg_bg,
                x="students",
                y="background_type",
                orientation="h",
                title="Students by background type",
                text="students",
            )
            # Ensure highest on TOP
            fig_bar.update_layout(
                yaxis=dict(
                    categoryorder="array",
                    categoryarray=list(agg_bg["background_type"][::-1])  # reverse so the first (largest) shows at top
                ),
                xaxis_title="Students",
                yaxis_title="Background type",
                showlegend=False
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.warning("Plotly is not installed; showing table instead. Install with: pip install plotly")
            st.dataframe(agg_bg, use_container_width=True)

        st.markdown("### Top / bottom / uncertain profiles")
        top5 = results_df.sort_values("program_alignment", ascending=False).head(5)
        bottom5 = results_df.sort_values("program_alignment", ascending=True).head(5)
        if "confidence_width" in results_df.columns:
            uncertain5 = results_df.sort_values("confidence_width", ascending=False).head(5)
        else:
            uncertain5 = results_df.copy().head(0)

        c_top, c_bottom, c_unc = st.columns(3)
        show_cols_tb = ["academic_background","background_type","academic_interests","professional_interests","decision","program_alignment","model_confidence"]
        with c_top:
            st.markdown("**Top 5 most aligned**")
            st.dataframe(top5[show_cols_tb], use_container_width=True)
        with c_bottom:
            st.markdown("**Bottom 5 least aligned**")
            st.dataframe(bottom5[show_cols_tb], use_container_width=True)
        with c_unc:
            st.markdown("**Most uncertain (widest certainty range)**")
            if len(uncertain5) > 0:
                st.dataframe(
                    uncertain5[show_cols_tb + ["certainty_range"]],
                    use_container_width=True,
                )
            else:
                st.write("No uncertainty data available.")

    # ---- TAB 3 ----
    with tab3:
        st.subheader("All student results")
        colf1, colf2 = st.columns(2)
        with colf1:
            decision_filter = st.selectbox("Filter by decision", ["All","Yes","No"])
        with colf2:
            min_align = st.slider("Minimum program alignment (%)", min_value=0, max_value=100, value=0, step=5)

        df_filtered = results_df.copy()
        if decision_filter != "All":
            df_filtered = df_filtered[df_filtered["decision"] == decision_filter]
        df_filtered = df_filtered[df_filtered["program_alignment"] >= min_align]

        # Ensure `background_type` is present and right after academic_background in display
        display_cols = [c for c in df_filtered.columns if c not in ("confidence_center","confidence_width")]
        if "academic_background" in display_cols and "background_type" in display_cols:
            cols_ordered = []
            for c in display_cols:
                cols_ordered.append(c)
                if c == "academic_background":
                    if display_cols.index("background_type") != len(cols_ordered):
                        cols_ordered.append("background_type")
            seen, final = set(), []
            for c in cols_ordered:
                if c not in seen:
                    final.append(c); seen.add(c)
            display_cols = final

        st.dataframe(df_filtered[display_cols], use_container_width=True)

        csv_buffer = StringIO()
        results_df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="⬇️ Download full results as CSV",
            data=csv_buffer.getvalue(),
            file_name=f"student_decisions_{datetime.now().strftime('%Y%m%d-%H%M%S')}.csv",
            mime="text/csv",
        )
