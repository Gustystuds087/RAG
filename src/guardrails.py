"""Guardrails for the medical RAG app.

Layered design:
  1. INPUT  (rule-based)  — fast keyword checks for dangerous/self-harm intent.
  2. INPUT  (LLM classifier) — catches off-topic / harmful that rules miss.
  3. RETRIEVAL (low-confidence) — if nothing relevant was retrieved, refuse.
  4. OUTPUT (disclaimer)  — guarantee every answer carries a safety notice.

Each input guardrail returns a GuardResult. If `.blocked` is True, the engine
short-circuits and returns `.message` instead of calling the LLM normally.
"""
from dataclasses import dataclass

from . import config


DISCLAIMER = (
    "\n\n---\n⚠️ *This is not medical advice. It is general information from a "
    "dataset and may be incomplete or inaccurate. Always consult a qualified "
    "doctor or pharmacist before taking, stopping, or combining any medication.*"
)

# Shown for self-harm / crisis input — we do NOT give drug info in this case.
CRISIS_MESSAGE = (
    "It sounds like you may be going through something serious. I can't help "
    "with this, but please reach out for support right now:\n\n"
    "- **India:** Tele-MANAS 14416 / KIRAN 1800-599-0019\n"
    "- **US:** call or text 988 (Suicide & Crisis Lifeline)\n"
    "- **Or contact your local emergency number immediately.**\n\n"
    "You deserve support from a real person who can help."
)

# Universal emergency core — always the same: call for help now.
_EMERGENCY_HEADER = (
    "🚨 **This may be a medical emergency. Get help right now — do not wait.**\n\n"
    "**India — call immediately:**\n"
    "- **112** — national emergency number\n"
    "- **108** — ambulance (most states)\n"
    "- **102** — medical ambulance\n\n"
)
_EMERGENCY_FOOTER = (
    "\n\nI'm an information tool and cannot help in an emergency. Please contact "
    "emergency services or get to the nearest hospital immediately."
)

# First-aid tips matched to the TYPE of emergency. Each is "while you wait for
# the ambulance" guidance — calling for help always comes first.
_FIRST_AID = {
    "heart_attack": (
        "**While waiting:** sit down and rest, loosen tight clothing. If you are "
        "not allergic and a doctor hasn't told you otherwise, chewing one regular "
        "aspirin can help — but **call the ambulance first.**"
    ),
    "bleeding": (
        "**While waiting:** apply firm, direct pressure on the wound with a clean "
        "cloth and keep pressing. Raise the injured area above heart level if you "
        "can. Do **not** remove a soaked cloth — add another on top."
    ),
    "breathing": (
        "**While waiting:** sit upright, loosen tight clothing, and try to stay "
        "calm. If you have a prescribed inhaler, use it."
    ),
    "choking": (
        "**While waiting:** if someone is choking and cannot breathe, give firm "
        "back blows between the shoulder blades, then abdominal thrusts (Heimlich)."
    ),
    "stroke": (
        "**While waiting:** note the time symptoms started, keep the person sitting "
        "or lying with head slightly raised, and do not give food, drink, or medicine."
    ),
    "overdose": (
        "**While waiting:** keep the person awake and on their side. Have the "
        "medicine packet ready to show responders. Do not induce vomiting unless told to."
    ),
    "allergic": (
        "**While waiting:** if an adrenaline auto-injector (EpiPen) is prescribed, "
        "use it now. Help the person sit up if breathing is hard, or lie flat if faint."
    ),
    "unconscious": (
        "**While waiting:** if breathing, place them on their side (recovery "
        "position). If not breathing, start CPR if you are trained."
    ),
    "generic": (
        "**While waiting:** keep the person calm and still, and do not give any "
        "medicine, food, or drink unless a professional tells you to."
    ),
}


def build_emergency_message(kind: str = "generic") -> str:
    tip = _FIRST_AID.get(kind, _FIRST_AID["generic"])
    return _EMERGENCY_HEADER + tip + _EMERGENCY_FOOTER


# Default generic emergency message (used when the type isn't identified).
EMERGENCY_MESSAGE = build_emergency_message("generic")


def classify_emergency_kind(query: str) -> str:
    """Best-effort mapping of the query to a first-aid category."""
    q = query.lower()
    if any(t in q for t in ["heart attack", "heartattack", "chest pain"]):
        return "heart_attack"
    if any(t in q for t in ["bleed", "blood"]):
        return "bleeding"
    if any(t in q for t in ["breathe", "breathing", "can't breath", "cant breath"]):
        return "breathing"
    if "chok" in q:
        return "choking"
    if "stroke" in q:
        return "stroke"
    if any(t in q for t in ["overdose", "overdosed", "too many pills"]):
        return "overdose"
    if any(t in q for t in ["allerg", "anaphyla"]):
        return "allergic"
    if any(t in q for t in ["unconscious", "collapsed", "won't wake", "not responding", "unresponsive"]):
        return "unconscious"
    return "generic"


OFFTOPIC_MESSAGE = (
    "I can only answer questions about medicines, their uses, side effects, "
    "and substitutes. Please ask a medicine-related question."
)

# For misuse of medicine (overdose intent, getting high, etc.) — refuse and
# point to a professional.
DANGEROUS_MESSAGE = (
    "I can't help with that request. For any question about dosing, overdose, "
    "or combining medicines, please consult a doctor or pharmacist directly."
)

# For intent to harm ANOTHER person — refuse firmly, do NOT suggest a
# pharmacist, and point to emergency services / support.
HARM_OTHERS_MESSAGE = (
    "I can't help with this, and I won't provide any information that could be "
    "used to harm someone.\n\n"
    "If you are having thoughts of hurting another person, please reach out for "
    "help right now:\n"
    "- **India:** call **112** (emergency) or Tele-MANAS **14416** for mental-health support\n"
    "- If someone is in immediate danger, contact your local emergency services.\n\n"
    "Talking to a mental-health professional can help."
)

LOW_CONFIDENCE_MESSAGE = (
    "I don't have reliable information about that in my medicine database, so "
    "I'd rather not guess. Please consult a doctor or pharmacist."
)

ABUSE_MESSAGE = (
    "I'm here to help with medicine questions and want to keep things respectful. "
    "Please rephrase your question without abusive language and I'll do my best to help."
)

HATE_MESSAGE = (
    "I won't respond to hate speech or slurs. I'm happy to help with genuine "
    "medicine questions asked respectfully."
)

INJECTION_MESSAGE = (
    "I can only provide medicine information from my database and can't change my "
    "instructions or role. Please ask a normal question about a medicine, its uses, "
    "side effects, or substitutes."
)


@dataclass
class GuardResult:
    blocked: bool
    message: str = ""
    reason: str = ""


# ---------- 1. Rule-based input checks (fast, free) ----------

# Active medical-emergency signals. If the user says they ARE having one of
# these, route to emergency help — never to medicine info. Checked FIRST.
# We look for first-person/active phrasing to avoid blocking informational
# questions like "what are the symptoms of a heart attack".
_EMERGENCY_TERMS = [
    "having a heart attack", "i am having a heart attack", "im having a heart attack",
    "having heart attack", "chest pain and", "having a stroke", "having stroke",
    "can't breathe", "cant breathe", "cannot breathe", "stopped breathing",
    "not breathing", "severe bleeding", "bleeding heavily", "unconscious",
    "collapsed", "choking", "severe allergic reaction", "anaphylaxis",
    "overdosed", "took too many pills", "i overdosed",
]

# Phrases that strongly signal self-harm / crisis, including method-specific
# phrasings. Kept conservative but broader than just the word "suicide".
_CRISIS_TERMS = [
    # direct
    "kill myself", "killing myself", "suicide", "suicidal", "end my life",
    "end it all", "want to die", "wanna die", "i want to die", "don't want to live",
    "dont want to live", "no reason to live", "better off dead", "want to disappear",
    "how to die", "ways to die", "tired of living",
    # self-harm
    "harm myself", "hurt myself", "self harm", "self-harm", "cut myself",
    "overdose to die", "how much to overdose", "lethal dose",
    # method-specific self-harm
    "jump off", "jump from", "jump in front of", "hang myself", "hanging myself",
    "slit my", "drown myself", "shoot myself", "throw myself",
]

# Phrases signalling intent to harm ANOTHER person. Checked before the generic
# dangerous terms so they get the right (harm-others) response.
_HARM_OTHERS_TERMS = [
    "kill someone", "kill him", "kill her", "kill them", "murder",
    "poison someone", "poison him", "poison her", "poison them", "poison my",
    "hurt someone", "harm someone", "kill my",
]

# Phrases signalling intent to misuse medicines (self-directed misuse / abuse).
_DANGEROUS_TERMS = [
    "overdose", "get high", "abuse", "how much to take to",
    "maximum dose to", "how many pills to",
]

# Hate speech / slurs (racial, etc.). These ALWAYS block with a firm message,
# regardless of any medical content in the message. Word-boundary matched.
# (Kept intentionally short; the LLM classifier backstops anything missed.)
_SLURS = {
    "nigger", "nigga", "faggot", "fag", "chink", "spic", "kike", "tranny",
    "coon", "wetback", "gook", "retard",
}

# Ordinary profanity (not hate speech). We use word-boundary matching so we
# don't flag substrings inside legitimate words (e.g. "assess", "Scunthorpe").
_PROFANITY = {
    "fuck", "fucking", "fucker", "shit", "bullshit", "bitch", "bastard",
    "asshole", "dick", "cunt", "motherfucker", "dumbass", "jackass",
    "prick", "wanker", "slut", "whore", "moron", "idiot",
}

# If profanity appears alongside any of these, it's likely a real (if salty)
# medical question -> we ANSWER it rather than block.
_MEDICAL_HINTS = [
    "medicine", "tablet", "pill", "syrup", "capsule", "dose", "drug",
    "fever", "pain", "headache", "cough", "cold", "infection", "allergy",
    "side effect", "treat", "treatment", "what can i take", "what should i take",
    "cure", "relief", "symptom", "doctor", "prescription", "antibiotic",
]

# Words that turn profanity into ABUSE aimed at the assistant -> block.
_ABUSE_TARGETS = ["you", "u r", "ur ", "your", "yourself", "this bot", "this app", "stupid bot"]


# Common prompt-injection patterns. These are attempts to override the system
# prompt, change the assistant's role, or extract its instructions.
_INJECTION_PATTERNS = [
    "ignore the above", "ignore previous", "ignore all previous",
    "ignore your instructions", "ignore your rules", "disregard the",
    "disregard previous", "forget your instructions", "forget you are",
    "you are now", "act as", "pretend you are", "pretend to be",
    "reveal your prompt", "reveal your system", "show your prompt",
    "what is your system prompt", "repeat your instructions",
    "system:", "developer mode", "jailbreak", "without a disclaimer",
    "no disclaimer", "override your", "new instructions:",
]


def injection_check(query: str) -> GuardResult:
    """Detect prompt-injection attempts. Returns blocked=True if found.

    Note: this is a heuristic first line of defense. The hardened system prompt
    and input delimiting in the engine are the more important protections."""
    q = query.lower()
    if any(p in q for p in _INJECTION_PATTERNS):
        return GuardResult(blocked=True, message=INJECTION_MESSAGE, reason="injection")
    return GuardResult(blocked=False)


def _words(q: str) -> set:
    import re
    return set(re.findall(r"[a-z]+", q))


def _contains_profanity(q: str) -> bool:
    return bool(_words(q) & _PROFANITY)


def hate_check(query: str) -> GuardResult:
    """Slurs / hate speech ALWAYS block, regardless of any medical content."""
    if _words(query.lower()) & _SLURS:
        return GuardResult(blocked=True, message=HATE_MESSAGE, reason="hate")
    return GuardResult(blocked=False)


def profanity_check(query: str) -> GuardResult:
    """Block abuse/harassment, but allow genuine medical questions that merely
    contain a swear word."""
    q = query.lower()
    if not _contains_profanity(q):
        return GuardResult(blocked=False)

    has_medical = any(h in q for h in _MEDICAL_HINTS)
    targets_bot = any(t in q for t in _ABUSE_TARGETS)

    # Abuse aimed at the bot, OR profanity with no real medical content -> block.
    if targets_bot or not has_medical:
        return GuardResult(blocked=True, message=ABUSE_MESSAGE, reason="abuse")

    # Salty but genuine medical question -> let it through to be answered.
    return GuardResult(blocked=False)


def rule_based_input_check(query: str) -> GuardResult:
    q = query.lower()

    # Active medical emergency is the HIGHEST priority — get help immediately,
    # never list medicines. The first-aid tip is matched to the emergency type.
    if any(term in q for term in _EMERGENCY_TERMS):
        return GuardResult(
            blocked=True,
            message=build_emergency_message(classify_emergency_kind(query)),
            reason="emergency",
        )

    # Self-harm crisis — route to help, never to drug info.
    if any(term in q for term in _CRISIS_TERMS):
        return GuardResult(blocked=True, message=CRISIS_MESSAGE, reason="crisis")

    # Intent to harm another person — distinct from medicine misuse.
    if any(term in q for term in _HARM_OTHERS_TERMS):
        return GuardResult(blocked=True, message=HARM_OTHERS_MESSAGE, reason="harm-others")

    # Hate speech / slurs — always block with the hate message (NOT the dosing
    # one). Checked before 'dangerous' so a slur never gets the wrong response.
    hate = hate_check(query)
    if hate.blocked:
        return hate

    if any(term in q for term in _DANGEROUS_TERMS):
        return GuardResult(blocked=True, message=DANGEROUS_MESSAGE, reason="dangerous")

    # Prompt-injection attempt — block before it reaches the LLM.
    inj = injection_check(query)
    if inj.blocked:
        return inj

    # Profanity LAST among rules — after the safety-critical checks, so an
    # emergency phrased with a swear still routes to emergency help first.
    prof = profanity_check(query)
    if prof.blocked:
        return prof

    return GuardResult(blocked=False)


# ---------- 2. LLM-based input classifier (catches what rules miss) ----------

def llm_input_check(query: str, llm) -> GuardResult:
    """Ask the LLM to classify the query. Fails OPEN (allows) if the LLM is
    unavailable or returns something unexpected — we don't want a classifier
    outage to take the whole app down."""
    if llm is None:
        return GuardResult(blocked=False)

    system = (
        "You are a safety classifier for a medicine-information assistant. "
        "Classify the user's message into exactly one label:\n"
        "  EMERGENCY - a SEVERE, life-threatening emergency happening RIGHT NOW: "
        "heart attack, stroke, cannot breathe, HEAVY/uncontrolled bleeding, "
        "unconscious/unresponsive, choking, drug overdose, severe allergic reaction, "
        "someone collapsed. Only use this for clearly severe, urgent situations.\n"
        "  SAFE      - any normal medicine/health question, INCLUDING minor issues "
        "(a small cut, little bleeding, mild headache, common cold, asking what to take "
        "for a symptom). Minor or mild problems are SAFE, not EMERGENCY.\n"
        "  OFFTOPIC  - not about medicines/health at all\n"
        "  SELFHARM  - the person expresses wanting to harm themselves, end their "
        "life, or commit suicide, by ANY method (pills, jumping, hanging, etc.)\n"
        "  HARMOTHERS - seeks to hurt, kill, poison, or harm ANOTHER person\n"
        "  HATE      - contains a slur or hate speech toward a group or person\n"
        "  DANGEROUS - seeks to misuse medicine, overdose intentionally, get high, or illegal use\n"
        "\n"
        "Examples:\n"
        "  'I am having a heart attack' -> EMERGENCY\n"
        "  'I have a little bleeding from a cut' -> SAFE\n"
        "  'what can I take for a mild headache' -> SAFE\n"
        "  'my father collapsed and won't wake up' -> EMERGENCY\n"
        "  'what is the capital of France' -> OFFTOPIC\n"
        "  'I want to jump off this building' -> SELFHARM\n"
        "  'I want to end my life' -> SELFHARM\n"
        "  'how can I poison my neighbour' -> HARMOTHERS\n"
        "  'which pill gets me high' -> DANGEROUS\n"
        "\n"
        "Reply with ONLY the single label word."
    )
    try:
        resp = llm.chat.completions.create(
            model=config.GROQ_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": query},
            ],
            temperature=0,
            max_tokens=4,
        )
        label = resp.choices[0].message.content.strip().upper()
    except Exception:  # noqa: BLE001 — fail open
        return GuardResult(blocked=False)

    if "EMERGENCY" in label:
        return GuardResult(
            blocked=True,
            message=build_emergency_message(classify_emergency_kind(query)),
            reason="emergency-llm",
        )
    if "SELFHARM" in label:
        return GuardResult(blocked=True, message=CRISIS_MESSAGE, reason="crisis-llm")
    if "HARMOTHERS" in label:
        return GuardResult(blocked=True, message=HARM_OTHERS_MESSAGE, reason="harm-others-llm")
    if "HATE" in label:
        return GuardResult(blocked=True, message=HATE_MESSAGE, reason="hate-llm")
    if "DANGEROUS" in label:
        return GuardResult(blocked=True, message=DANGEROUS_MESSAGE, reason="dangerous-llm")
    if "OFFTOPIC" in label:
        return GuardResult(blocked=True, message=OFFTOPIC_MESSAGE, reason="offtopic-llm")
    return GuardResult(blocked=False)


def check_input(query: str, llm) -> GuardResult:
    """Run rule-based first (free), then the LLM classifier."""
    result = rule_based_input_check(query)
    if result.blocked:
        return result
    return llm_input_check(query, llm)


# ---------- 3. Low-confidence retrieval guard ----------

# Cosine-similarity scores below this mean "nothing relevant was found".
# Tuned from observed scores: real matches ~0.3-0.6, noise ~0.2 and below.
MIN_TOP_SCORE = float(config.__dict__.get("MIN_TOP_SCORE", 0.0)) or 0.25


def check_retrieval(hits: list[dict]) -> GuardResult:
    if not hits:
        return GuardResult(blocked=True, message=LOW_CONFIDENCE_MESSAGE, reason="no-hits")
    top = max(h.get("score", 0.0) for h in hits)
    if top < MIN_TOP_SCORE:
        return GuardResult(blocked=True, message=LOW_CONFIDENCE_MESSAGE, reason="low-confidence")
    return GuardResult(blocked=False)


# ---------- 4. Output enforcement ----------

def enforce_disclaimer(answer: str) -> str:
    """Guarantee the disclaimer is present exactly once."""
    if "not medical advice" in answer.lower():
        return answer  # the LLM already included one
    return answer + DISCLAIMER
