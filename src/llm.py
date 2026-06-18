"""LLM wrapper with two providers, used for different jobs:

  - Gemini (2.0-flash): Cypher generation + final answers (needs reasoning/quality)
  - Groq (Llama):       guardrail classification (fast, simple, separate quota)

This spreads load across two free tiers so neither hits its rate limit, and
each falls back to the other if its provider is unavailable.

Public API:
  complete(system, user, ...)        -> default provider (Gemini, else Groq)
  complete_fast(system, user, ...)   -> prefers Groq (for guardrails), else Gemini
  available()                        -> True if any provider is configured
"""
from . import config

_gemini = None
_groq = None
_init_done = False


def _init():
    global _gemini, _groq, _init_done
    if _init_done:
        return
    if config.GEMINI_API_KEY:
        try:
            import google.generativeai as genai
            genai.configure(api_key=config.GEMINI_API_KEY)
            _gemini = genai.GenerativeModel(config.GEMINI_MODEL)
        except Exception as e:  # noqa: BLE001
            print(f"[llm] Gemini init failed: {e}")
            _gemini = None
    if config.GROQ_API_KEY:
        try:
            from groq import Groq
            _groq = Groq(api_key=config.GROQ_API_KEY)
        except Exception as e:  # noqa: BLE001
            print(f"[llm] Groq init failed: {e}")
            _groq = None
    _init_done = True


def available() -> bool:
    return bool(config.GEMINI_API_KEY or config.GROQ_API_KEY)


def _gemini_call(system: str, user: str, temperature: float, max_tokens: int) -> str:
    resp = _gemini.generate_content(
        f"{system}\n\n{user}",
        generation_config={"temperature": temperature, "max_output_tokens": max_tokens},
    )
    return (resp.text or "").strip()


def _groq_call(system: str, user: str, temperature: float, max_tokens: int) -> str:
    resp = _groq.chat.completions.create(
        model=config.GROQ_MODEL,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()


def complete(system: str, user: str, temperature: float = 0.2, max_tokens: int = 1024) -> str:
    """Primary completion (Cypher gen + answers): Gemini first, Groq fallback."""
    _init()
    if _gemini is not None:
        try:
            return _gemini_call(system, user, temperature, max_tokens)
        except Exception as e:  # noqa: BLE001 — fall through to Groq
            print(f"[llm] Gemini failed, falling back to Groq: {e}")
    if _groq is not None:
        return _groq_call(system, user, temperature, max_tokens)
    raise RuntimeError("No LLM available.")


def complete_fast(system: str, user: str, temperature: float = 0.0, max_tokens: int = 8) -> str:
    """Lightweight completion (guardrail classification): Groq first, Gemini fallback.
    Keeps guardrail calls off Gemini's quota."""
    _init()
    if _groq is not None:
        try:
            return _groq_call(system, user, temperature, max_tokens)
        except Exception as e:  # noqa: BLE001 — fall through to Gemini
            print(f"[llm] Groq failed, falling back to Gemini: {e}")
    if _gemini is not None:
        return _gemini_call(system, user, temperature, max_tokens)
    raise RuntimeError("No LLM available.")
