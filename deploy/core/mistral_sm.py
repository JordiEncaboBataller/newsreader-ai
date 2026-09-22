import os
import json
import re
import time
import logging
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

logger = logging.getLogger(__name__)

# Migrado de Mistral a Groq: el tier gratuito de Mistral resultó ser
# demasiado restrictivo en la práctica (403 tier_not_allowed en modelos
# grandes, 429 persistente incluso en modelos pequeños tras pocas
# peticiones). Groq ofrece un tier gratuito sin tarjeta de crédito con
# límites mucho más generosos (30 peticiones/min, 1.000/día con este
# modelo) y una API compatible con el mismo patrón de código.
MODEL_NAME = 'openai/gpt-oss-120b'

# Reintentos ante un 429 (límite de peticiones agotado). Un 429 aislado no
# significa que la petición sea inválida, solo que hay que esperar un poco
# antes de volver a intentarlo.
MAX_RETRIES = 3
RETRY_BASE_DELAY_SECONDS = 4


def _is_rate_limit_error(e):
    """Detecta si la excepción corresponde a un 429 / rate_limited de la API,
    inspeccionando el mensaje de error (la SDK no siempre expone un
    atributo status_code fiable en todas las versiones)."""
    text = str(e)
    return "429" in text or "rate_limited" in text or "Rate limit" in text


def _get_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found. Check your .env file.")
    return Groq(api_key=api_key)


def prompt_1(text, result1, result2):
    user_prompt = f"""
    You will analyze the following news article and explain its potential reliability and political bias for general readers.

    NEWS TEXT:
    {text}

    MODEL OUTPUT:

    - Fake news probability:
        - Fake: {result2['probabilities']['fake']:.2f}
        - Real: {result2['probabilities']['real']:.2f}

    - Political bias probabilities:
        - Left: {result1['probabilities']['left']:.2f}
        - Leaning-left: {result1['probabilities']['leaning-left']:.2f}
        - Center: {result1['probabilities']['center']:.2f}
        - Leaning-right: {result1['probabilities']['leaning-right']:.2f}
        - Right: {result1['probabilities']['right']:.2f}

    TASK:
    Write three separate and concise paragraphs fulfilling the following roles:

    1. Interpretation: Clearly summarize the model results using accessible language. Explain whether the article is more likely to be fake or real, and what political leaning it is most associated with. Stay neutral and factual.

    2. Justification: Elaborate on why the model might have assigned those values. Go beyond tone and vocabulary: consider who the main actors or institutions mentioned are, what political or social issue is being discussed, whether the narrative aligns with typical ideological frames, and how the argument is constructed. If relevant, comment on the emotional charge of the language, presence of sensationalism, one-sided arguments, or omission of context.

    3. Risk analysis: Speak directly to the reader. If the article appears biased or unreliable, explain why it is important to question its content before accepting it as truth. Caution the reader about how such content might shape their perception, reinforce ideological biases, or mislead them about complex issues. Encourage critical thinking.

    You MUST respond with ONLY a valid JSON object — no preamble, no markdown, no text outside the JSON.
    Use exactly these three keys:

    {{
      "interpretation": "paragraph text here",
      "justification": "paragraph text here",
      "risk_analysis": "paragraph text here"
    }}
    """
    return user_prompt


def prompt_2(text):
    summary_prompt = f"""
    You will read the following news article and write a clear and concise summary of its main points.

    TEXT:
    {text}

    TASK:
    Summarize the article in one paragraph (6-8 lines), covering the main topic, key facts, people or institutions involved, and any relevant outcomes or context. Avoid personal opinions or speculation. Write in neutral and accessible language, suitable for a general audience.

    Make sure the summary is self-contained and understandable without needing to read the full article.

    IMPORTANT: Respond with ONLY the summary paragraph itself. Do NOT include a title, heading, or label such as "Summary:", "Summary", or similar — the reader already knows it is a summary. Do not use markdown formatting (no bold, no headers, no bullet points). Start directly with the first sentence of the summary.
    """
    return summary_prompt


def prompt_3(text):
    translate_prompt = f"""
    Translate the following news article to English.

    TEXT:
    {text}

    TASK:
    Provide a complete, faithful translation of the article into English. Preserve the original meaning, tone, structure and paragraph breaks. Do not summarize, shorten, comment on, or add anything to the text — translate it in full.

    Respond with ONLY the translated text — no preamble, no notes, no markdown.
    """
    return translate_prompt


def call_mistral_json(prompt, role):
    """Calls the LLM (Groq) and returns a parsed Python dict (JSON mode).
    Reintenta automáticamente si la API responde con un 429 (rate limit del
    tier gratuito), esperando un poco más en cada intento."""
    client = _get_client()

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": role},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                # temperature=0 minimiza (aunque no garantiza al 100%) la
                # variabilidad de la respuesta entre llamadas idénticas.
                temperature=0
            )
            raw = response.choices[0].message.content
            # Strip markdown fences if the model wraps the JSON anyway
            cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            return json.loads(cleaned)
        except Exception as e:
            last_error = e
            if _is_rate_limit_error(e) and attempt < MAX_RETRIES:
                wait = RETRY_BASE_DELAY_SECONDS * attempt
                logger.warning(f"Rate limit alcanzado (intento {attempt}/{MAX_RETRIES}), reintentando en {wait}s...")
                time.sleep(wait)
                continue
            raise last_error


def call_mistral_text(prompt, role):
    """Calls the LLM (Groq) and returns plain text.
    Reintenta automáticamente si la API responde con un 429 (rate limit del
    tier gratuito), esperando un poco más en cada intento."""
    client = _get_client()

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": role},
                    {"role": "user", "content": prompt}
                ],
                # temperature=0 minimiza (aunque no garantiza al 100%) la
                # variabilidad de la respuesta entre llamadas idénticas.
                temperature=0
            )
            return response.choices[0].message.content
        except Exception as e:
            last_error = e
            if _is_rate_limit_error(e) and attempt < MAX_RETRIES:
                wait = RETRY_BASE_DELAY_SECONDS * attempt
                logger.warning(f"Rate limit alcanzado (intento {attempt}/{MAX_RETRIES}), reintentando en {wait}s...")
                time.sleep(wait)
                continue
            raise last_error


def EXPLAIN(text, result1, result2):
    """Returns a dict with keys: interpretation, justification, risk_analysis.
    Returns None if the call or parsing fails."""
    prompt = prompt_1(text, result1, result2)
    role = "You are a helpful assistant that analyzes news articles to explain their reliability and bias to the general public. You always respond with valid JSON."
    try:
        return call_mistral_json(prompt, role)
    except Exception as e:
        logger.error(f"EXPLAIN error: {e}")
        return None


def _strip_leading_heading(text):
    """Elimina un posible encabezado inicial tipo '**Summary:**', 'Summary:'
    o 'Resumen:' que el modelo a veces añade pese a que se le pide que no
    lo haga. Se aplica como red de seguridad adicional, ya que el prompt
    por sí solo no garantiza al 100% que el modelo lo omita.
    """
    if not text:
        return text

    cleaned = text.strip()
    # Coincide con variantes como:
    #   **Summary:**
    #   Summary:
    #   ### Summary
    #   Resumen:
    # al principio del texto, seguidas de saltos de línea/espacios.
    cleaned = re.sub(
        r'^\s*(#{1,3}\s*)?\**\s*(summary|resumen)\s*:?\**\s*\n*',
        '',
        cleaned,
        flags=re.IGNORECASE
    )
    return cleaned.strip()


def SUMMARY(text):
    """Returns a plain text summary string, or None on failure."""
    prompt = prompt_2(text)
    role = "You are a helpful assistant that summarizes news articles clearly and concisely for a general audience."
    try:
        raw_summary = call_mistral_text(prompt, role)
        return _strip_leading_heading(raw_summary)
    except Exception as e:
        logger.error(f"SUMMARY error: {e}")
        return None


def TRANSLATE(text):
    """Translates an article to English. Returns the translated text,
    or None on failure (caller should fall back to the original text)."""
    prompt = prompt_3(text)
    role = "You are a professional translator that produces accurate, complete English translations of news articles."
    try:
        return call_mistral_text(prompt, role)
    except Exception as e:
        logger.error(f"TRANSLATE error: {e}")
        return None