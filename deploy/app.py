import streamlit as st
import streamlit.components.v1 as components
import requests
from core.auto_scraper import AutoScraper
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import joblib
from core.predictor_bias import BIAS
from core.predictor_fake import FAKE
from core.groq import EXPLAIN, SUMMARY, TRANSLATE
from core.figuras import FigBarras, FigTarta
import time
import html
import base64
import logging
from urllib.parse import urlparse
from langdetect import detect, DetectorFactory, LangDetectException
from core.chunking_utils import MAX_CHUNKS
import concurrent.futures  # Needed for LLM call timeout control and BIAS/FAKE parallelization

# Centralized logging configuration: every module (auto_scraper, groq,
# etc.) uses logging.getLogger(__name__) and automatically inherits this
# configuration. Previously print() was used, which doesn't distinguish
# severity levels or include timestamps.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# Deterministic results on every run
DetectorFactory.seed = 0

# Max character limit for text sent to the LLM interpretation module
MAX_TEXT_CHARS_LLM = 15000

st.set_page_config(page_title="NewsReaderAI", layout="wide")

# ─────────────────────────────────────────────────────────────────────────
# DESIGN SYSTEM
# Centralized design system: typography (Inter), premium dark palette, and
# radius/shadow tokens reused across all result cards in the app.
# Doesn't change any layout or component, only its appearance.
# ─────────────────────────────────────────────────────────────────────────
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">

<style>
:root{
    --bg: #0a0a0c;
    --surface: #16161a;
    --surface-2: #1c1c21;
    --border: #27272d;
    --border-soft: #1e1e23;
    --text: #f4f4f5;
    --text-dim: #c7c7cd;
    --text-mute: #8a8a93;

    --accent: #5b8cff;
    --accent-strong: #3d6ee8;
    --accent-soft: rgba(91, 140, 255, 0.12);

    --emerald: #34d399;
    --emerald-soft: rgba(52, 211, 153, 0.12);
    --amber: #d99a3d;
    --amber-soft: rgba(217, 154, 61, 0.12);
    --coral: #e8735c;
    --coral-soft: rgba(232, 115, 92, 0.12);
    --sky: #38bdf8;
    --sky-soft: rgba(56, 189, 248, 0.12);
    --slate: #9a9aa3;
    --slate-soft: rgba(154, 154, 163, 0.12);

    --radius-lg: 20px;
    --radius-md: 14px;
    --radius-sm: 10px;
    --shadow-card: 0 10px 30px rgba(0,0,0,0.35), 0 1px 0 rgba(255,255,255,0.03) inset;
}

html, body, [class*="css"]  {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}

.stApp {
    background: var(--bg);
}

/* Main button (ANALYZE) */
div.stButton > button:first-child {
    background: linear-gradient(135deg, var(--accent), var(--accent-strong));
    color: #ffffff;
    border: none;
    border-radius: var(--radius-sm);
    padding: 0.7rem 1.4rem;
    font-weight: 600;
    letter-spacing: 0.01em;
    box-shadow: 0 4px 16px rgba(91, 140, 255, 0.35);
    transition: transform 0.2s ease, box-shadow 0.2s ease, filter 0.2s ease;
}
div.stButton > button:first-child:hover {
    transform: translateY(-1px);
    box-shadow: 0 10px 24px rgba(91, 140, 255, 0.45);
    filter: brightness(1.06);
}
div.stButton > button:first-child:active {
    transform: translateY(0px);
}

/* URL input */
div[data-testid="stTextInput"] input {
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    color: var(--text);
    padding: 0.85rem 1rem 0.85rem 2.75rem;
    font-size: 0.95rem;
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='18' height='18' viewBox='0 0 24 24' fill='none' stroke='%238a8a93' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71'%3E%3C/path%3E%3Cpath d='M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71'%3E%3C/path%3E%3C/svg%3E");
    background-repeat: no-repeat;
    background-position: 14px center;
    background-size: 18px 18px;
}
div[data-testid="stTextInput"] input:focus {
    border-color: var(--accent);
    box-shadow: 0 0 0 3px var(--accent-soft);
    outline: none;
}
div[data-testid="stTextInput"] input::placeholder {
    color: var(--text-mute);
}

/* Tabs */
div[data-baseweb="tab-list"] {
    gap: 6px;
    border-bottom: 1px solid var(--border-soft);
}
button[data-baseweb="tab"] {
    color: var(--text-mute);
    font-weight: 500;
    font-size: 0.92rem;
    padding: 0.7rem 1.1rem;
    border-radius: 10px 10px 0 0;
    transition: background 0.2s ease, color 0.2s ease;
}
button[data-baseweb="tab"]:hover {
    color: var(--text);
    background: var(--surface);
}
button[aria-selected="true"][data-baseweb="tab"] {
    color: var(--text) !important;
    background: var(--surface) !important;
    border-bottom: 2px solid var(--accent) !important;
}

/* Expanders */
details {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
}
summary {
    color: var(--text-dim) !important;
    font-weight: 500 !important;
}

/* Alerts (warning / error) */
div[data-testid="stAlert"] {
    border-radius: var(--radius-sm);
    border: 1px solid var(--border);
}

/* Spinner text */
div[data-testid="stSpinner"] > div {
    color: var(--text-mute);
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────
# UI helpers: reusable section labels and prediction cards.
# Keep the same position/order as before, only the HTML markup changes.
# ─────────────────────────────────────────────────────────────────────────

def section_label(texto):
    """Eyebrow label with an accent bar, replaces the large <h3> headers."""
    st.markdown(
        f"""
        <div style="display:flex;align-items:center;gap:10px;margin:0 0 18px 0;">
            <div style="width:3px;height:16px;background:var(--accent);border-radius:2px;"></div>
            <span style="font-size:0.78rem;font-weight:600;letter-spacing:0.12em;
                         text-transform:uppercase;color:var(--text-mute);">{texto}</span>
        </div>
        """,
        unsafe_allow_html=True
    )


# Subtle palettes (avoiding pure red/green) for the predictions.
BIAS_COLORS = {
    "left":           ("#5b8cff", "var(--accent-soft)"),
    "leaning-left":   ("#38bdf8", "var(--sky-soft)"),
    "center":         ("#9a9aa3", "var(--slate-soft)"),
    "leaning-right":  ("#d99a3d", "var(--amber-soft)"),
    "right":          ("#e8735c", "var(--coral-soft)"),
}
FAKE_COLORS = {
    "real": ("#34d399", "var(--emerald-soft)"),
    "fake": ("#e8735c", "var(--coral-soft)"),
}


def render_prediction_card(pred_label, color_map, duration, chunk_note):
    """Renders a result card showing the predicted label, execution time,
    and a note on how many chunks the article was analyzed in."""
    color, bg = color_map.get(pred_label, ("#5b8cff", "var(--accent-soft)"))
    label_display = pred_label.replace("-", " ").title()
    st.markdown(
        f"""
        <div style="background:var(--surface);border:1px solid var(--border);
                    border-radius:var(--radius-lg);padding:1.75rem 2rem;
                    box-shadow:var(--shadow-card);margin-bottom:1.75rem;">
            <div style="font-size:0.75rem;font-weight:600;letter-spacing:0.1em;
                        text-transform:uppercase;color:var(--text-mute);margin-bottom:0.9rem;">
                Prediction
            </div>
            <div style="display:inline-flex;align-items:center;gap:8px;background:{bg};
                        color:{color};border:1px solid {color}40;padding:0.45rem 1rem;
                        border-radius:999px;font-weight:600;font-size:1.05rem;">
                <span style="width:8px;height:8px;border-radius:50%;background:{color};
                             display:inline-block;"></span>
                {label_display}
            </div>
            <div style="margin-top:1.1rem;color:var(--text-mute);font-size:0.85rem;
                        display:flex;gap:18px;flex-wrap:wrap;">
                <span>⏱ {duration:.2f}s</span>
                <span>{chunk_note}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# Accent colors for each block of the LLM interpretation
LLM_SECTION_COLORS = {
    "interpretation": "#5b8cff",
    "justification": "#d99a3d",
    "risk_analysis": "#e8735c",
}


logo_path = "./assets/logo_oscuro.png"
with open(logo_path, "rb") as f:
    img_bytes = f.read()
    encoded_logo = base64.b64encode(img_bytes).decode()

st.markdown(f"""
<div style='
    position: relative;
    display: flex;
    align-items: center;
    gap: 26px;
    width: 100%;
    padding: 38px 44px;
    margin-bottom: 28px;
    border-radius: 24px;
    border: 1px solid var(--border-soft);
    background:
        radial-gradient(900px circle at 0% 0%, rgba(91,140,255,0.10), transparent 55%),
        var(--surface);
    overflow: hidden;
'>
    <div style='
    flex: 0 0 auto;
    width: 200px;
    height: 200px;
    display: flex;
    align-items: center;
    justify-content: center;
'>
    <img src='data:image/png;base64,{encoded_logo}' style='width: 100%; height: 100%; object-fit: contain;' />
</div>
    <div style='flex: 1; position: relative; z-index: 1;'>
        <div style='
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: var(--accent-soft);
            color: var(--accent);
            border: 1px solid rgba(91,140,255,0.28);
            padding: 4px 12px;
            border-radius: 999px;
            font-size: 0.68rem;
            font-weight: 700;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            margin-bottom: 12px;
        '>✦ AI-Powered Analysis</div>
        <h1 style='font-size: 3.3rem; font-weight: 800; margin: 0; color: var(--text);
                   letter-spacing: -0.02em; line-height: 1.1;'>NewsReader <span style="color:var(--accent);">AI</span></h1>
        <p style='margin: 12px 0 0 0; color: var(--text-mute); font-size: 1.02rem; max-width: 560px;'>
            Bias &amp; fake news detection, powered by transformer models and LLM interpretation.
        </p>
    </div>
</div>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    """Loads and caches the bias and fake-news classifier models, their
    tokenizers, and the bias label encoder, so they're only loaded once
    per session rather than on every prediction."""
    dir1 = "./ModelsBias"
    model_bias = AutoModelForSequenceClassification.from_pretrained(dir1)
    tokenizer_bias = AutoTokenizer.from_pretrained(dir1)
    le = joblib.load(f"{dir1}/label_encoder.pkl")

    dir2 = './ModelsFake'
    tokenizer_fake = AutoTokenizer.from_pretrained(dir2)
    model_fake = AutoModelForSequenceClassification.from_pretrained(
                dir2,
                num_labels=2,
                use_safetensors=True
            )

    # IMPORTANT: by default a PyTorch model stays in train() mode after
    # loading, which keeps dropout active during inference (torch.no_grad()
    # only disables gradient computation, NOT dropout). Without this, the
    # same article could yield slightly different probabilities on every
    # analysis. .eval() disables it, making predictions deterministic.
    model_bias.eval()
    model_fake.eval()

    return model_bias, tokenizer_bias, le, model_fake, tokenizer_fake


col1, col2 = st.columns([7, 1])
with col1:
    url = st.text_input(
        label="Introduce una URL para analizar:",
        placeholder="Paste a news article URL to analyze...",
        label_visibility="collapsed",
        key="url_input"
    )
with col2:
    analizar = st.button("ANALYZE", use_container_width=True)

st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)


if analizar and url:
    # Format validation before firing off any network request. Without
    # this, a malformed URL (missing scheme, spaces, etc.) produced
    # confusing errors later in requests.head() or the scraper, without
    # the user knowing the problem was simply the URL.
    url = url.strip()
    parsed_url = urlparse(url)
    if parsed_url.scheme not in ("http", "https") or not parsed_url.netloc:
        st.error(
            "❌ La URL introducida no es válida. Asegúrate de que empiece "
            "por **http://** o **https://** y tenga un dominio "
            "(ej: `https://ejemplo.com/noticia`)."
        )
        st.stop()

    texto_completo = None
    resumen = None

    can_embed = True
    try:
        response = requests.head(url, timeout=8, allow_redirects=True)
        xfo = response.headers.get('X-Frame-Options', '').lower()
        if 'deny' in xfo or 'sameorigin' in xfo:
            can_embed = False
    except Exception:
        can_embed = False

    col1, col2 = st.columns([1.5, 1])

    with col1:
        section_label("Page Preview")

        if can_embed:
            components.iframe(url, height=500, scrolling=True)
        else:
            st.warning("⚠️ The content preview is limited by the website's settings.")

    with col2:
        with st.spinner("Retrieving and summarizing the page content..."):
            try:
                texto_completo = AutoScraper(url)
                resumen = SUMMARY(texto_completo)
            except Exception as e:
                logger.error(f"Error extracting or summarizing the article: {e}")
                resumen = None

        if not texto_completo:
            st.error("❌ No se pudo recuperar el contenido del artículo. Comprueba la URL o intenta con otra página.")
            st.stop()

        if texto_completo and not resumen:
            st.warning("⚠️ The text was retrieved, but the summary could not be generated.")

        if resumen:
            section_label("Summary")

            resumen_html = html.escape(resumen).replace("\n", "<br>")
            st.markdown(
                f"""
                <div style="
                    height: 460px;
                    overflow-y: auto;
                    padding: 1.75rem 2rem;
                    font-size: 15.5px;
                    line-height: 1.8;
                    background-color: var(--surface);
                    border: 1px solid var(--border);
                    border-radius: var(--radius-lg);
                    box-shadow: var(--shadow-card);
                    color: var(--text-dim);
                    margin-top: 4px;
                    margin-bottom: 40px;
                ">
                    {resumen_html}
                </div>
                """,
                unsafe_allow_html=True
            )

        else:
            st.warning("⚠️ Error retrieving the content — the source may require a subscription.")

    # Language detection and dynamic translation.
    # The BERT bias/fake-news models are trained in English, so if the
    # article isn't in English we translate it before classifying it.
    # The original text is still used for the Groq-powered summary and
    # interpretation, which works well across languages.
    texto_para_modelos = texto_completo
    idioma_detectado = None
    try:
        idioma_detectado = detect(texto_completo)
    except LangDetectException:
        idioma_detectado = None

    if idioma_detectado and idioma_detectado != 'en':
        # Only translate up to the token budget that will actually be
        # classified (MAX_CHUNKS chunks of ~510 tokens each). Translating
        # the entire article when BIAS()/FAKE() will only use the first
        # chunks wastes time and API cost without improving the result. A
        # generous characters-per-token margin is used to avoid the risk
        # of cutting off content that would actually be analyzed.
        CHARS_POR_TOKEN_MARGEN = 6
        max_chars_traduccion = MAX_CHUNKS * 510 * CHARS_POR_TOKEN_MARGEN
        texto_para_traducir = texto_completo[:max_chars_traduccion]

        with st.spinner(f"Translating article ('{idioma_detectado}' → English) for analysis..."):
            traduccion = TRANSLATE(texto_para_traducir)
        if traduccion:
            texto_para_modelos = traduccion
        else:
            st.warning("⚠️ Translation failed — analyzing the original text instead. Results may be less accurate for non-English content.")

    with st.spinner("Loading models..."):
        model_bias, tokenizer_bias, le, model_fake, tokenizer_fake = load_models()

    with st.spinner("Making predictions..."):

        # BIAS and FAKE are two fully independent models (neither needs
        # the other's result), so instead of running them one after the
        # other (duration1 + duration2 seconds total), they're launched
        # concurrently in two threads. PyTorch releases the GIL during
        # the heavy forward-pass computation, so both threads can make
        # real progress in parallel on multi-core machines — total time
        # ends up depending on the slower of the two, not their sum.
        start_parallel = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_bias = executor.submit(BIAS, texto_para_modelos, tokenizer_bias, model_bias, le)
            future_fake = executor.submit(FAKE, texto_para_modelos, tokenizer_fake, model_fake)

            try:
                results1 = future_bias.result()
            except Exception as e:
                logger.error(f"Error in bias prediction (BIAS): {e}")
                results1 = None

            try:
                results2 = future_fake.result()
            except Exception as e:
                logger.error(f"Error in fake-news prediction (FAKE): {e}")
                results2 = None

        duration_parallel = time.time() - start_parallel
        # Since they run in parallel, both share roughly the same
        # wall-clock time; the same value is shown on both result cards
        # instead of timing each one separately, which would no longer
        # reflect the actual perceived time.
        duration1 = duration_parallel
        duration2 = duration_parallel

        if results1:
            pred1 = results1['prediction']
            bias_fig1 = FigBarras(results1)
            bias_fig2 = FigTarta(results1)

        if results2:
            pred2 = results2['prediction']
            fake_fig1 = FigBarras(results2)
            fake_fig2 = FigTarta(results2)

        # LLM interpretation via Groq — guarded with truncation and a robust timeout
        if results1 and results2:
            start3 = time.time()
            # Safe truncation of the input text to avoid excess cost and token-context overflow
            texto_truncado_llm = texto_completo[:MAX_TEXT_CHARS_LLM] if texto_completo else ""

            # Run in a separate thread to avoid the app hanging indefinitely if the API doesn't respond
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(EXPLAIN, texto_truncado_llm, results1, results2)
                try:
                    llm_result = future.result(timeout=35)  # 35-second hard timeout guard
                except concurrent.futures.TimeoutError:
                    logger.warning("The EXPLAIN call exceeded the configured timeout (35s).")
                    llm_result = None
                except Exception as ex_llm:
                    logger.error(f"Internal exception in the LLM call: {ex_llm}")
                    llm_result = None
            duration3 = time.time() - start3
        else:
            llm_result = None

    tab1, tab2, tab3 = st.tabs(["BiasDetector", "FakeNewsDetector", "LLM-Interpretation"])

    with tab1:
        if results1:
            st.subheader("Political Bias Predictor")
            num_chunks1 = results1.get('num_chunks', 1)
            chunk_note1 = f"📄 Analyzed in {num_chunks1} chunk{'s' if num_chunks1 != 1 else ''} of up to 512 tokens"
            render_prediction_card(pred1, BIAS_COLORS, duration1, chunk_note1)

            chart11, chart12 = st.tabs(["Bar Chart", "Pie Chart"])
            with chart11:
                st.plotly_chart(bias_fig1, use_container_width=True)
            with chart12:
                st.plotly_chart(bias_fig2, use_container_width=True)

            with st.expander("View raw results (JSON)"):
                st.json(results1)
        else:
            st.warning("⚠️ Political bias prediction could not be performed. Please check the URL or the page content.")

    with tab2:
        if results2:
            st.subheader("Fake News Detector")
            num_chunks2 = results2.get('num_chunks', 1)
            chunk_note2 = f"📄 Analyzed in {num_chunks2} chunk{'s' if num_chunks2 != 1 else ''} of up to 512 tokens"
            render_prediction_card(pred2, FAKE_COLORS, duration2, chunk_note2)

            chart21, chart22 = st.tabs(["Bar Chart", "Pie Chart"])
            with chart21:
                st.plotly_chart(fake_fig1, use_container_width=True)
            with chart22:
                st.plotly_chart(fake_fig2, use_container_width=True)

            with st.expander("View raw results (JSON)"):
                st.json(results2)
        else:
            st.warning("⚠️ Fake News detection could not be performed. Please check the URL or the page content.")

    with tab3:
        if llm_result:
            st.subheader("Interpretation")

            st.markdown(
                f"""
                <div style="display:inline-flex;align-items:center;gap:8px;background:var(--surface);
                            border:1px solid var(--border);padding:0.5rem 1.1rem;border-radius:999px;
                            color:var(--text-mute);font-size:0.85rem;margin-bottom:1.75rem;">
                    ⏱ Execution time: <b style="color:var(--text);">{duration3:.2f}s</b>
                </div>
                """,
                unsafe_allow_html=True
            )

            # llm_result is a dict with guaranteed keys — no fragile string parsing needed
            for titulo, key in [
                ("EXPLANATION", "interpretation"),
                ("JUSTIFICATION", "justification"),
                ("RISK WARNING", "risk_analysis")
            ]:
                contenido = llm_result.get(key, "")
                contenido_html = html.escape(contenido).replace("\n", "<br>")
                accent = LLM_SECTION_COLORS.get(key, "#5b8cff")
                st.markdown(
                    f"""
                    <div style="
                        background-color: var(--surface);
                        border: 1px solid var(--border);
                        border-top: 3px solid {accent};
                        border-radius: 18px;
                        box-shadow: var(--shadow-card);
                        font-size: 15.5px;
                        line-height: 1.8;
                        color: var(--text-dim);
                        margin: 24px 0;
                        padding: 2rem 2.2rem;
                    ">
                        <h5 style="
                            margin-top: 0;
                            margin-bottom: 1.1rem;
                            color: var(--text);
                            font-size: 0.8rem;
                            letter-spacing: 0.12em;
                            text-transform: uppercase;
                            font-weight: 700;
                        ">{titulo}</h5>
                        {contenido_html}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with st.expander("View raw LLM output (JSON)"):
                st.json(llm_result)

        else:
            st.warning("⚠️ Analysis could not be generated using the LLM model.")