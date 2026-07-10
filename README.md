# NewsReader AI

Aplicación en Streamlit que analiza artículos de noticias a partir de una URL:
detecta **sesgo político**, probabilidad de **fake news**, genera un **resumen**
y una **interpretación en lenguaje natural** de los resultados usando modelos
Transformer locales + la API de Mistral.

## Índice

- [Requisitos previos](#requisitos-previos)
- [Instalación](#instalación)
- [Variables de entorno](#variables-de-entorno)
- [Modelos (`ModelsBias` / `ModelsFake`)](#modelos-modelsbias--modelsfake)
- [Ejecutar la aplicación](#ejecutar-la-aplicación)
- [Estructura del proyecto](#estructura-del-proyecto)
- [Limitaciones conocidas](#limitaciones-conocidas)

---

## Requisitos previos

- Python 3.10+
- Google Chrome instalado (necesario para el scraping de respaldo vía Selenium/`undetected_chromedriver` cuando la petición HTTP simple falla)
- Una API key de [Mistral AI](https://console.mistral.ai/)

## Instalación

```bash
git clone <url-de-tu-repo>
cd <carpeta-del-repo>

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

## Variables de entorno

Copia la plantilla y rellena tus propias claves:

```bash
cp env.example .env
```

```
MISTRAL_API_KEY="tu_clave_de_mistral"
NGROK_AUTH_TOKEN="tu_token_de_ngrok"   # solo si usas start.ipynb para exponer la app vía túnel
```

⚠️ **Nunca subas tu `.env` real a Git.** Ya está incluido en `.gitignore`.

## Modelos (`ModelsBias` / `ModelsFake`)

Los pesos de los modelos **no están incluidos en este repositorio** (pesan
demasiado para GitHub) — están alojados en Hugging Face Hub:

- `ModelsBias` → https://huggingface.co/JordiEncabo/newsreader-bias
- `ModelsFake` → https://huggingface.co/JordiEncabo/newsreader-fake

### Descargarlos

```bash
pip install huggingface_hub

huggingface-cli download JordiEncabo/newsreader-bias --local-dir ./ModelsBias
huggingface-cli download JordiEncabo/newsreader-fake --local-dir ./ModelsFake
```

Esto coloca los archivos directamente donde `app.py` los espera — no hace
falta tocar el código.

### Estructura de carpetas esperada

```
ModelsBias/
├── config.json
├── label_encoder.pkl          ← archivo propio del proyecto (no lo genera from_pretrained)
├── model.safetensors
├── special_tokens_map.json
├── tokenizer_config.json
├── tokenizer.json
└── vocab.txt

ModelsFake/
├── config.json
├── model.safetensors           ← el que usa el código (use_safetensors=True)
├── special_tokens_map.json
├── tokenizer_config.json
├── tokenizer.json
└── vocab.txt
```

## Ejecutar la aplicación

```bash
streamlit run app.py
```

La app se abrirá en `http://localhost:8501`.

Alternativamente, `start.ipynb` lanza la app y la expone públicamente a través
de un túnel de ngrok (requiere `NGROK_AUTH_TOKEN` en tu `.env`).

## Estructura del proyecto

```
App/                              ← raíz del repositorio (inicializa git aquí)
├── Imags/
│   ├── LogoClaro.png
│   └── LogoOscuro.png            ← el único que usa app.py actualmente
├── ModelsBias/                   ← NO se sube (ver sección Modelos)
├── ModelsFake/                   ← NO se sube (ver sección Modelos)
├── Notebooks Ejemplos/           ← opcional, no necesario para ejecutar la app
├── .env                          ← NO se sube (tus claves reales)
├── .gitignore
├── app.py                        # Interfaz Streamlit y orquestación del pipeline
├── auto_scraper.py                # Extracción de texto del artículo (requests + Selenium de respaldo)
├── chunking_utils.py               # División y batching de texto largo en bloques de tokens
├── env.example
├── figuras.py                      # Gráficos Plotly (barras y tarta) de las probabilidades
├── mistral_sm.py                    # Prompts y llamadas a la API de Mistral (resumen, traducción, interpretación)
├── predictor_bias.py                 # Inferencia del modelo de sesgo político
├── predictor_fake.py                  # Inferencia del modelo de fake news
├── README.md
├── requirements.txt
└── start.ipynb                        # Lanzador opcional vía ngrok
```

## Limitaciones conocidas

- El scraping de respaldo usa Chrome headless vía `undetected_chromedriver`.
  Si despliegas esta app en un servidor/PaaS que no tenga Chrome instalado
  (por ejemplo, Streamlit Community Cloud sin configuración adicional), ese
  respaldo fallará silenciosamente y solo funcionará el método de `requests`
  simple.
- Los modelos se cargan y ejecutan en CPU por defecto. Si tu servidor tiene
  GPU disponible, hay margen de mejora de rendimiento moviendo los modelos a
  `cuda`.