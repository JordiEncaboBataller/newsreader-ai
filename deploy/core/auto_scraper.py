import requests
import logging
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import trafilatura
import time

logger = logging.getLogger(__name__)

# Cabecera realista para reducir el bloqueo de simple_try() por parte de
# sitios con protección anti-bot (El País, The Hill, etc.)
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9,es;q=0.8",
}


def get_driver():
    options = uc.ChromeOptions()
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--disable-popup-blocking")
    
    # Flags de robustez para evitar crashes en servidores Linux/entornos VPS
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    
    # Refuerza el disfraz anti-detección (además del que ya aplica uc por
    # defecto) para reducir el bloqueo por parte de sistemas anti-bot como
    # PerimeterX/Datadome/Cloudflare, que suelen usar El País o The Hill.
    options.add_argument("--disable-blink-features=AutomationControlled")

    prefs = {
        "profile.default_content_setting_values.popups": 0,
        "profile.default_content_settings.popups": 0,
        "profile.managed_default_content_settings.popups": 0,
    }
    options.add_experimental_option("prefs", prefs)

    # IMPORTANTE: el modo headless de undetected_chromedriver debe activarse
    # con el parámetro headless=True del constructor, NO con
    # options.add_argument("--headless=..."). uc necesita aplicar sus propios
    # parches internos (userAgent, navigator.webdriver, etc.) para que el
    # modo headless siga pareciendo un navegador normal; si se fuerza el
    # flag manualmente, en Windows puede seguir abriendo una ventana visible
    # Y además delatar que es un bot, provocando que sitios con protección
    # anti-bot bloqueen el contenido.
    driver = uc.Chrome(options=options, headless=True)
    
    # Evita que el driver se quede colgado indefinidamente esperando la respuesta de un servidor lento
    driver.set_page_load_timeout(25)
    return driver

def simple_try(url):
    try:
        response = requests.get(url, headers=HEADERS, timeout=8)
        if response.status_code == 200:
            return response.text
    except requests.RequestException as e:
        logger.warning(f"Request failed: {e}")
    return None

def driver_try(url):
    driver = None
    try:
        driver = get_driver()
        driver.get(url)
        
        wait = WebDriverWait(driver, 20)
        wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
        
        # Pequeña espera adicional: algunos sistemas anti-bot muestran un
        # "challenge" JS de unos segundos antes de servir el contenido real.
        time.sleep(2)
        page_source = driver.page_source
        return page_source
    except Exception as e:
        logger.warning(f"Driver failed: {e}")
        return None
    finally:
        # Garantía absoluta de cierre del proceso de Chrome para evitar fugas de memoria (RAM leak)
        if driver is not None:
            try:
                driver.quit()
                logger.info("Driver cerrado de manera segura.")
            except Exception as ce:
                logger.error(f"Error crítico al intentar cerrar el driver: {ce}")

def AutoScraper(url):
    html = simple_try(url)
    logger.info(f"simple_try -> {len(html) if html else 0} chars")

    if not html:
        html = driver_try(url)
        logger.info(f"driver_try -> {len(html) if html else 0} chars")

    if html:
        try:
            resultado = trafilatura.extract(html, include_comments=False, include_tables=False)
            logger.info(f"trafilatura -> {len(resultado) if resultado else 0} chars")
            return resultado
        except Exception as e:
            logger.error(f"trafilatura error: {e}")
            return None

    logger.error("No se pudo obtener el texto del artículo (todos los métodos de scraping fallaron).")
    return None