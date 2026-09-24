"""Article scraping utilities.

Fetches the raw HTML of a news article URL and extracts its main text.
Uses a two-tier strategy: a fast plain HTTP request first, falling back to
a headless Chrome browser (via undetected-chromedriver) only when the
simple request is blocked, e.g. by anti-bot protection.
"""

import requests
import logging
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import trafilatura
import time

logger = logging.getLogger(__name__)

# Realistic browser header to reduce simple_try() being blocked by sites
# with anti-bot protection (El País, The Hill, etc.).
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9,es;q=0.8",
}


def get_driver():
    """Creates and configures an undetected-chromedriver Chrome instance
    for headless scraping, hardened against crashes and bot detection."""
    options = uc.ChromeOptions()
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--disable-popup-blocking")

    # Robustness flags to avoid crashes on Linux servers / VPS environments.
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")

    # Reinforces anti-detection (on top of what undetected-chromedriver
    # already applies by default) to reduce blocking by anti-bot systems
    # such as PerimeterX/Datadome/Cloudflare, commonly used by El País or
    # The Hill.
    options.add_argument("--disable-blink-features=AutomationControlled")

    prefs = {
        "profile.default_content_setting_values.popups": 0,
        "profile.default_content_settings.popups": 0,
        "profile.managed_default_content_settings.popups": 0,
    }
    options.add_experimental_option("prefs", prefs)

    # IMPORTANT: undetected-chromedriver's headless mode must be enabled via
    # the constructor's headless=True argument, NOT via
    # options.add_argument("--headless=..."). uc needs to apply its own
    # internal patches (userAgent, navigator.webdriver, etc.) for headless
    # mode to still look like a normal browser; forcing the flag manually
    # can leave a visible window open on Windows AND reveal that it's a
    # bot, causing anti-bot-protected sites to block the content.
    driver = uc.Chrome(options=options, headless=True)

    # Prevents the driver from hanging indefinitely on a slow server response.
    driver.set_page_load_timeout(25)
    return driver


def simple_try(url):
    """Attempts a plain HTTP GET request. Returns the raw HTML on success
    (status 200), or None if the request fails or is blocked."""
    try:
        response = requests.get(url, headers=HEADERS, timeout=8)
        if response.status_code == 200:
            return response.text
    except requests.RequestException as e:
        logger.warning(f"Request failed: {e}")
    return None


def driver_try(url):
    """Fallback strategy: loads the page in a headless Chrome browser and
    returns the rendered HTML. Used when simple_try() is blocked. Always
    closes the browser process, even on failure, to avoid memory leaks."""
    driver = None
    try:
        driver = get_driver()
        driver.get(url)

        wait = WebDriverWait(driver, 20)
        wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))

        # Small extra wait: some anti-bot systems show a JS "challenge"
        # for a few seconds before serving the real content.
        time.sleep(2)
        page_source = driver.page_source
        return page_source
    except Exception as e:
        logger.warning(f"Driver failed: {e}")
        return None
    finally:
        # Absolute guarantee that the Chrome process is closed, to avoid RAM leaks.
        if driver is not None:
            try:
                driver.quit()
                logger.info("Driver closed safely.")
            except Exception as ce:
                logger.error(f"Critical error while closing the driver: {ce}")


def AutoScraper(url):
    """Extracts the main article text from a URL.

    Tries a plain HTTP request first; if that fails, falls back to a
    headless browser. The resulting HTML (from either method) is then
    cleaned with trafilatura to strip navigation, ads and boilerplate,
    leaving only the article body.

    Returns the extracted text, or None if every step fails.
    """
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

    logger.error("Could not retrieve the article text (all scraping methods failed).")
    return None