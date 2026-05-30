import os
import json
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import requests
import schedule
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ---------------------------------------------------------------------------
# Konfigurim
# ---------------------------------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GUMROAD_ACCESS_TOKEN = os.getenv("GUMROAD_ACCESS_TOKEN", "")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
SCHEDULE_TIMES = os.getenv("SCHEDULE_TIMES", "08:00,13:00,20:00")
MIN_PRICE_USD = float(os.getenv("MIN_PRICE_USD", "5.00"))
MAX_DISCOUNT_PERCENT = float(os.getenv("MAX_DISCOUNT_PERCENT", "40"))
DRY_RUN = os.getenv("DRY_RUN", "false").lower() == "true"
USE_GROQ = os.getenv("USE_GROQ", "false").lower() == "true"
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

GUMROAD_API = "https://api.gumroad.com/v2"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
log_file = LOG_DIR / f"bot_{datetime.now().strftime('%Y%m%d')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Telegram
# ---------------------------------------------------------------------------
def telegram_send(message: str) -> bool:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return False
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message[:4000],
            "parse_mode": "HTML",
        }
        r = requests.post(url, json=payload, timeout=10)
        r.raise_for_status()
        return True
    except Exception as e:
        log.warning("Telegram dështoi: %s", e)
        return False


def notify(text: str):
    log.info(text)
    telegram_send(f"{text}")


def notify_error(text: str):
    log.error(text)
    telegram_send(f"[GABIM] {text}")


def notify_success(text: str):
    log.info(text)
    telegram_send(f"[OK] {text}")


# ---------------------------------------------------------------------------
# Gumroad API
# ---------------------------------------------------------------------------
def _gumroad_get(endpoint: str, params: Optional[dict] = None) -> dict:
    if params is None:
        params = {}
    params["access_token"] = GUMROAD_ACCESS_TOKEN
    url = f"{GUMROAD_API}/{endpoint}"
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def _gumroad_put(endpoint: str, data: dict) -> dict:
    data["access_token"] = GUMROAD_ACCESS_TOKEN
    url = f"{GUMROAD_API}/{endpoint}"
    r = requests.put(url, data=data, timeout=30)
    r.raise_for_status()
    return r.json()


def _gumroad_post(endpoint: str, data: dict) -> dict:
    data["access_token"] = GUMROAD_ACCESS_TOKEN
    url = f"{GUMROAD_API}/{endpoint}"
    r = requests.post(url, data=data, timeout=30)
    r.raise_for_status()
    return r.json()


def get_products() -> list[dict]:
    resp = _gumroad_get("products")
    return resp.get("products", [])


def get_sales(days: int = 7) -> list[dict]:
    since = (datetime.now() - timedelta(days=days)).isoformat()
    resp = _gumroad_get("sales", {"after": since})
    return resp.get("sales", [])


def get_product_sales(product_id: str, days: int = 7) -> list[dict]:
    since = (datetime.now() - timedelta(days=days)).isoformat()
    resp = _gumroad_get(f"products/{product_id}/sales", {"after": since})
    return resp.get("sales", [])


def update_product(product_id: str, data: dict) -> dict:
    if DRY_RUN:
        log.info("[DRY-RUN] do ta përditësonte produktin %s: %s", product_id, data)
        return {"success": True, "dry_run": True}
    return _gumroad_put(f"products/{product_id}", data)


def create_offer_code(product_id: str, name: str, amount_cents: int, code: str = None) -> dict:
    if DRY_RUN:
        log.info("[DRY-RUN] do të krijonte ofertë për %s: %s (%d cent)", product_id, name, amount_cents)
        return {"success": True, "dry_run": True}
    payload = {"name": name, "amount_cents": amount_cents}
    if code:
        payload["code"] = code
    return _gumroad_post(f"products/{product_id}/offer_codes", payload)


# ---------------------------------------------------------------------------
# AI Provider
# ---------------------------------------------------------------------------
AI_PROVIDER = "groq" if (USE_GROQ and GROQ_API_KEY) else "openai"
AI_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile") if AI_PROVIDER == "groq" else os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def get_client():
    if AI_PROVIDER == "groq":
        return OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")
    return OpenAI(api_key=OPENAI_API_KEY)


def ai_analizo(products: list[dict], sales: list[dict]) -> dict:
    prompt = f"""
Je një asistent marketingu për Gumroad. Analizo të dhënat dhe propozo strategji.

Data: {datetime.now().strftime('%Y-%m-%d %H:%M')}

Produktet:
{json.dumps([{
    'id': p['id'],
    'name': p.get('name'),
    'price': p.get('price'),
    'sales_count': p.get('sales_count'),
    'revenue': p.get('revenue'),
} for p in products], indent=2)}

Shitjet e fundit ({len(sales)} gjithsej):
{json.dumps([{
    'product_name': s.get('product_name'),
    'price': s.get('price'),
    'created_at': s.get('created_at'),
} for s in sales[:20]], indent=2)}

Kthe një JSON të vlefshëm pa tekst tjetër:
{{
    "analiza": "përmbledhje e shkurtër e gjendjes",
    "produkti_target": "id e produktit që ka nevojë për vëmendje",
    "veprimi": "ulo_cmimin" | "krijo_zbritje" | "asgje",
    "cmimi_ri": 9.99,
    "zbritja_perqindje": 20,
    "arsyeja": "pse rekomandon këtë veprim"
}}
"""
    try:
        client = get_client()
        resp = client.chat.completions.create(
            model=AI_MODEL,
            messages=[
                {"role": "system", "content": "Je një analist marketingu për Gumroad. Kthe vetëm JSON të vlefshëm."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=800,
        )
        text = resp.choices[0].message.content.strip()
        text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        return json.loads(text)
    except Exception as e:
        log.error("Gabim gjatë thirrjes së OpenAI: %s", e)
        return {"analiza": "Nuk u arrit analiza.", "produkti_target": None, "veprimi": "asgje"}


# ---------------------------------------------------------------------------
# Guardrails
# ---------------------------------------------------------------------------
def guardrails_kontrollo(veprimi: str, cmimi_ri: Optional[float], zbritja: Optional[float]) -> tuple[bool, str]:
    if veprimi == "ulo_cmimin" and cmimi_ri is not None:
        if cmimi_ri < MIN_PRICE_USD:
            return False, f"Cmimi ${cmimi_ri:.2f} është nën minimumin ${MIN_PRICE_USD:.2f}. BLLOKUAR."
    if veprimi == "krijo_zbritje" and zbritja is not None:
        if zbritja > MAX_DISCOUNT_PERCENT:
            return False, f"Zbritja {zbritja:.0f}% kalon maksimumin {MAX_DISCOUNT_PERCENT:.0f}%. BLLOKUAR."
    return True, ""


# ---------------------------------------------------------------------------
# Raporti
# ---------------------------------------------------------------------------
def gjenero_raport(products: list[dict], sales: list[dict], analiza: dict) -> str:
    now = datetime.now()
    total_revenue = sum(float(s.get("price", 0) or 0) for s in sales)
    total_sales = len(sales)
    active_products = len(products)
    total_product_sales = sum(int(p.get("sales_count", 0) or 0) for p in products)

    lines = [
        f"{'='*50}",
        f"  RAPORTI GUMROAD AI BOT",
        f"  Data: {now.strftime('%Y-%m-%d %H:%M')}",
        f"{'='*50}",
        f"",
        f"  Produkte aktive: {active_products}",
        f"  Shitje totale: {total_sales} (periudhe)",
        f"  Te ardhura: ${total_revenue:.2f}",
        f"  Shitje produktesh (gjithsej): {total_product_sales}",
        f"",
        f"  Analiza AI:",
        f"  {analiza.get('analiza', 'N/A')}",
        f"",
        f"  Veprimi i rekomanduar: {analiza.get('veprimi', 'asgje')}",
        f"  Produkti: {analiza.get('produkti_target', 'N/A')}",
        f"  Arsyja: {analiza.get('arsyeja', 'N/A')}",
        f"",
        f"  Guardrails: MIN_PRICE=${MIN_PRICE_USD:.2f}, MAX_DISCOUNT={MAX_DISCOUNT_PERCENT:.0f}%",
        f"{'='*50}",
    ]
    return "\n".join(lines)


def ruaj_raport(content: str):
    now = datetime.now()
    fname = LOG_DIR / f"raport_{now.strftime('%Y%m%d_%H%M')}.txt"
    fname.write_text(content, encoding="utf-8")
    log.info("Raporti u ruajt: %s", fname)


# ---------------------------------------------------------------------------
# Cikli kryesor
# ---------------------------------------------------------------------------
def ekzekuto_ciklin(cikli_emri: str = "Automatik"):
    notify(f"[Cikli: {cikli_emri}] Fillon analiza...")

    try:
        products = get_products()
        sales = get_sales(days=7)
    except Exception as e:
        notify_error(f"Gabim gjatë marrjes së të dhënave nga Gumroad: {e}")
        return

    if not products:
        notify_error("Nuk u gjetën produkte në llogari.")
        return

    analiza = ai_analizo(products, sales)
    raport = gjenero_raport(products, sales, analiza)
    ruaj_raport(raport)
    notify(raport)

    veprimi = analiza.get("veprimi", "asgje")

    if veprimi == "asgje":
        notify_success("Bot nuk sheh nevojë për ndryshime. Gjithçka në rregull.")
        return

    produkti_id = analiza.get("produkti_target")
    cmimi_ri = analiza.get("cmimi_ri")
    zbritja_p = analiza.get("zbritja_perqindje")

    if veprimi == "ulo_cmimin" and produkti_id and cmimi_ri:
        kalon, mesazh = guardrails_kontrollo(veprimi, cmimi_ri, None)
        if not kalon:
            notify_error(f"Guardrails bllokoi uljen e cmimit: {mesazh}")
            return
        data = {"price": cmimi_ri}
        try:
            update_product(produkti_id, data)
            notify_success(f"Cmimi u ul në ${cmimi_ri:.2f} për produktin {produkti_id}")
        except Exception as e:
            notify_error(f"Gabim gjatë uljes së cmimit: {e}")

    elif veprimi == "krijo_zbritje" and produkti_id and zbritja_p:
        kalon, mesazh = guardrails_kontrollo(veprimi, None, zbritja_p)
        if not kalon:
            notify_error(f"Guardrails bllokoi zbritjen: {mesazh}")
            return
        try:
            product = next((p for p in products if p["id"] == produkti_id), None)
            price_cents = int(float(product.get("price", 0) or 0) * 100) if product else 0
            amount_cents = int(price_cents * (zbritja_p / 100))
            code_name = f"AI{datetime.now().strftime('%d%m')}"
            create_offer_code(produkti_id, f"Zbritje AI {zbritja_p:.0f}%", amount_cents, code_name)
            notify_success(f"Kodi i zbritjes {code_name} ({zbritja_p:.0f}%) u krijua për produktin {produkti_id}")
        except Exception as e:
            notify_error(f"Gabim gjatë krijimit të zbritjes: {e}")


def raport_ditor():
    try:
        products = get_products()
        sales = get_sales(days=1)
    except Exception as e:
        notify_error(f"Gabim në raportin ditor: {e}")
        return
    total = sum(float(s.get("price", 0) or 0) for s in sales)
    count = len(sales)
    msg = (
        f"RAPORTI DITOR\n"
        f"Shitje sot: {count}\n"
        f"Te ardhura: ${total:.2f}\n"
        f"Produkte: {len(products)}"
    )
    notify(msg)


# ---------------------------------------------------------------------------
# Planifikuesi
# ---------------------------------------------------------------------------
def setup_schedule():
    times = [t.strip() for t in SCHEDULE_TIMES.split(",") if t.strip()]
    for t in times:
        schedule.every().day.at(t).do(ekzekuto_ciklin, cikli_emri=f"Ora {t}")
        log.info("Cikli i planifikuar: %s", t)
    schedule.every().day.at("23:59").do(raport_ditor)
    log.info("Raporti ditor i planifikuar: 23:59")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    banner = r"""
  ____ ____  _   _ ____  _   _ ____    _    ____ ___ _____ ___ ____
 / ___|  _ \| | | |  _ \| | | |  _ \  / \  |  _ \_ _|_   _|_ _|  _ \
| |  _| |_) | | | | |_) | | | | | | |/ _ \ | |_) | |  | |  | || |_) |
| |_| |  _ <| |_| |  _ <| |_| | |_| / ___ \|  _ <| |  | |  | ||  _ <
 \____|_| \_\\___/|_| \_\\___/|____/_/   \_\_| \_\_|  |_| |___|_| \_\
    """
    print(banner)
    log.info("=" * 50)
    log.info("GUMROAD AI BOT 24/7 — NISJE")
    log.info("Data: %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    ai_provider_display = "Groq (Falas)" if AI_PROVIDER == "groq" else "OpenAI"
    log.info("DRY_RUN: %s", DRY_RUN)
    log.info("AI Provider: %s (%s)", ai_provider_display, AI_MODEL)
    log.info("Min cmim: $%.2f | Zbritje max: %.0f%%", MIN_PRICE_USD, MAX_DISCOUNT_PERCENT)
    log.info("=" * 50)

    notify(
        f"Gumroad AI Bot 24/7 u nis!\n"
        f"DRY_RUN: {DRY_RUN}\n"
        f"AI: {ai_provider_display} ({AI_MODEL})\n"
        f"Ciklet: {SCHEDULE_TIMES}\n"
        f"Min cmim: ${MIN_PRICE_USD:.2f} | Max zbritje: {MAX_DISCOUNT_PERCENT:.0f}%"
    )

    if AI_PROVIDER == "groq" and not GROQ_API_KEY:
        notify_error("GROQ_API_KEY mungon.")
    elif AI_PROVIDER == "openai" and not OPENAI_API_KEY.startswith("sk-"):
        notify_error("OPENAI_API_KEY nuk duket e vlefshme.")
    if not GUMROAD_ACCESS_TOKEN:
        notify_error("GUMROAD_ACCESS_TOKEN mungon.")

    setup_schedule()

    if not DRY_RUN:
        ekzekuto_ciklin("Nisje")

    log.info("Bot-i në pritje të cikleve të planifikuara...")

    try:
        while True:
            schedule.run_pending()
            time.sleep(30)
    except KeyboardInterrupt:
        log.info("Bot u ndal nga përdoruesi (Ctrl+C).")
        notify("Bot u ndal manualisht.")


if __name__ == "__main__":
    main()
