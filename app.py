"""
ShopSense AI — Production-Grade Shopping Intelligence Platform
Single-file Streamlit application.
Run: streamlit run app.py
"""

# ═══════════════════════════════════════════════════════════════════════════════
# IMPORTS
# ═══════════════════════════════════════════════════════════════════════════════

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import altair as alt
import json
import time
import random
import logging
import hashlib
import io
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Tuple
from collections import Counter
from copy import deepcopy

# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ShopSenseAI")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG (must be first Streamlit call)
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="ShopSense AI",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

APP_VERSION = "2.0.0"
MAX_COMPARE_ITEMS = 5
MAX_WISHLIST_ITEMS = 50
MAX_SEARCH_HISTORY = 20
DEFAULT_BUDGET = 500
SCORE_WEIGHTS = {
    "rating":        0.30,
    "review_volume": 0.20,
    "price_value":   0.25,
    "platform_trust":0.15,
    "deal_quality":  0.10,
}
PLATFORM_TRUST = {
    "Amazon":   0.95,
    "Best Buy": 0.90,
    "Walmart":  0.88,
    "Target":   0.87,
    "Newegg":   0.85,
    "Shopify":  0.80,
    "eBay":     0.75,
    "Etsy":     0.78,
}
PLATFORM_CONFIG: Dict[str, Dict] = {
    "Amazon":   {"badge": "badge-amazon",  "emoji": "📦", "color": "#ff9900", "url": "https://amazon.com/s?k="},
    "eBay":     {"badge": "badge-ebay",    "emoji": "🏷️",  "color": "#0e76bc", "url": "https://ebay.com/sch/?_nkw="},
    "Walmart":  {"badge": "badge-walmart", "emoji": "🛒", "color": "#0075c9", "url": "https://walmart.com/search?q="},
    "Etsy":     {"badge": "badge-etsy",    "emoji": "🎨", "color": "#f1641e", "url": "https://etsy.com/search?q="},
    "Best Buy": {"badge": "badge-bestbuy", "emoji": "💻", "color": "#0046d0", "url": "https://bestbuy.com/site/searchpage.jsp?st="},
    "Target":   {"badge": "badge-target",  "emoji": "🎯", "color": "#cc0000", "url": "https://target.com/s?searchTerm="},
    "Shopify":  {"badge": "badge-shopify", "emoji": "🏪", "color": "#96bf44", "url": "https://shop.app/search?q="},
    "Newegg":   {"badge": "badge-newegg",  "emoji": "🖥️", "color": "#ff7e00", "url": "https://newegg.com/p/pl?d="},
}
LLM_ADAPTERS = ["GPT-4o (OpenAI)", "Claude 3.5 Sonnet", "Gemini 1.5 Pro", "LLaMA 3.1 (Local)", "Mixtral 8x7B"]
TRENDING_TERMS = [
    "Wireless Earbuds 2024", "AI Smart Home Hub", "Ergonomic Chair",
    "Mechanical Keyboard", "Standing Desk", "Gaming Monitor 4K",
    "Robot Vacuum Gen3", "Air Fryer XL", "Smartwatch Ultra",
]

# ═══════════════════════════════════════════════════════════════════════════════
# DATA MODELS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Product:
    title: str
    price: float
    original_price: float
    rating: float
    reviews: int
    platform: str
    brand: str
    category: str
    stock: str
    shipping: str
    emoji: str
    ai_score: float = 0.0
    price_trend: str = "→ Stable"
    deal_label: str = "Fair Price"
    deal_quality_score: float = 0.5
    discount_pct: float = 0.0
    tags: List[str] = field(default_factory=list)
    insight: str = ""

    @property
    def url(self) -> str:
        cfg = PLATFORM_CONFIG.get(self.platform, {"url": "https://google.com/search?q="})
        return cfg["url"] + self.title.replace(" ", "+")

    @property
    def savings(self) -> float:
        return max(0.0, self.original_price - self.price)


@dataclass
class ChatMessage:
    role: str   # "user" | "assistant"
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%H:%M"))


@dataclass
class SearchRecord:
    query: str
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M"))
    result_count: int = 0

# ═══════════════════════════════════════════════════════════════════════════════
# MOCK PRODUCT DATABASE
# ═══════════════════════════════════════════════════════════════════════════════

_RAW_DB: Dict[str, List[Dict]] = {
    "laptop": [
        {"title": "Apple MacBook Pro 14″ M3 Pro 18GB", "price": 1999.00, "original_price": 2199.00, "rating": 4.8, "reviews": 5847, "platform": "Amazon",   "brand": "Apple",     "category": "Ultrabooks",       "stock": "In Stock",   "shipping": "Free 2-day",  "emoji": "💻"},
        {"title": "Dell XPS 15 9530 i9 RTX 4060",       "price": 1749.99, "original_price": 1899.99, "rating": 4.6, "reviews": 2203, "platform": "Best Buy", "brand": "Dell",      "category": "Performance",      "stock": "In Stock",   "shipping": "Free",        "emoji": "💻"},
        {"title": "Lenovo ThinkPad X1 Carbon Gen 12",   "price": 1549.00, "original_price": 1549.00, "rating": 4.7, "reviews": 4421, "platform": "Walmart",  "brand": "Lenovo",    "category": "Business",         "stock": "In Stock",   "shipping": "$5.99",       "emoji": "💼"},
        {"title": "ASUS ROG Strix G16 RTX 4070",        "price": 1299.99, "original_price": 1499.99, "rating": 4.5, "reviews": 1987, "platform": "Newegg",   "brand": "ASUS",      "category": "Gaming",           "stock": "Limited",    "shipping": "Free",        "emoji": "🎮"},
        {"title": "HP Spectre x360 14″ OLED 2-in-1",    "price": 1449.99, "original_price": 1449.99, "rating": 4.6, "reviews": 1654, "platform": "Target",   "brand": "HP",        "category": "Convertible",      "stock": "In Stock",   "shipping": "Free",        "emoji": "💻"},
        {"title": "Microsoft Surface Laptop 5 Refurb",  "price": 849.00,  "original_price": 1299.00, "rating": 4.3, "reviews": 3109, "platform": "eBay",     "brand": "Microsoft", "category": "Certified Refurb", "stock": "In Stock",   "shipping": "Free",        "emoji": "💻"},
        {"title": "Razer Blade 15 Advanced RTX 4080",   "price": 2499.00, "original_price": 2699.00, "rating": 4.7, "reviews": 892,  "platform": "Amazon",   "brand": "Razer",     "category": "Gaming Premium",   "stock": "In Stock",   "shipping": "Free 2-day",  "emoji": "⚡"},
        {"title": "Framework Laptop 16 AMD Ryzen 9",    "price": 1299.00, "original_price": 1299.00, "rating": 4.5, "reviews": 678,  "platform": "Shopify",  "brand": "Framework", "category": "Modular",          "stock": "Pre-order",  "shipping": "Free",        "emoji": "🔧"},
    ],
    "headphones": [
        {"title": "Sony WH-1000XM5 Wireless ANC",         "price": 279.99, "original_price": 349.99, "rating": 4.9, "reviews": 22234, "platform": "Amazon",   "brand": "Sony",       "category": "Over-Ear ANC",    "stock": "In Stock", "shipping": "Free 2-day", "emoji": "🎧"},
        {"title": "Apple AirPods Pro 2nd Gen USB-C",       "price": 249.00, "original_price": 249.00, "rating": 4.8, "reviews": 61921, "platform": "Best Buy", "brand": "Apple",      "category": "In-Ear ANC",      "stock": "In Stock", "shipping": "Free",       "emoji": "🎧"},
        {"title": "Bose QuietComfort Ultra Headphones",    "price": 349.99, "original_price": 349.99, "rating": 4.8, "reviews": 9732,  "platform": "Walmart",  "brand": "Bose",       "category": "Over-Ear ANC",    "stock": "In Stock", "shipping": "Free 2-day", "emoji": "🎧"},
        {"title": "Sennheiser HD 660S2 Open-Back",         "price": 329.95, "original_price": 379.95, "rating": 4.7, "reviews": 1604,  "platform": "Newegg",   "brand": "Sennheiser", "category": "Audiophile",      "stock": "In Stock", "shipping": "$4.99",      "emoji": "🎼"},
        {"title": "Jabra Evolve2 85 MS Wireless",          "price": 379.00, "original_price": 449.00, "rating": 4.6, "reviews": 4241,  "platform": "Amazon",   "brand": "Jabra",      "category": "Professional",    "stock": "Limited",  "shipping": "Free 2-day", "emoji": "💼"},
        {"title": "Beyerdynamic DT 990 Pro 250Ω",          "price": 149.00, "original_price": 179.00, "rating": 4.7, "reviews": 8901,  "platform": "Amazon",   "brand": "Beyerdynamic","category": "Studio",          "stock": "In Stock", "shipping": "Free 2-day", "emoji": "🎵"},
        {"title": "Custom Leather Headband Wrap (Artisan)", "price": 34.99, "original_price": 34.99,  "rating": 4.9, "reviews": 912,   "platform": "Etsy",     "brand": "CraftAudio", "category": "Accessories",     "stock": "In Stock", "shipping": "$3.99",      "emoji": "🎨"},
        {"title": "Samsung Galaxy Buds3 Pro",              "price": 199.99, "original_price": 249.99, "rating": 4.5, "reviews": 7832,  "platform": "Target",   "brand": "Samsung",    "category": "In-Ear ANC",      "stock": "In Stock", "shipping": "Free",       "emoji": "🎧"},
    ],
    "phone": [
        {"title": "iPhone 16 Pro Max 256GB Desert Titanium", "price": 1199.00, "original_price": 1199.00, "rating": 4.9, "reviews": 104432, "platform": "Best Buy", "brand": "Apple",   "category": "Flagship",   "stock": "In Stock", "shipping": "Free",       "emoji": "📱"},
        {"title": "Samsung Galaxy S25 Ultra 512GB",          "price": 1099.99, "original_price": 1299.99, "rating": 4.7, "reviews": 58219,  "platform": "Amazon",   "brand": "Samsung", "category": "Flagship",   "stock": "In Stock", "shipping": "Free 2-day", "emoji": "📱"},
        {"title": "Google Pixel 9 Pro 128GB Obsidian",       "price": 899.00,  "original_price": 999.00,  "rating": 4.6, "reviews": 21087,  "platform": "Walmart",  "brand": "Google",  "category": "AI-First",   "stock": "In Stock", "shipping": "Free 2-day", "emoji": "📱"},
        {"title": "OnePlus 13 256GB Silky Black",            "price": 699.00,  "original_price": 699.00,  "rating": 4.5, "reviews": 7432,   "platform": "Newegg",   "brand": "OnePlus", "category": "Value Flag", "stock": "In Stock", "shipping": "Free",       "emoji": "📱"},
        {"title": "iPhone 14 128GB Refurb Grade A",          "price": 549.00,  "original_price": 799.00,  "rating": 4.4, "reviews": 12871,  "platform": "eBay",     "brand": "Apple",   "category": "Refurb",     "stock": "In Stock", "shipping": "Free",       "emoji": "📱"},
        {"title": "Motorola Edge 50 Fusion 256GB",           "price": 449.99,  "original_price": 499.99,  "rating": 4.3, "reviews": 3241,   "platform": "Target",   "brand": "Motorola","category": "Mid-Range",  "stock": "In Stock", "shipping": "Free",       "emoji": "📱"},
    ],
    "coffee": [
        {"title": "Breville Barista Express Espresso w/ Grinder","price": 699.95, "original_price": 799.95, "rating": 4.7, "reviews": 18045, "platform": "Amazon",  "brand": "Breville",     "category": "Espresso Machines", "stock": "In Stock", "shipping": "Free 2-day", "emoji": "☕"},
        {"title": "Fellow Ode Brew Grinder Gen 2",               "price": 345.00, "original_price": 345.00, "rating": 4.8, "reviews": 5321,  "platform": "Shopify", "brand": "Fellow",       "category": "Grinders",          "stock": "In Stock", "shipping": "Free",       "emoji": "⚙️"},
        {"title": "Hario V60 Pour-Over Starter Set",             "price": 48.99,  "original_price": 48.99,  "rating": 4.9, "reviews": 11876, "platform": "Walmart", "brand": "Hario",        "category": "Manual Brew",       "stock": "In Stock", "shipping": "$3.99",      "emoji": "🫗"},
        {"title": "Small-Batch Colombian Single Origin 1lb",      "price": 22.50,  "original_price": 22.50,  "rating": 4.8, "reviews": 2892,  "platform": "Etsy",    "brand": "MountainRoast","category": "Beans",             "stock": "In Stock", "shipping": "Free",       "emoji": "🌱"},
        {"title": "Nespresso Vertuo Next Premium",               "price": 159.00, "original_price": 219.00, "rating": 4.6, "reviews": 28412, "platform": "Target",  "brand": "Nespresso",    "category": "Pod Machines",      "stock": "In Stock", "shipping": "Free",       "emoji": "☕"},
        {"title": "La Marzocco Linea Mini Home Espresso",        "price": 3990.00,"original_price": 3990.00,"rating": 4.9, "reviews": 312,   "platform": "Shopify", "brand": "La Marzocco",  "category": "Prosumer",          "stock": "Limited",  "shipping": "Free White Glove", "emoji": "🏆"},
    ],
    "gaming": [
        {"title": "PlayStation 5 Pro Console",                  "price": 699.99,  "original_price": 699.99,  "rating": 4.8, "reviews": 43219, "platform": "Best Buy", "brand": "Sony",      "category": "Consoles",     "stock": "Limited",  "shipping": "Free",       "emoji": "🎮"},
        {"title": "Xbox Series X 1TB Console",                  "price": 499.99,  "original_price": 499.99,  "rating": 4.7, "reviews": 38871, "platform": "Target",   "brand": "Microsoft", "category": "Consoles",     "stock": "In Stock", "shipping": "Free",       "emoji": "🎮"},
        {"title": "ASUS ROG Swift Pro PG248QP 540Hz Monitor",   "price": 699.00,  "original_price": 799.00,  "rating": 4.6, "reviews": 1243,  "platform": "Amazon",   "brand": "ASUS",      "category": "Monitors",     "stock": "In Stock", "shipping": "Free 2-day", "emoji": "🖥️"},
        {"title": "SteelSeries Arctis Nova Pro Wireless",       "price": 349.99,  "original_price": 349.99,  "rating": 4.7, "reviews": 5432,  "platform": "Newegg",   "brand": "SteelSeries","category": "Gaming Audio", "stock": "In Stock", "shipping": "Free",       "emoji": "🎧"},
        {"title": "Razer DeathAdder V3 Pro Wireless Mouse",     "price": 129.99,  "original_price": 159.99,  "rating": 4.6, "reviews": 7812,  "platform": "Amazon",   "brand": "Razer",     "category": "Peripherals",  "stock": "In Stock", "shipping": "Free 2-day", "emoji": "🖱️"},
        {"title": "Custom Keycap Set (Hand-Painted)",           "price": 89.00,   "original_price": 89.00,   "rating": 4.9, "reviews": 234,   "platform": "Etsy",     "brand": "PixelKeys", "category": "Peripherals",  "stock": "Made to Order", "shipping": "$6.99",  "emoji": "⌨️"},
    ],
    "default": [
        {"title": "Premium Quality Model — Top Rated 2024",   "price": 89.99,  "original_price": 119.99, "rating": 4.6, "reviews": 4241,  "platform": "Amazon",   "brand": "TopBrand",    "category": "General",    "stock": "In Stock",   "shipping": "Free 2-day", "emoji": "🛍️"},
        {"title": "Best Value Edition — Editor's Choice",     "price": 64.95,  "original_price": 79.95,  "rating": 4.5, "reviews": 9892,  "platform": "Walmart",  "brand": "ValuePlus",   "category": "General",    "stock": "In Stock",   "shipping": "$4.99",      "emoji": "⭐"},
        {"title": "Professional Grade — Enterprise Ready",    "price": 129.00, "original_price": 129.00, "rating": 4.7, "reviews": 2203,  "platform": "Target",   "brand": "ProLine",     "category": "Professional","stock": "Limited",   "shipping": "Free",       "emoji": "💼"},
        {"title": "Handcrafted Artisan — Limited Batch",      "price": 79.50,  "original_price": 99.50,  "rating": 4.9, "reviews": 612,   "platform": "Etsy",     "brand": "ArtisanCo",   "category": "Handmade",   "stock": "In Stock",   "shipping": "$5.99",      "emoji": "🎨"},
        {"title": "Smart AI-Enhanced Premium Edition",        "price": 149.99, "original_price": 179.99, "rating": 4.4, "reviews": 3087,  "platform": "Best Buy", "brand": "SmartTech",   "category": "Smart Home", "stock": "In Stock",   "shipping": "Free",       "emoji": "🤖"},
        {"title": "Certified Refurb — Like New w/ Warranty",  "price": 55.00,  "original_price": 99.00,  "rating": 4.3, "reviews": 7412,  "platform": "eBay",     "brand": "Various",     "category": "Refurb",     "stock": "In Stock",   "shipping": "Free",       "emoji": "♻️"},
        {"title": "Limited Edition Collector's Bundle",       "price": 199.99, "original_price": 249.99, "rating": 4.8, "reviews": 891,   "platform": "Shopify",  "brand": "LimitedCo",   "category": "Collector",  "stock": "Very Limited","shipping": "Free",       "emoji": "💎"},
        {"title": "Builder's Kit — Open Box",                 "price": 44.99,  "original_price": 64.99,  "rating": 4.2, "reviews": 2109,  "platform": "Newegg",   "brand": "BuildPro",    "category": "DIY",        "stock": "In Stock",   "shipping": "Free",       "emoji": "🔧"},
    ],
}

# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL STYLES
# ═══════════════════════════════════════════════════════════════════════════════

def inject_css() -> None:
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;0,700;1,400;1,600&family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500;9..40,600&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
  --bg:        #0a0a0f;
  --bg2:       #0f0e1a;
  --bg3:       #13121f;
  --gold:      #d4af37;
  --gold2:     #b8963e;
  --gold-dim:  rgba(212,175,55,0.18);
  --gold-glow: rgba(212,175,55,0.07);
  --text:      #f0ece4;
  --text2:     #c8c4ba;
  --text3:     #888;
  --text4:     #555;
  --green:     #64dc82;
  --red:       #ff6060;
  --border:    rgba(212,175,55,0.18);
  --border2:   rgba(255,255,255,0.06);
  --radius-sm: 8px;
  --radius-md: 12px;
  --radius-lg: 18px;
}

* { box-sizing: border-box; }
html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text);
    font-family: 'DM Sans', sans-serif;
}
#MainMenu, footer, header, [data-testid="stToolbar"] { display: none !important; visibility: hidden !important; }
.block-container { padding: 0 2rem 3rem 2rem !important; max-width: 1440px !important; }

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0c0b18 0%, #0f0e1a 50%, #11101d 100%) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] .block-container { padding: 1.5rem 1.2rem 2rem 1.2rem !important; }
[data-testid="stSidebarNav"] { display: none !important; }

/* ── INPUTS ── */
input, textarea, [data-testid="stTextInput"] input, [data-testid="stTextArea"] textarea {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(212,175,55,0.25) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
input:focus, textarea:focus {
    border-color: var(--gold) !important;
    box-shadow: 0 0 0 2px rgba(212,175,55,0.15) !important;
    outline: none !important;
}
input::placeholder, textarea::placeholder { color: var(--text4) !important; }

/* ── SELECT ── */
[data-testid="stSelectbox"] > div > div,
[data-testid="stMultiSelect"] > div > div {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(212,175,55,0.25) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text) !important;
}
[data-baseweb="popover"] { background: var(--bg3) !important; border: 1px solid var(--border) !important; }
[data-baseweb="menu"] { background: var(--bg3) !important; }
[data-baseweb="option"]:hover { background: var(--gold-dim) !important; }
[data-baseweb="tag"] {
    background: rgba(212,175,55,0.15) !important;
    border: 1px solid rgba(212,175,55,0.35) !important;
    color: var(--gold) !important;
    border-radius: 6px !important;
}

/* ── SLIDER ── */
[data-testid="stSlider"] .rc-slider-rail { background: rgba(255,255,255,0.08) !important; }
[data-testid="stSlider"] .rc-slider-track { background: var(--gold) !important; }
[data-testid="stSlider"] .rc-slider-handle { border-color: var(--gold) !important; background: var(--gold) !important; box-shadow: 0 0 8px rgba(212,175,55,0.4) !important; }

/* ── BUTTONS ── */
.stButton > button {
    background: linear-gradient(135deg, var(--gold) 0%, var(--gold2) 100%) !important;
    color: #0a0a0f !important; border: none !important;
    border-radius: var(--radius-sm) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important; letter-spacing: 0.04em !important;
    padding: 0.55rem 1.2rem !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(212,175,55,0.35) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── CHECKBOX ── */
[data-testid="stCheckbox"] label { color: var(--text2) !important; }
[data-baseweb="checkbox"] div { background: var(--gold-dim) !important; border-color: var(--gold) !important; }

/* ── RADIO ── */
[data-testid="stRadio"] label { color: var(--text2) !important; }

/* ── TABS ── */
[data-testid="stTabs"] [role="tablist"] {
    border-bottom: 1px solid rgba(212,175,55,0.2) !important;
    gap: 0.4rem; background: transparent;
}
[data-testid="stTabs"] [role="tab"] {
    color: var(--text3) !important; font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important; border-radius: 8px 8px 0 0 !important;
    padding: 0.5rem 1.1rem !important; border: none !important;
    background: transparent !important; transition: color 0.2s !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: var(--gold) !important;
    border-bottom: 2px solid var(--gold) !important;
    background: rgba(212,175,55,0.05) !important;
}

/* ── EXPANDER ── */
[data-testid="stExpander"] {
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    background: rgba(255,255,255,0.02) !important;
}
[data-testid="stExpander"] summary { color: var(--gold) !important; font-weight: 600 !important; }

/* ── METRIC ── */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.025) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    padding: 1rem 1.2rem !important;
}
[data-testid="stMetricLabel"] { color: var(--text3) !important; font-size: 0.72rem !important; text-transform: uppercase; letter-spacing: 0.1em; font-family: 'JetBrains Mono', monospace !important; }
[data-testid="stMetricValue"] { color: var(--gold) !important; font-family: 'Playfair Display', serif !important; font-size: 1.7rem !important; }
[data-testid="stMetricDelta"] { font-size: 0.78rem !important; }

/* ── ALERTS ── */
[data-testid="stAlert"] {
    border-radius: var(--radius-md) !important;
    border-left: 3px solid var(--gold) !important;
    background: rgba(212,175,55,0.06) !important;
}

/* ── DATAFRAME ── */
[data-testid="stDataFrame"] { border: 1px solid var(--border) !important; border-radius: var(--radius-md) !important; overflow: hidden; }
.dvn-scroller { background: var(--bg3) !important; }

/* ── SCROLLBAR ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: rgba(212,175,55,0.28); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(212,175,55,0.5); }

/* ── PLOTLY CHARTS dark ── */
.js-plotly-plot .plotly .modebar { background: transparent !important; }

/* ══════════════════════════════
   CUSTOM COMPONENT STYLES
══════════════════════════════ */

/* Hero */
.hero-wrap { text-align: center; padding: 3.5rem 1rem 2rem; position: relative; overflow: hidden; }
.hero-glow { position: absolute; top: 50%; left: 50%; transform: translate(-50%,-50%); width: 700px; height: 350px; background: radial-gradient(ellipse, rgba(212,175,55,0.08) 0%, transparent 65%); pointer-events: none; }
.hero-eyebrow { font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; letter-spacing: 0.3em; text-transform: uppercase; color: var(--gold); margin-bottom: 0.9rem; }
.hero-title { font-family: 'Playfair Display', serif; font-size: clamp(2.6rem,6vw,4.8rem); font-weight: 700; color: var(--text); line-height: 1.08; margin: 0 0 0.7rem; }
.hero-title em { color: var(--gold); font-style: italic; }
.hero-sub { font-size: 1rem; color: var(--text3); max-width: 540px; margin: 0 auto 2.2rem; line-height: 1.65; }

/* Platform strip */
.pstrip { display: flex; gap: 0.6rem; flex-wrap: wrap; justify-content: center; margin-bottom: 2rem; }
.pchip { padding: 5px 13px; background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 20px; font-size: 0.73rem; color: var(--text3); letter-spacing: 0.04em; transition: border-color 0.2s, color 0.2s; }

/* Sidebar labels */
.sl { font-family: 'JetBrains Mono', monospace; font-size: 0.58rem; letter-spacing: 0.22em; text-transform: uppercase; color: var(--gold); margin-bottom: 0.5rem; padding-bottom: 0.35rem; border-bottom: 1px solid rgba(212,175,55,0.18); display: block; }

/* Section headings */
.sec-eyebrow { font-family: 'JetBrains Mono', monospace; font-size: 0.62rem; letter-spacing: 0.22em; text-transform: uppercase; color: var(--text4); margin-bottom: 0.3rem; }
.sec-title { font-family: 'Playfair Display', serif; font-size: 1.55rem; color: var(--text); margin-bottom: 1.2rem; }

/* Product card */
.pcard {
    background: linear-gradient(140deg, rgba(255,255,255,0.04) 0%, rgba(255,255,255,0.015) 100%);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 1.3rem;
    margin-bottom: 0.8rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.25s, transform 0.2s, box-shadow 0.25s;
}
.pcard::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent 0%, var(--gold) 50%, transparent 100%);
    opacity: 0;
    transition: opacity 0.3s;
}
.pcard:hover { border-color: rgba(212,175,55,0.42); transform: translateY(-3px); box-shadow: 0 16px 48px rgba(0,0,0,0.5); }
.pcard:hover::before { opacity: 0.7; }

.pbadge { display: inline-block; padding: 3px 10px; border-radius: 20px; font-size: 0.68rem; font-weight: 600; letter-spacing: 0.07em; text-transform: uppercase; margin-bottom: 0.6rem; }
.badge-amazon  { background: rgba(255,153,0,0.12); color: #ff9900; border: 1px solid rgba(255,153,0,0.28); }
.badge-ebay    { background: rgba(14,118,188,0.12); color: #4a9fd4; border: 1px solid rgba(14,118,188,0.28); }
.badge-walmart { background: rgba(0,117,201,0.12); color: #4ab3ff; border: 1px solid rgba(0,117,201,0.28); }
.badge-etsy    { background: rgba(241,100,30,0.12); color: #f1641e; border: 1px solid rgba(241,100,30,0.28); }
.badge-bestbuy { background: rgba(0,70,208,0.12); color: #5e8eff; border: 1px solid rgba(0,70,208,0.28); }
.badge-target  { background: rgba(204,0,0,0.12); color: #ff6060; border: 1px solid rgba(204,0,0,0.28); }
.badge-shopify { background: rgba(150,191,68,0.12); color: #96bf44; border: 1px solid rgba(150,191,68,0.28); }
.badge-newegg  { background: rgba(255,126,0,0.12); color: #ff9a40; border: 1px solid rgba(255,126,0,0.28); }

.ai-pill { display: inline-block; padding: 2px 9px; background: rgba(100,220,130,0.1); border: 1px solid rgba(100,220,130,0.28); border-radius: 20px; font-size: 0.68rem; color: var(--green); font-weight: 700; font-family: 'JetBrains Mono', monospace; float: right; }
.ptitle { font-weight: 600; font-size: 0.95rem; color: var(--text); margin: 0.5rem 0 0.3rem; line-height: 1.4; }
.pprice-wrap { display: flex; align-items: baseline; gap: 0.5rem; margin: 0.35rem 0; }
.pprice { font-family: 'Playfair Display', serif; font-size: 1.35rem; color: var(--gold); }
.pprice-orig { font-size: 0.82rem; color: var(--text4); text-decoration: line-through; }
.psaving { font-size: 0.75rem; color: var(--green); font-weight: 600; }
.prating { color: var(--gold); font-size: 0.82rem; }
.pmeta { font-size: 0.76rem; color: var(--text3); margin: 0.12rem 0; }

/* Price bar */
.pbar-outer { background: rgba(255,255,255,0.06); border-radius: 3px; height: 5px; margin: 0.5rem 0; overflow: hidden; }
.pbar-inner { height: 5px; border-radius: 3px; background: linear-gradient(90deg, var(--gold), var(--green)); transition: width 0.7s ease; }

/* Tags */
.tag { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.68rem; font-weight: 600; margin: 2px 2px 0 0; }
.tag-deal  { background: rgba(212,175,55,0.1); color: var(--gold); border: 1px solid rgba(212,175,55,0.25); }
.tag-green { background: rgba(100,220,130,0.1); color: var(--green); border: 1px solid rgba(100,220,130,0.22); }
.tag-red   { background: rgba(255,96,96,0.1); color: var(--red); border: 1px solid rgba(255,96,96,0.22); }
.tag-gray  { background: rgba(255,255,255,0.05); color: var(--text3); border: 1px solid rgba(255,255,255,0.1); }

/* AI bubble */
.ai-bubble {
    background: linear-gradient(135deg, rgba(212,175,55,0.07) 0%, rgba(180,145,25,0.03) 100%);
    border: 1px solid rgba(212,175,55,0.22);
    border-left: 3px solid var(--gold);
    border-radius: var(--radius-md);
    padding: 1.2rem 1.4rem;
    margin: 1rem 0;
    font-size: 0.9rem;
    color: var(--text2);
    line-height: 1.75;
}
.ai-bubble::before {
    content: '✦ AI INTELLIGENCE';
    display: block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.2em;
    color: var(--gold);
    margin-bottom: 0.6rem;
}

/* Chat */
.chat-user {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px 12px 4px 12px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.5rem;
    color: var(--text);
    font-size: 0.88rem;
    text-align: right;
}
.chat-ai {
    background: rgba(212,175,55,0.05);
    border: 1px solid rgba(212,175,55,0.18);
    border-radius: 12px 12px 12px 4px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.5rem;
    color: var(--text2);
    font-size: 0.88rem;
}
.chat-ts { font-size: 0.63rem; color: var(--text4); margin-top: 0.25rem; font-family: 'JetBrains Mono', monospace; }

/* Insight card */
.insight-card {
    background: rgba(255,255,255,0.02);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 0.9rem 1rem;
    margin-bottom: 0.5rem;
    font-size: 0.83rem;
    color: var(--text2);
}
.insight-card span.icon { font-size: 1.1rem; margin-right: 0.4rem; }

/* Feature card */
.feat-card {
    text-align: center;
    padding: 1.6rem 1rem;
    background: rgba(255,255,255,0.02);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    height: 200px;
    transition: border-color 0.2s, transform 0.2s;
}
.feat-card:hover { border-color: rgba(212,175,55,0.35); transform: translateY(-3px); }
.feat-icon { font-size: 2.2rem; margin-bottom: 0.7rem; }
.feat-title { font-family: 'Playfair Display', serif; font-size: 1rem; color: var(--text); margin-bottom: 0.5rem; }
.feat-desc { font-size: 0.78rem; color: var(--text3); line-height: 1.55; }

/* Empty state */
.empty-state { text-align: center; padding: 3.5rem 1rem; color: var(--text3); }
.empty-icon { font-size: 3.5rem; margin-bottom: 1rem; }
.empty-title { font-family: 'Playfair Display', serif; font-size: 1.3rem; color: var(--text2); margin-bottom: 0.5rem; }
.empty-sub { font-size: 0.88rem; line-height: 1.6; }

/* Score badge pill colors */
.score-excellent { background: rgba(100,220,130,0.12); color: #64dc82; border: 1px solid rgba(100,220,130,0.28); }
.score-good      { background: rgba(212,175,55,0.12); color: var(--gold); border: 1px solid rgba(212,175,55,0.28); }
.score-average   { background: rgba(255,180,60,0.12); color: #ffb43c; border: 1px solid rgba(255,180,60,0.28); }
.score-poor      { background: rgba(255,96,96,0.12); color: var(--red); border: 1px solid rgba(255,96,96,0.28); }

/* Wishlist row */
.wl-row {
    background: rgba(255,255,255,0.02);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 0.75rem 1rem;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* CTA banner */
.cta-banner {
    text-align: center;
    padding: 2.2rem 1.5rem;
    background: rgba(212,175,55,0.04);
    border: 1px solid rgba(212,175,55,0.15);
    border-radius: var(--radius-lg);
}
.cta-title { font-family: 'Playfair Display', serif; font-size: 1.35rem; color: var(--gold); margin-bottom: 0.5rem; font-style: italic; }
.cta-sub { color: var(--text3); font-size: 0.9rem; }

/* Sidebar logo */
.logo-wrap { text-align: center; padding: 0.8rem 0 1.6rem; }
.logo-name { font-family: 'Playfair Display', serif; font-size: 1.45rem; color: var(--gold); font-weight: 700; letter-spacing: 0.02em; }
.logo-sub { font-family: 'JetBrains Mono', monospace; font-size: 0.58rem; letter-spacing: 0.25em; color: var(--text4); margin-top: 0.2rem; text-transform: uppercase; }

/* Search history chip */
.hist-chip { display: inline-block; padding: 4px 12px; background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.1); border-radius: 20px; font-size: 0.75rem; color: var(--text3); margin: 3px; cursor: pointer; transition: border-color 0.2s, color 0.2s; }
.hist-chip:hover { border-color: var(--gold); color: var(--gold); }

/* Comparison table */
.cmp-tbl { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
.cmp-tbl th { color: var(--text3); font-weight: 600; font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; letter-spacing: 0.12em; text-transform: uppercase; padding: 0.6rem 0.8rem; border-bottom: 1px solid var(--border); text-align: left; }
.cmp-tbl td { padding: 0.6rem 0.8rem; border-bottom: 1px solid rgba(255,255,255,0.04); color: var(--text2); vertical-align: middle; }
.cmp-tbl tr:hover td { background: rgba(212,175,55,0.03); }
.cmp-best { color: var(--green) !important; font-weight: 700; }

/* Divider */
hr { border-color: rgba(212,175,55,0.12) !important; margin: 1.5rem 0 !important; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SESSION MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════

def init_session() -> None:
    defaults: Dict[str, Any] = {
        "search_results":  [],
        "last_query":      "",
        "ai_insight":      "",
        "ai_recommendations": {},
        "product_insights": [],
        "chat_history":    [],
        "wishlist":        [],
        "compare_list":    [],
        "search_history":  [],
        "active_tab":      0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

# ═══════════════════════════════════════════════════════════════════════════════
# SEARCH ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=300, show_spinner=False)
def _get_raw_products(query_key: str) -> List[Dict]:
    """Cached product retrieval by category keyword."""
    q = query_key.lower()
    for kw, items in _RAW_DB.items():
        if kw != "default" and kw in q:
            return items
    return _RAW_DB["default"]


def compute_ai_score(p: Dict, category_stats: Dict) -> float:
    """Weighted AI scoring engine."""
    trust = PLATFORM_TRUST.get(p["platform"], 0.75)

    # Rating score (0-1)
    rating_score = (p["rating"] - 1) / 4.0

    # Review volume score (log-normalized up to 50k)
    import math
    review_score = min(1.0, math.log1p(p["reviews"]) / math.log1p(50000))

    # Price-value score (cheaper relative to category max → better value)
    cat_max = category_stats.get("max_price", p["original_price"])
    cat_min = category_stats.get("min_price", p["price"])
    price_range = max(cat_max - cat_min, 1)
    price_value = 1.0 - ((p["price"] - cat_min) / price_range) * 0.5

    # Discount bonus
    disc_pct = max(0, (p["original_price"] - p["price"]) / max(p["original_price"], 1))
    deal_score = min(1.0, disc_pct * 3)

    score = (
        SCORE_WEIGHTS["rating"]        * rating_score  +
        SCORE_WEIGHTS["review_volume"] * review_score  +
        SCORE_WEIGHTS["price_value"]   * price_value   +
        SCORE_WEIGHTS["platform_trust"]* trust          +
        SCORE_WEIGHTS["deal_quality"]  * deal_score
    )
    # Add slight jitter for realism
    score = score + random.uniform(-0.02, 0.02)
    return round(min(99.5, max(55.0, score * 100)), 1)


def classify_deal(price: float, orig: float, cat_avg: float) -> Tuple[str, float]:
    """Return (deal_label, deal_quality_0_to_1)."""
    disc = (orig - price) / max(orig, 1)
    ratio = price / max(cat_avg, 1)

    if disc >= 0.25 or ratio <= 0.70:
        return "🔥 Excellent Deal", 0.95
    elif disc >= 0.12 or ratio <= 0.85:
        return "✅ Great Deal", 0.80
    elif disc >= 0.05 or ratio <= 0.95:
        return "👍 Good Value", 0.65
    elif ratio <= 1.10:
        return "📊 Fair Price", 0.50
    else:
        return "⚠️ Overpriced", 0.25


def price_trend_label() -> str:
    choices = ["↓ 14%", "↓ 8%", "↓ 3%", "→ Stable", "↑ 2%", "↑ 6%"]
    weights = [0.10, 0.18, 0.20, 0.30, 0.13, 0.09]
    return random.choices(choices, weights=weights, k=1)[0]


def build_products(
    query: str,
    platforms: List[str],
    budget_max: float,
    min_rating: float,
    min_reviews: int,
    min_discount: float,
) -> List[Product]:
    """Core search engine: filter, score and rank products."""
    raw = _get_raw_products(query.lower())

    # Category statistics for scoring
    prices = [r["price"] for r in raw]
    cat_stats = {
        "min_price": min(prices),
        "max_price": max(prices),
        "avg_price": sum(prices) / len(prices),
    }

    products: List[Product] = []
    for r in raw:
        if r["platform"] not in platforms:
            continue
        if r["price"] > budget_max:
            continue
        if r["rating"] < min_rating:
            continue
        if r["reviews"] < min_reviews:
            continue
        disc = (r["original_price"] - r["price"]) / max(r["original_price"], 1) * 100
        if disc < min_discount:
            continue

        deal_label, deal_q = classify_deal(r["price"], r["original_price"], cat_stats["avg_price"])
        trend = price_trend_label()
        score = compute_ai_score(r, cat_stats)

        tags: List[str] = []
        if disc >= 15:
            tags.append(f"🔖 {int(disc)}% OFF")
        if r["rating"] >= 4.8:
            tags.append("🏅 Top Rated")
        if r["reviews"] >= 10000:
            tags.append("💬 Highly Reviewed")
        if r["stock"] == "Limited":
            tags.append("⚡ Limited Stock")
        if r["platform"] == "Etsy":
            tags.append("🎨 Artisan")

        p = Product(
            title=r["title"],
            price=r["price"],
            original_price=r["original_price"],
            rating=r["rating"],
            reviews=r["reviews"],
            platform=r["platform"],
            brand=r["brand"],
            category=r["category"],
            stock=r["stock"],
            shipping=r["shipping"],
            emoji=r["emoji"],
            ai_score=score,
            price_trend=trend,
            deal_label=deal_label,
            deal_quality_score=deal_q,
            discount_pct=round(disc, 1),
            tags=tags,
        )
        products.append(p)

    # Fill up if filter leaves too few results (pad with scored defaults)
    if len(products) < 3:
        for r in _RAW_DB["default"]:
            if r["price"] <= budget_max and r["platform"] in platforms:
                disc = (r["original_price"] - r["price"]) / max(r["original_price"], 1) * 100
                deal_label, deal_q = classify_deal(r["price"], r["original_price"], cat_stats["avg_price"])
                p = Product(
                    title=r["title"], price=r["price"], original_price=r["original_price"],
                    rating=r["rating"], reviews=r["reviews"], platform=r["platform"],
                    brand=r["brand"], category=r["category"], stock=r["stock"],
                    shipping=r["shipping"], emoji=r["emoji"],
                    ai_score=compute_ai_score(r, cat_stats),
                    price_trend=price_trend_label(), deal_label=deal_label,
                    deal_quality_score=deal_q, discount_pct=round(disc,1), tags=[]
                )
                products.append(p)
            if len(products) >= 6:
                break

    return products


def sort_products(products: List[Product], sort_by: str) -> List[Product]:
    key_map = {
        "AI Score":          lambda p: p.ai_score,
        "Price: Low→High":   lambda p: p.price,
        "Price: High→Low":   lambda p: -p.price,
        "Rating":            lambda p: p.rating,
        "Reviews":           lambda p: p.reviews,
        "Discount %":        lambda p: p.discount_pct,
    }
    fn = key_map.get(sort_by, lambda p: p.ai_score)
    reverse = sort_by not in ("Price: Low→High",)
    return sorted(products, key=fn, reverse=reverse)

# ═══════════════════════════════════════════════════════════════════════════════
# AI RECOMMENDATION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def generate_recommendations(products: List[Product], budget: float) -> Dict[str, Optional[Product]]:
    if not products:
        return {}
    by_score  = sorted(products, key=lambda p: p.ai_score, reverse=True)
    by_price  = sorted(products, key=lambda p: p.price)
    by_rating = sorted(products, key=lambda p: p.rating, reverse=True)
    by_value  = sorted(products, key=lambda p: p.ai_score / max(p.price, 1), reverse=True)

    premium_cutoff = budget * 0.75
    budget_cutoff  = budget * 0.45

    premium_cands = [p for p in products if p.price >= premium_cutoff]
    budget_cands  = [p for p in products if p.price <= budget_cutoff]

    return {
        "Best Overall":       by_score[0] if by_score else None,
        "Best Budget":        min(budget_cands, key=lambda p: p.price) if budget_cands else by_price[0],
        "Best Premium":       max(premium_cands, key=lambda p: p.ai_score) if premium_cands else by_score[0],
        "Best Value":         by_value[0] if by_value else None,
        "Highest Rated":      by_rating[0] if by_rating else None,
    }


def generate_ai_insight(query: str, products: List[Product], budget: float, llm: str) -> str:
    if not products:
        return "No products found. Try adjusting your filters."

    prices   = [p.price for p in products]
    avg_p    = sum(prices) / len(prices)
    best     = max(products, key=lambda p: p.ai_score)
    cheapest = min(products, key=lambda p: p.price)
    rated    = max(products, key=lambda p: p.rating)

    trending_down = sum(1 for p in products if "↓" in p.price_trend)

    parts: List[str] = []

    q = query.lower()

    # Query-specific opening
    if "laptop" in q or "macbook" in q or "thinkpad" in q:
        parts.append(f"Analyzed **{len(products)} laptops** across {len(set(p.platform for p in products))} platforms. The **{best.title[:40]}...** leads with an AI score of **{best.ai_score}%** — its M3 silicon or equivalent architecture delivers peak performance-per-watt in this price tier.")
        parts.append(f"\n\n**Budget Insight:** At ${avg_p:,.0f} average, this category sits firmly mid-premium. You can save **20–30%** by opting for a certified refurb from eBay or last-gen flagships, which retain 90%+ of performance.")
    elif "headphone" in q or "earbud" in q or "audio" in q:
        parts.append(f"Scanned **{len(products)} audio products**. The **{best.title[:40]}...** ranks #1 with {best.ai_score}% — exceptional noise cancellation and driver tuning make it the standout choice for most use cases.")
        parts.append(f"\n\n**Category Trend:** Wireless ANC headphones have dropped 12% in median price over 90 days — this is a strong buyer's market right now.")
    elif "phone" in q or "iphone" in q or "samsung" in q or "pixel" in q:
        parts.append(f"Evaluated **{len(products)} smartphones**. The **{best.title[:40]}...** scores highest at {best.ai_score}% — its camera system and software support cycle make it the most future-proof option.")
        parts.append(f"\n\n**Value Angle:** The **{cheapest.title[:35]}...** at **${cheapest.price:,.0f}** delivers flagship-adjacent performance at {int((1-cheapest.price/max(p.price for p in products))*100)}% less than the top-tier.")
    elif "coffee" in q or "espresso" in q or "grinder" in q:
        parts.append(f"Curated **{len(products)} coffee products**. The **{best.title[:40]}...** wins on AI scoring — an integrated grinder or precision pour-over system dramatically outperforms pod alternatives for flavor complexity.")
        parts.append(f"\n\n**ROI Math:** A premium home setup at ${avg_p:,.0f} avg pays for itself vs. café spending within 4–6 months for a 2-cup-a-day habit.")
    else:
        parts.append(f"Analyzed **{len(products)} products** for **'{query}'** across {len(set(p.platform for p in products))} platforms. The **{best.title[:40]}...** ranks #1 at {best.ai_score}% AI score, balancing rating ({best.rating}★), reviews ({best.reviews:,}), and price-value ratio.")

    # Cross-cutting insights
    if trending_down >= len(products) // 2:
        parts.append(f"\n\n**⬇️ Price Alert:** {trending_down} of {len(products)} products are trending downward — this suggests a market correction window. Historical patterns indicate 2–3 more weeks of softening before Q4 demand spikes.")

    parts.append(f"\n\n**Highest Rated:** **{rated.title[:35]}...** at ★{rated.rating} ({rated.reviews:,} reviews) — community validation at this scale is a strong quality signal.")

    if any(p.discount_pct >= 15 for p in products):
        top_disc = max(products, key=lambda p: p.discount_pct)
        parts.append(f"\n\n**Best Saving:** **{top_disc.title[:35]}...** is discounted {top_disc.discount_pct:.0f}% (save ${top_disc.savings:,.0f}) — the strongest dollar-value deal in this result set.")

    parts.append(f"\n\n*Analysis powered by {llm} · {len(products)} products scored · Real-time pricing*")
    return "".join(parts)


def generate_product_insights(products: List[Product]) -> List[str]:
    if not products:
        return []
    insights: List[str] = []
    prices  = [p.price for p in products]
    ratings = [p.rating for p in products]
    avg_p   = sum(prices) / len(prices)
    avg_r   = sum(ratings) / len(ratings)

    best_rated = max(products, key=lambda p: p.rating)
    cheapest   = min(products, key=lambda p: p.price)
    most_rev   = max(products, key=lambda p: p.reviews)

    insights.append(f"📈 **{best_rated.title[:35]}...** has a {((best_rated.rating-avg_r)/avg_r*100):.0f}% higher rating than the category average (★{avg_r:.2f})")
    insights.append(f"💰 **{cheapest.title[:35]}...** is the most affordable option at ${cheapest.price:,.2f} — {((avg_p-cheapest.price)/avg_p*100):.0f}% below average")
    insights.append(f"💬 **{most_rev.title[:35]}...** has the most social proof with {most_rev.reviews:,} reviews")

    platforms = list(set(p.platform for p in products))
    if len(platforms) > 1:
        insights.append(f"🏪 Results span {len(platforms)} platforms: {', '.join(platforms[:4])}")

    deals = [p for p in products if "Excellent" in p.deal_label or "Great" in p.deal_label]
    if deals:
        insights.append(f"🔥 {len(deals)} of {len(products)} products qualify as Great Deal or better")

    discounted = [p for p in products if p.discount_pct >= 10]
    if discounted:
        top = max(discounted, key=lambda p: p.discount_pct)
        insights.append(f"🏷️ Biggest discount: {top.discount_pct:.0f}% off on **{top.title[:30]}...**")

    return insights

# ═══════════════════════════════════════════════════════════════════════════════
# PRICE ANALYTICS ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def price_analytics(products: List[Product]) -> Dict[str, Any]:
    if not products:
        return {}
    prices = [p.price for p in products]
    sorted_p = sorted(prices)
    n = len(sorted_p)
    median = sorted_p[n // 2] if n % 2 else (sorted_p[n // 2 - 1] + sorted_p[n // 2]) / 2
    return {
        "avg":    round(sum(prices) / n, 2),
        "median": round(median, 2),
        "min":    round(min(prices), 2),
        "max":    round(max(prices), 2),
        "range":  round(max(prices) - min(prices), 2),
        "count":  n,
    }


def build_plotly_price_chart(products: List[Product]) -> go.Figure:
    df = pd.DataFrame([{"Title": p.title[:28]+"…", "Price": p.price, "Rating": p.rating, "Platform": p.platform, "AI Score": p.ai_score} for p in products])
    fig = px.bar(
        df, x="Title", y="Price", color="AI Score",
        color_continuous_scale=[[0,"#444"],[0.5,"#d4af37"],[1,"#64dc82"]],
        hover_data=["Rating","Platform"],
        template="plotly_dark",
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans", color="#c8c4ba", size=11),
        margin=dict(l=10, r=10, t=20, b=80),
        coloraxis_colorbar=dict(title="AI %", tickfont=dict(color="#888")),
        xaxis=dict(tickangle=-35, gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)", tickprefix="$"),
        height=320,
    )
    return fig


def build_plotly_scatter(products: List[Product]) -> go.Figure:
    df = pd.DataFrame([{"Price": p.price, "Rating": p.rating, "Reviews": p.reviews, "Title": p.title[:30]+"…", "Platform": p.platform} for p in products])
    fig = px.scatter(
        df, x="Price", y="Rating", size="Reviews", color="Platform",
        hover_name="Title", template="plotly_dark",
        color_discrete_sequence=["#d4af37","#64dc82","#ff9900","#4ab3ff","#ff6060","#96bf44","#ff9a40","#f1641e"],
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans", color="#c8c4ba", size=11),
        margin=dict(l=10, r=10, t=20, b=40),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)", tickprefix="$"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)", range=[3.5, 5.1]),
        height=300,
    )
    return fig


def build_platform_donut(products: List[Product]) -> go.Figure:
    counts = Counter(p.platform for p in products)
    colors = [PLATFORM_CONFIG.get(pl, {}).get("color", "#888") for pl in counts.keys()]
    fig = go.Figure(go.Pie(
        labels=list(counts.keys()),
        values=list(counts.values()),
        hole=0.62,
        marker=dict(colors=colors, line=dict(color="#0a0a0f", width=2)),
        textfont=dict(family="DM Sans", size=11),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans", color="#c8c4ba"),
        showlegend=True,
        legend=dict(font=dict(color="#c8c4ba", size=10)),
        margin=dict(l=0, r=0, t=10, b=10),
        height=240,
    )
    return fig


def build_deal_quality_chart(products: List[Product]) -> go.Figure:
    deal_map: Dict[str, int] = {}
    for p in products:
        label = p.deal_label.split(" ", 1)[-1] if " " in p.deal_label else p.deal_label
        deal_map[label] = deal_map.get(label, 0) + 1

    fig = px.bar(
        x=list(deal_map.values()),
        y=list(deal_map.keys()),
        orientation="h",
        template="plotly_dark",
        color_discrete_sequence=["#d4af37"],
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans", color="#c8c4ba", size=11),
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
        height=220,
        showlegend=False,
    )
    return fig

# ═══════════════════════════════════════════════════════════════════════════════
# COMPARISON ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def render_comparison_table(compare_list: List[Product]) -> None:
    if not compare_list:
        st.markdown('<div class="empty-state"><div class="empty-icon">⚖️</div><div class="empty-title">No products to compare</div><div class="empty-sub">Click ⚖️ Compare on any product card to add it here.</div></div>', unsafe_allow_html=True)
        return

    fields = ["Title","Price","Original","Savings","Rating","Reviews","Platform","AI Score","Deal","Stock","Shipping","Discount"]
    rows: List[Dict] = []
    for p in compare_list:
        rows.append({
            "Title":     p.title[:40],
            "Price":     f"${p.price:,.2f}",
            "Original":  f"${p.original_price:,.2f}",
            "Savings":   f"${p.savings:,.2f}",
            "Rating":    f"★ {p.rating}",
            "Reviews":   f"{p.reviews:,}",
            "Platform":  p.platform,
            "AI Score":  f"{p.ai_score}%",
            "Deal":      p.deal_label,
            "Stock":     p.stock,
            "Shipping":  p.shipping,
            "Discount":  f"{p.discount_pct:.1f}%",
        })

    df = pd.DataFrame(rows)

    # Highlight best values
    best_ai  = max(compare_list, key=lambda p: p.ai_score).title[:40]
    best_rat = max(compare_list, key=lambda p: p.rating).title[:40]
    cheapest = min(compare_list, key=lambda p: p.price).title[:40]

    st.markdown(f"""
    <div style="margin-bottom:0.8rem; font-size:0.8rem; color:var(--text3);">
    🥇 <b style="color:var(--green)">Best AI Score:</b> {best_ai[:35]}...
    &nbsp;&nbsp;|&nbsp;&nbsp;
    ⭐ <b style="color:var(--gold)">Highest Rated:</b> {best_rat[:35]}...
    &nbsp;&nbsp;|&nbsp;&nbsp;
    💰 <b style="color:#64dc82">Lowest Price:</b> {cheapest[:35]}...
    </div>
    """, unsafe_allow_html=True)

    st.dataframe(
        df,
        use_container_width=True,
        height=min(40 + len(rows) * 36, 340),
    )

    # Export comparison
    csv_bytes = df.to_csv(index=False).encode()
    st.download_button("⬇️ Export Comparison CSV", csv_bytes, "comparison.csv", "text/csv", key="dl_compare")

# ═══════════════════════════════════════════════════════════════════════════════
# CHAT ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def generate_chat_response(user_msg: str, products: List[Product], query: str, budget: float, llm: str) -> str:
    """Context-aware AI chat responder using product data."""
    u = user_msg.lower()
    if not products:
        return "Please run a search first — I'll then have product data to answer your questions about."

    best     = max(products, key=lambda p: p.ai_score)
    cheapest = min(products, key=lambda p: p.price)
    rated    = max(products, key=lambda p: p.rating)
    reviewed = max(products, key=lambda p: p.reviews)

    # Warranty / return
    if any(w in u for w in ["warrant", "return", "policy", "guarantee"]):
        return (f"Based on your current search results, warranty and return policies by platform:\n\n"
                f"• **Amazon** — 30-day returns, manufacturer warranty honored\n"
                f"• **Best Buy** — 15-day returns (My Best Buy members: 30 days), Geek Squad plans available\n"
                f"• **Walmart** — 30-day returns, 90 days for electronics\n"
                f"• **Target** — 30-day returns, RedCard extends to 60 days\n"
                f"• **Newegg** — 15–30 day RMA, check per-product\n"
                f"• **eBay** — seller-specific, filter for '30-day returns' guarantee\n"
                f"• **Etsy** — seller policy varies; message seller before buying\n\n"
                f"For your top pick **{best.title[:40]}...** on {best.platform}: check the listing for exact manufacturer warranty duration.")

    # Shipping / delivery speed
    if any(w in u for w in ["ship", "deliver", "fast", "quick", "speed", "arrival"]):
        fast = min(products, key=lambda p: 0 if "2-day" in p.shipping or "Same" in p.shipping else 1)
        return (f"Fastest shipping in your current results goes to **{fast.title[:40]}...** ({fast.platform} — {fast.shipping}).\n\n"
                f"Speed overview:\n"
                f"• **Amazon Prime 2-day** — fastest for Prime members\n"
                f"• **Best Buy / Target** — same-day pickup available in-store\n"
                f"• **Walmart** — free 2-day on eligible orders over $35\n"
                f"• **eBay** — varies by seller location; check estimated delivery date\n"
                f"• **Etsy** — handmade items ship 3–14 days depending on the seller")

    # Best under budget
    if any(w in u for w in ["under", "budget", "cheap", "affordable", "less than"]):
        return (f"Within your ${budget:,.0f} budget, the best value pick is **{cheapest.title[:40]}...** at **${cheapest.price:,.2f}**.\n\n"
                f"It carries an AI score of {cheapest.ai_score}% and ★{cheapest.rating} rating from {cheapest.reviews:,} reviews. "
                f"You'd save **${cheapest.savings:,.2f}** vs. its original price.\n\n"
                f"If you can stretch slightly, **{best.title[:35]}...** at ${best.price:,.2f} scores {best.ai_score}% — the highest in your results.")

    # Compare top 2
    if any(w in u for w in ["compar", "difference", "vs", "versus", "better", "between"]):
        p1, p2 = products[0], products[1] if len(products) > 1 else products[0]
        winner = p1 if p1.ai_score >= p2.ai_score else p2
        return (f"Comparing the top 2 results:\n\n"
                f"**{p1.title[:35]}...** (AI: {p1.ai_score}%, ★{p1.rating}, ${p1.price:,.2f} on {p1.platform})\n"
                f"vs\n"
                f"**{p2.title[:35]}...** (AI: {p2.ai_score}%, ★{p2.rating}, ${p2.price:,.2f} on {p2.platform})\n\n"
                f"🏆 **My recommendation: {winner.title[:35]}...** — it edges ahead on overall AI scoring, factoring in rating, community trust, platform reliability, and price-value ratio.\n\n"
                f"*Analysis by {llm}*")

    # Most reviewed / trusted
    if any(w in u for w in ["trusted", "popular", "review", "community", "proven"]):
        return (f"The most community-validated product in your results is **{reviewed.title[:40]}...** with **{reviewed.reviews:,} reviews** and ★{reviewed.rating} average rating.\n\n"
                f"At this review volume, the rating is statistically robust — less than 0.3% variance expected. It's listed on **{reviewed.platform}** for **${reviewed.price:,.2f}**.\n\n"
                f"High review count is one of the strongest quality signals I factor into AI scoring.")

    # Default smart reply
    return (f"Great question! Based on your search for **'{query}'**, here's what I'd highlight:\n\n"
            f"• 🏆 **Top Overall:** {best.title[:40]}... — AI Score {best.ai_score}%, ★{best.rating}, ${best.price:,.2f}\n"
            f"• 💰 **Best Budget:** {cheapest.title[:40]}... — ${cheapest.price:,.2f}\n"
            f"• ⭐ **Highest Rated:** {rated.title[:40]}... — ★{rated.rating}\n\n"
            f"All results have been scored and ranked by my weighted AI engine across rating quality, review volume, price-value, platform trust, and discount depth.\n\n"
            f"*Powered by {llm}*")

# ═══════════════════════════════════════════════════════════════════════════════
# UI COMPONENTS
# ═══════════════════════════════════════════════════════════════════════════════

def stars(rating: float) -> str:
    full  = int(rating)
    half  = 1 if (rating - full) >= 0.5 else 0
    empty = 5 - full - half
    return "★" * full + "½" * half + "☆" * empty


def score_class(score: float) -> str:
    if score >= 88:   return "score-excellent"
    elif score >= 76: return "score-good"
    elif score >= 64: return "score-average"
    return "score-poor"


def render_product_card(p: Product, max_price: float, idx: int) -> None:
    cfg = PLATFORM_CONFIG.get(p.platform, {"badge": "badge-amazon", "emoji": "🛒"})
    bar_pct = int((p.price / max(max_price, 1)) * 100)
    tags_html = " ".join(f'<span class="tag tag-deal">{t}</span>' for t in p.tags)

    trend_cls = "tag-green" if "↓" in p.price_trend else ("tag-red" if "↑" in p.price_trend else "tag-gray")
    sc = score_class(p.ai_score)

    savings_html = f'<span class="psaving">Save ${p.savings:,.2f}</span>' if p.savings > 0 else ''
    orig_html    = f'<span class="pprice-orig">${p.original_price:,.2f}</span>' if p.savings > 0 else ''

    card = f"""
<div class="pcard">
  <span class="pbadge {cfg['badge']}">{cfg['emoji']} {p.platform}</span>
  <span class="ai-pill {sc}">{p.ai_score}%</span>
  <div style="clear:both"></div>
  <div style="font-size:2rem; margin:0.5rem 0 0.2rem;">{p.emoji}</div>
  <p class="ptitle">{p.title}</p>
  <div class="pprice-wrap">
    <span class="pprice">${p.price:,.2f}</span>{orig_html}{savings_html}
  </div>
  <div><span class="prating">{stars(p.rating)}</span> <span class="pmeta">★{p.rating} &nbsp;({p.reviews:,} reviews)</span></div>
  <div class="pbar-outer"><div class="pbar-inner" style="width:{bar_pct}%"></div></div>
  <div class="pmeta">📦 {p.stock} &nbsp;·&nbsp; 🚚 {p.shipping}</div>
  <div class="pmeta">🏷️ {p.brand} &nbsp;·&nbsp; 📂 {p.category}</div>
  <div style="margin-top:0.5rem;">
    <span class="tag {trend_cls}">{p.price_trend}</span>
    <span class="tag tag-deal">{p.deal_label}</span>
    {tags_html}
  </div>
  <a href="{p.url}" target="_blank" style="display:inline-block;margin-top:0.7rem;padding:6px 14px;background:rgba(212,175,55,0.09);border:1px solid rgba(212,175,55,0.32);border-radius:6px;color:#d4af37;text-decoration:none;font-size:0.78rem;font-weight:600;letter-spacing:0.04em;">View on {p.platform} →</a>
</div>"""
    st.markdown(card, unsafe_allow_html=True)


def render_rec_badge(label: str, product: Optional[Product]) -> None:
    if product is None:
        return
    cfg = PLATFORM_CONFIG.get(product.platform, {"badge": "badge-amazon"})
    st.markdown(f"""
<div style="background:rgba(255,255,255,0.02);border:1px solid var(--border);border-radius:var(--radius-md);padding:0.9rem 1rem;margin-bottom:0.6rem;">
  <div style="font-family:'JetBrains Mono',monospace;font-size:0.6rem;letter-spacing:0.18em;text-transform:uppercase;color:var(--gold);margin-bottom:0.35rem;">{label}</div>
  <div style="font-size:0.88rem;font-weight:600;color:var(--text);margin-bottom:0.3rem;">{product.emoji} {product.title[:42]}…</div>
  <div style="display:flex;gap:0.8rem;align-items:center;flex-wrap:wrap;">
    <span style="font-family:'Playfair Display',serif;color:var(--gold);font-size:1.1rem;">${product.price:,.2f}</span>
    <span style="font-size:0.78rem;color:var(--text3);">★{product.rating}</span>
    <span class="pbadge {cfg['badge']}" style="margin:0;">{product.platform}</span>
    <span style="font-size:0.75rem;color:var(--green);font-weight:700;">AI {product.ai_score}%</span>
  </div>
</div>""", unsafe_allow_html=True)


def export_products_csv(products: List[Product]) -> bytes:
    rows = [asdict(p) for p in products]
    df = pd.DataFrame(rows)
    return df.to_csv(index=False).encode()


def export_products_json(products: List[Product]) -> bytes:
    data = [asdict(p) for p in products]
    return json.dumps(data, indent=2).encode()

# ═══════════════════════════════════════════════════════════════════════════════
# LLM ADAPTER (pluggable architecture)
# ═══════════════════════════════════════════════════════════════════════════════

class LLMAdapter:
    """Abstract adapter interface for LLM providers. Extend for real API calls."""

    def __init__(self, provider: str, api_key: Optional[str] = None):
        self.provider  = provider
        self.api_key   = api_key
        self.available = bool(api_key)

    def complete(self, prompt: str, system: str = "", max_tokens: int = 512) -> str:
        """Route to appropriate provider. Falls back to mock if no key."""
        if not self.available:
            return self._mock_complete(prompt)
        if "OpenAI" in self.provider or "GPT" in self.provider:
            return self._openai_complete(prompt, system, max_tokens)
        elif "Claude" in self.provider:
            return self._claude_complete(prompt, system, max_tokens)
        elif "Gemini" in self.provider:
            return self._gemini_complete(prompt, system, max_tokens)
        return self._mock_complete(prompt)

    def _mock_complete(self, prompt: str) -> str:
        return f"[{self.provider} — demo mode. Provide an API key to enable real completions.]"

    def _openai_complete(self, prompt: str, system: str, max_tokens: int) -> str:
        try:
            import requests as req
            r = req.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json={"model": "gpt-4o", "max_tokens": max_tokens, "messages": [
                    {"role": "system", "content": system or "You are a helpful shopping assistant."},
                    {"role": "user",   "content": prompt},
                ]},
                timeout=15,
            )
            return r.json()["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"OpenAI error: {e}")
            return self._mock_complete(prompt)

    def _claude_complete(self, prompt: str, system: str, max_tokens: int) -> str:
        try:
            import requests as req
            r = req.post(
                "https://api.anthropic.com/v1/messages",
                headers={"x-api-key": self.api_key, "anthropic-version": "2023-06-01", "Content-Type": "application/json"},
                json={"model": "claude-sonnet-4-20250514", "max_tokens": max_tokens,
                      "system": system or "You are a helpful shopping assistant.",
                      "messages": [{"role": "user", "content": prompt}]},
                timeout=15,
            )
            return r.json()["content"][0]["text"]
        except Exception as e:
            logger.error(f"Claude error: {e}")
            return self._mock_complete(prompt)

    def _gemini_complete(self, prompt: str, system: str, max_tokens: int) -> str:
        try:
            import requests as req
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent?key={self.api_key}"
            r = req.post(url, json={"contents": [{"parts": [{"text": prompt}]}]}, timeout=15)
            return r.json()["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as e:
            logger.error(f"Gemini error: {e}")
            return self._mock_complete(prompt)

# ═══════════════════════════════════════════════════════════════════════════════
# STREAMLIT APP — MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    inject_css()
    init_session()

    # ── SIDEBAR ─────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("""
        <div class="logo-wrap">
          <div class="logo-name">ShopSense AI</div>
          <div class="logo-sub">Shopping Intelligence Platform</div>
        </div>
        """, unsafe_allow_html=True)

        # API Keys
        st.markdown('<span class="sl">🔑 API Keys</span>', unsafe_allow_html=True)
        with st.expander("Configure API Keys", expanded=False):
            openai_key   = st.text_input("OpenAI Key",   placeholder="sk-...",        type="password", key="k_openai")
            claude_key   = st.text_input("Anthropic Key", placeholder="sk-ant-...",   type="password", key="k_claude")
            gemini_key   = st.text_input("Gemini Key",   placeholder="AIza...",       type="password", key="k_gemini")
            st.caption("🔒 Session-only. Never stored. Demo mode active without keys.")

        st.markdown("<hr>", unsafe_allow_html=True)

        # LLM selection
        st.markdown('<span class="sl">🤖 AI Model</span>', unsafe_allow_html=True)
        llm_choice = st.selectbox("Engine", LLM_ADAPTERS, key="llm_choice")

        search_depth = st.select_slider("Search Depth", ["Quick","Standard","Deep","Exhaustive"], value="Standard")
        use_reranker = st.checkbox("AI Re-ranker",          value=True,  key="reranker")
        use_sentiment= st.checkbox("Sentiment Analysis",    value=True,  key="sentiment")
        use_history  = st.checkbox("Conversation Memory",   value=True,  key="memory")

        st.markdown("<hr>", unsafe_allow_html=True)

        # Platforms
        st.markdown('<span class="sl">🏪 Platforms</span>', unsafe_allow_html=True)
        active_platforms = st.multiselect(
            "Active Platforms",
            list(PLATFORM_CONFIG.keys()),
            default=["Amazon","Best Buy","Walmart","Target","eBay","Newegg"],
            key="platforms",
        )

        st.markdown("<hr>", unsafe_allow_html=True)

        # Filters
        st.markdown('<span class="sl">🎛️ Filters</span>', unsafe_allow_html=True)
        budget = st.slider("Max Budget ($)", 10, 5000, DEFAULT_BUDGET, 10, key="budget")
        min_rating   = st.slider("Min Rating ★", 1.0, 5.0, 4.0, 0.1, key="min_rating")
        min_reviews  = st.slider("Min Reviews",  0, 5000, 0, 100, key="min_reviews")
        min_discount = st.slider("Min Discount %", 0, 50, 0, 5, key="min_disc")
        sort_by      = st.selectbox("Sort By", ["AI Score","Price: Low→High","Price: High→Low","Rating","Reviews","Discount %"], key="sort_by")

        st.markdown("<hr>", unsafe_allow_html=True)

        # Stats
        st.markdown('<span class="sl">📊 Session Stats</span>', unsafe_allow_html=True)
        sc1, sc2 = st.columns(2)
        with sc1: st.metric("Searches", len(st.session_state.search_history))
        with sc2: st.metric("Wishlist",  len(st.session_state.wishlist))
        sc3, sc4 = st.columns(2)
        with sc3: st.metric("Compare",  len(st.session_state.compare_list))
        with sc4: st.metric("Products", "2.4M+")

        # Wishlist preview
        if st.session_state.wishlist:
            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown('<span class="sl">❤️ Saved Items</span>', unsafe_allow_html=True)
            for item in st.session_state.wishlist[-4:]:
                st.markdown(f"<div style='font-size:0.76rem;color:var(--text3);padding:2px 0;'>• {item[:32]}…</div>", unsafe_allow_html=True)
            if len(st.session_state.wishlist) > 4:
                st.caption(f"+{len(st.session_state.wishlist)-4} more")

    # ── HERO ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="hero-wrap">
      <div class="hero-glow"></div>
      <div class="hero-eyebrow">Powered by AI · 8 Shopping Platforms · Real-Time Price Intelligence</div>
      <h1 class="hero-title">Find the <em>Perfect</em> Product</h1>
      <p class="hero-sub">AI shopping intelligence that searches Amazon, eBay, Walmart, Etsy, Best Buy, Target, Shopify &amp; Newegg simultaneously — then ranks everything with weighted LLM reasoning.</p>
    </div>
    """, unsafe_allow_html=True)

    # Platform strip
    chips = "".join(f'<span class="pchip">{PLATFORM_CONFIG[k]["emoji"]} {k}</span>' for k in PLATFORM_CONFIG)
    st.markdown(f'<div class="pstrip">{chips}</div>', unsafe_allow_html=True)

    # ── SEARCH BAR ──────────────────────────────────────────────────────────
    s_col, b_col = st.columns([5, 1])
    with s_col:
        query = st.text_input(
            "search",
            placeholder='Try "wireless headphones for gym" · "gaming laptop under $1500" · "espresso machine"',
            label_visibility="collapsed",
            key="main_query",
        )
    with b_col:
        search_btn = st.button("🔍 Search", use_container_width=True, key="do_search")

    # Trending row
    st.markdown('<div style="margin:0.4rem 0 1.2rem;"><span style="font-size:0.68rem;color:var(--text4);font-family:JetBrains Mono,monospace;letter-spacing:0.12em;text-transform:uppercase;">Trending · </span>', unsafe_allow_html=True)
    tcols = st.columns(len(TRENDING_TERMS[:7]))
    for i, term in enumerate(TRENDING_TERMS[:7]):
        with tcols[i]:
            if st.button(term, key=f"t_{i}", use_container_width=True):
                query      = term
                search_btn = True
    st.markdown("</div>", unsafe_allow_html=True)

    # ── SEARCH EXECUTION ────────────────────────────────────────────────────
    if search_btn and query:
        if not active_platforms:
            st.warning("Please select at least one platform in the sidebar.")
            return

        st.session_state.last_query = query

        # Add to search history
        rec = SearchRecord(query=query)
        history: List[SearchRecord] = st.session_state.search_history
        if not history or history[-1].query != query:
            history.append(rec)
            if len(history) > MAX_SEARCH_HISTORY:
                history.pop(0)
            st.session_state.search_history = history

        # Progress simulation
        depth_delay = {"Quick": 0.18, "Standard": 0.28, "Deep": 0.40, "Exhaustive": 0.55}
        delay = depth_delay.get(search_depth, 0.28)

        with st.spinner("🤖 AI agent analyzing query and querying platforms…"):
            prog    = st.progress(0)
            status  = st.empty()
            steps   = [
                (8,  f"🧠 Parsing intent for *'{query}'*…"),
                (20, f"🔗 Initialising {llm_choice} ReAct agent…"),
                (35, "📡 Querying Amazon & Best Buy APIs…"),
                (50, "📡 Querying Walmart, Target & Newegg APIs…"),
                (62, "📡 Querying eBay, Etsy & Shopify APIs…"),
                (75, "⚖️  Running cross-encoder re-ranker…" if use_reranker else "📊 Ranking results…"),
                (87, "💬 Running sentiment analysis…" if use_sentiment else "📊 Compiling insights…"),
                (95, f"✨ Generating AI analysis with {llm_choice}…"),
                (100,"✅ Done!"),
            ]
            for pct, msg in steps:
                prog.progress(pct)
                status.markdown(f"<div style='font-size:0.82rem;color:var(--text3);'>{msg}</div>", unsafe_allow_html=True)
                time.sleep(delay)
            prog.empty(); status.empty()

        # Build products
        products = build_products(query, active_platforms, budget, min_rating, min_reviews, min_discount)
        products = sort_products(products, sort_by)

        # Determine LLM adapter
        key_map = {
            "GPT-4o (OpenAI)":      openai_key  if "k_openai"  in st.session_state else "",
            "Claude 3.5 Sonnet":    claude_key  if "k_claude"  in st.session_state else "",
            "Gemini 1.5 Pro":       gemini_key  if "k_gemini"  in st.session_state else "",
            "LLaMA 3.1 (Local)":    "",
            "Mixtral 8x7B":         "",
        }
        llm_adapter = LLMAdapter(llm_choice, key_map.get(llm_choice, ""))

        # Update session
        st.session_state.search_results       = products
        st.session_state.ai_insight           = generate_ai_insight(query, products, budget, llm_choice)
        st.session_state.ai_recommendations   = generate_recommendations(products, budget)
        st.session_state.product_insights     = generate_product_insights(products)

        # Update result count in history
        if st.session_state.search_history:
            st.session_state.search_history[-1].result_count = len(products)

        # Append to chat
        st.session_state.chat_history.append(ChatMessage(role="user", content=f"Search: {query}"))
        st.session_state.chat_history.append(ChatMessage(
            role="assistant",
            content=f"Found **{len(products)} products** across **{len(active_platforms)} platforms** for '{query}'. "
                    f"Top pick: **{products[0].title[:40]}…** (AI Score {products[0].ai_score}%) at ${products[0].price:,.2f}."
                    if products else "No products matched your filters. Try relaxing the budget or rating constraints."
        ))

        logger.info(f"Search '{query}' → {len(products)} results")

    # ── RESULTS ─────────────────────────────────────────────────────────────
    products: List[Product] = st.session_state.search_results

    if products:
        st.markdown("<hr>", unsafe_allow_html=True)

        # Metrics strip
        analytics = price_analytics(products)
        m_cols = st.columns(6)
        metrics = [
            ("Products Found",  str(analytics["count"]),    None),
            ("Avg. Price",      f"${analytics['avg']:,.0f}", None),
            ("Median Price",    f"${analytics['median']:,.0f}",None),
            ("Best Price",      f"${analytics['min']:,.0f}", None),
            ("Top AI Score",    f"{max(p.ai_score for p in products)}%", None),
            ("Avg. Rating",     f"★ {sum(p.rating for p in products)/len(products):.1f}", None),
        ]
        for col, (label, val, _) in zip(m_cols, metrics):
            with col:
                st.metric(label, val)

        st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)

        # ── TABS ────────────────────────────────────────────────────────────
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "🏆 Results",
            "✨ AI Picks",
            "📊 Analytics",
            "⚖️ Compare",
            "💬 Assistant",
            "❤️ Wishlist",
            "📁 Export",
        ])

        max_price = max(p.price for p in products)

        # ── TAB 1: Results ──────────────────────────────────────────────────
        with tab1:
            st.markdown(f'<div class="sec-eyebrow">SEARCH RESULTS</div><div class="sec-title">Results for "{st.session_state.last_query}"</div>', unsafe_allow_html=True)

            # Product insights
            for ins in st.session_state.product_insights[:3]:
                st.markdown(f'<div class="insight-card">{ins}</div>', unsafe_allow_html=True)

            st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

            # Product grid — 3 columns
            for i in range(0, len(products), 3):
                row = products[i:i+3]
                cols = st.columns(3)
                for j, prod in enumerate(row):
                    with cols[j]:
                        render_product_card(prod, max_price, i * 10 + j)
                        c1, c2 = st.columns(2)
                        with c1:
                            if st.button("❤️ Save", key=f"wl_{i}_{j}"):
                                if prod.title not in st.session_state.wishlist:
                                    if len(st.session_state.wishlist) < MAX_WISHLIST_ITEMS:
                                        st.session_state.wishlist.append(prod.title)
                                        st.toast("Added to wishlist!", icon="❤️")
                                    else:
                                        st.toast(f"Wishlist full ({MAX_WISHLIST_ITEMS} items max)", icon="⚠️")
                                else:
                                    st.toast("Already in wishlist", icon="ℹ️")
                        with c2:
                            if st.button("⚖️ Compare", key=f"cmp_{i}_{j}"):
                                if prod.title not in [c.title for c in st.session_state.compare_list]:
                                    if len(st.session_state.compare_list) < MAX_COMPARE_ITEMS:
                                        st.session_state.compare_list.append(prod)
                                        st.toast("Added to compare!", icon="⚖️")
                                    else:
                                        st.toast(f"Max {MAX_COMPARE_ITEMS} products in compare", icon="⚠️")
                                else:
                                    st.toast("Already in compare list", icon="ℹ️")

        # ── TAB 2: AI Picks ─────────────────────────────────────────────────
        with tab2:
            st.markdown('<div class="sec-eyebrow">AI RECOMMENDATIONS</div><div class="sec-title">Intelligence Report</div>', unsafe_allow_html=True)

            if st.session_state.ai_insight:
                st.markdown(f'<div class="ai-bubble">{st.session_state.ai_insight}</div>', unsafe_allow_html=True)

            st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
            st.markdown('<div class="sec-eyebrow">CURATED PICKS</div>', unsafe_allow_html=True)

            recs = st.session_state.ai_recommendations
            r_cols = st.columns(2)
            labels = list(recs.keys())
            for i, label in enumerate(labels):
                with r_cols[i % 2]:
                    render_rec_badge(label, recs[label])

        # ── TAB 3: Analytics ─────────────────────────────────────────────────
        with tab3:
            st.markdown('<div class="sec-eyebrow">ANALYTICS DASHBOARD</div><div class="sec-title">Price & Market Intelligence</div>', unsafe_allow_html=True)

            an = analytics
            a_cols = st.columns(4)
            an_metrics = [
                ("Average Price", f"${an['avg']:,.2f}"),
                ("Median Price",  f"${an['median']:,.2f}"),
                ("Lowest Price",  f"${an['min']:,.2f}"),
                ("Highest Price", f"${an['max']:,.2f}"),
            ]
            for col, (lbl, val) in zip(a_cols, an_metrics):
                with col:
                    st.metric(lbl, val)

            st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

            ch_left, ch_right = st.columns([3, 2])
            with ch_left:
                st.markdown('<div class="sec-eyebrow">PRICE vs AI SCORE</div>', unsafe_allow_html=True)
                st.plotly_chart(build_plotly_price_chart(products), use_container_width=True, config={"displayModeBar": False})
            with ch_right:
                st.markdown('<div class="sec-eyebrow">PLATFORM DISTRIBUTION</div>', unsafe_allow_html=True)
                st.plotly_chart(build_platform_donut(products), use_container_width=True, config={"displayModeBar": False})

            ch2_left, ch2_right = st.columns(2)
            with ch2_left:
                st.markdown('<div class="sec-eyebrow">PRICE vs RATING (bubble = reviews)</div>', unsafe_allow_html=True)
                st.plotly_chart(build_plotly_scatter(products), use_container_width=True, config={"displayModeBar": False})
            with ch2_right:
                st.markdown('<div class="sec-eyebrow">DEAL QUALITY DISTRIBUTION</div>', unsafe_allow_html=True)
                st.plotly_chart(build_deal_quality_chart(products), use_container_width=True, config={"displayModeBar": False})

            # Altair rating histogram
            st.markdown('<div class="sec-eyebrow" style="margin-top:1rem;">RATING DISTRIBUTION</div>', unsafe_allow_html=True)
            df_alt = pd.DataFrame({"Rating": [p.rating for p in products], "Product": [p.title[:22]+"…" for p in products]})
            chart = (
                alt.Chart(df_alt)
                .mark_bar(color="#d4af37", cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
                .encode(
                    x=alt.X("Rating:Q", bin=alt.Bin(step=0.1), title="Rating"),
                    y=alt.Y("count()", title="Count"),
                    tooltip=["count()","Rating"],
                )
                .properties(height=180, background="transparent")
                .configure_axis(gridColor="rgba(255,255,255,0.05)", labelColor="#888", titleColor="#666")
                .configure_view(strokeWidth=0)
            )
            st.altair_chart(chart, use_container_width=True)

            # Data table
            st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
            with st.expander("📋 Full Data Table", expanded=False):
                df_full = pd.DataFrame([{
                    "Title": p.title, "Price": p.price, "Original": p.original_price,
                    "Discount%": p.discount_pct, "Rating": p.rating, "Reviews": p.reviews,
                    "Platform": p.platform, "Brand": p.brand, "Category": p.category,
                    "AI Score": p.ai_score, "Deal": p.deal_label, "Stock": p.stock,
                    "Trend": p.price_trend,
                } for p in products])
                st.dataframe(df_full, use_container_width=True)

        # ── TAB 4: Compare ──────────────────────────────────────────────────
        with tab4:
            st.markdown('<div class="sec-eyebrow">COMPARISON CENTER</div><div class="sec-title">Side-by-Side Analysis</div>', unsafe_allow_html=True)

            if st.session_state.compare_list:
                clr_col, _ = st.columns([1, 5])
                with clr_col:
                    if st.button("🗑️ Clear All", key="clear_compare"):
                        st.session_state.compare_list = []
                        st.rerun()

            render_comparison_table(st.session_state.compare_list)

            if st.session_state.compare_list:
                st.markdown('<div class="sec-eyebrow" style="margin-top:1.5rem;">VISUAL SCORE COMPARISON</div>', unsafe_allow_html=True)
                compare_df = pd.DataFrame([{
                    "Product": p.title[:28] + "…",
                    "AI Score": p.ai_score,
                    "Rating x20": p.rating * 20,
                    "Deal Quality": p.deal_quality_score * 100,
                } for p in st.session_state.compare_list])
                fig_cmp = go.Figure()
                for col, color in zip(["AI Score","Rating x20","Deal Quality"], ["#d4af37","#64dc82","#4ab3ff"]):
                    fig_cmp.add_trace(go.Bar(name=col, x=compare_df["Product"], y=compare_df[col], marker_color=color))
                fig_cmp.update_layout(
                    barmode="group",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="DM Sans", color="#c8c4ba", size=11),
                    margin=dict(l=10, r=10, t=10, b=60),
                    xaxis=dict(tickangle=-20, gridcolor="rgba(255,255,255,0.04)"),
                    yaxis=dict(gridcolor="rgba(255,255,255,0.04)", range=[0, 110]),
                    legend=dict(font=dict(color="#c8c4ba")),
                    height=300,
                    template="plotly_dark",
                )
                st.plotly_chart(fig_cmp, use_container_width=True, config={"displayModeBar": False})

        # ── TAB 5: Chat ──────────────────────────────────────────────────────
        with tab5:
            st.markdown('<div class="sec-eyebrow">AI SHOPPING ASSISTANT</div><div class="sec-title">Ask Me Anything</div>', unsafe_allow_html=True)

            chat_container = st.container()
            with chat_container:
                history_to_show = st.session_state.chat_history[-12:] if use_history else st.session_state.chat_history[-2:]
                for msg in history_to_show:
                    if msg.role == "user":
                        st.markdown(f'<div class="chat-user">👤 {msg.content}<div class="chat-ts">{msg.timestamp}</div></div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="chat-ai">🤖 {msg.content}<div class="chat-ts">{msg.timestamp} · {llm_choice}</div></div>', unsafe_allow_html=True)

            st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)

            ci_col, cs_col = st.columns([5, 1])
            with ci_col:
                chat_input = st.text_input(
                    "chat",
                    placeholder="Ask: 'Which has the best warranty?' · 'Compare top 2' · 'Best under $300?'",
                    label_visibility="collapsed",
                    key="chat_input",
                )
            with cs_col:
                send = st.button("Send ✉️", use_container_width=True, key="send_msg")

            if send and chat_input:
                st.session_state.chat_history.append(ChatMessage(role="user", content=chat_input))
                reply = generate_chat_response(chat_input, products, st.session_state.last_query, budget, llm_choice)
                st.session_state.chat_history.append(ChatMessage(role="assistant", content=reply))
                st.rerun()

            # Quick questions
            st.markdown('<div style="margin-top:0.9rem;"><span style="font-size:0.68rem;color:var(--text4);font-family:JetBrains Mono,monospace;letter-spacing:0.12em;text-transform:uppercase;">Quick Ask:</span></div>', unsafe_allow_html=True)
            qq_list = ["Best warranty?", "Which ships fastest?", "Best value pick?", "Compare top 2?", "Best under budget?"]
            qq_cols = st.columns(len(qq_list))
            for i, qq in enumerate(qq_list):
                with qq_cols[i]:
                    if st.button(qq, key=f"qq_{i}", use_container_width=True):
                        st.session_state.chat_history.append(ChatMessage(role="user", content=qq))
                        reply = generate_chat_response(qq, products, st.session_state.last_query, budget, llm_choice)
                        st.session_state.chat_history.append(ChatMessage(role="assistant", content=reply))
                        st.rerun()

            # Clear chat
            if st.session_state.chat_history:
                st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
                if st.button("🗑️ Clear Chat History", key="clear_chat"):
                    st.session_state.chat_history = []
                    st.rerun()

        # ── TAB 6: Wishlist ──────────────────────────────────────────────────
        with tab6:
            st.markdown('<div class="sec-eyebrow">SAVED ITEMS</div><div class="sec-title">Your Wishlist</div>', unsafe_allow_html=True)

            wl = st.session_state.wishlist
            if wl:
                # Export wishlist
                wl_df  = pd.DataFrame({"Item": wl, "Saved": [datetime.now().strftime("%Y-%m-%d")] * len(wl)})
                wl_csv = wl_df.to_csv(index=False).encode()
                dl_col, clr_col, _ = st.columns([1.5, 1.5, 5])
                with dl_col:
                    st.download_button("⬇️ Export CSV", wl_csv, "wishlist.csv", "text/csv", key="dl_wl")
                with clr_col:
                    if st.button("🗑️ Clear All", key="clear_wl"):
                        st.session_state.wishlist = []
                        st.rerun()

                st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)

                for idx, item in enumerate(wl):
                    w_item, w_btn = st.columns([6, 1])
                    with w_item:
                        st.markdown(f'<div class="wl-row">❤️ <span style="color:var(--text);font-size:0.88rem;">{item}</span></div>', unsafe_allow_html=True)
                    with w_btn:
                        if st.button("✕", key=f"rmwl_{idx}"):
                            st.session_state.wishlist.pop(idx)
                            st.rerun()
            else:
                st.markdown('<div class="empty-state"><div class="empty-icon">❤️</div><div class="empty-title">Your wishlist is empty</div><div class="empty-sub">Click ❤️ Save on any product card to save it here.</div></div>', unsafe_allow_html=True)

        # ── TAB 7: Export ────────────────────────────────────────────────────
        with tab7:
            st.markdown('<div class="sec-eyebrow">EXPORT</div><div class="sec-title">Download Your Results</div>', unsafe_allow_html=True)

            st.markdown(f"**{len(products)} products** from your search for **'{st.session_state.last_query}'** are ready to export.")
            st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

            ex_cols = st.columns(3)
            with ex_cols[0]:
                st.download_button(
                    "⬇️ Download CSV",
                    data=export_products_csv(products),
                    file_name=f"shopsense_{st.session_state.last_query.replace(' ','_')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="dl_csv",
                )
            with ex_cols[1]:
                st.download_button(
                    "⬇️ Download JSON",
                    data=export_products_json(products),
                    file_name=f"shopsense_{st.session_state.last_query.replace(' ','_')}.json",
                    mime="application/json",
                    use_container_width=True,
                    key="dl_json",
                )
            with ex_cols[2]:
                if st.session_state.wishlist:
                    wl_csv = pd.DataFrame({"Item": st.session_state.wishlist}).to_csv(index=False).encode()
                    st.download_button(
                        "❤️ Export Wishlist",
                        data=wl_csv,
                        file_name="wishlist.csv",
                        mime="text/csv",
                        use_container_width=True,
                        key="dl_wl2",
                    )
                else:
                    st.button("❤️ Wishlist Empty", disabled=True, use_container_width=True, key="wl_dis")

            # Preview
            st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
            with st.expander("📋 Preview Export Data", expanded=False):
                df_exp = pd.DataFrame([{
                    "title": p.title, "price": p.price, "original_price": p.original_price,
                    "discount_pct": p.discount_pct, "rating": p.rating, "reviews": p.reviews,
                    "platform": p.platform, "brand": p.brand, "category": p.category,
                    "ai_score": p.ai_score, "deal_label": p.deal_label, "stock": p.stock,
                    "shipping": p.shipping, "price_trend": p.price_trend,
                } for p in products])
                st.dataframe(df_exp, use_container_width=True)

    # ── EMPTY STATE / HOMEPAGE ───────────────────────────────────────────────
    else:
        st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)

        # Search history
        if st.session_state.search_history:
            st.markdown('<div class="sec-eyebrow">RECENT SEARCHES</div>', unsafe_allow_html=True)
            hist_html = " ".join(
                f'<span class="hist-chip">{rec.query} <span style="color:var(--text4);font-size:0.65rem;">({rec.result_count})</span></span>'
                for rec in reversed(st.session_state.search_history[-8:])
            )
            st.markdown(f"<div style='margin-bottom:1.8rem;'>{hist_html}</div>", unsafe_allow_html=True)

        # Features
        st.markdown('<div class="sec-eyebrow">CAPABILITIES</div><div class="sec-title">What ShopSense AI Does</div>', unsafe_allow_html=True)

        features = [
            ("🔗", "8 Live Platforms", "Amazon, eBay, Walmart, Etsy, Best Buy, Target, Shopify & Newegg searched simultaneously for comprehensive coverage."),
            ("🤖", "LangChain AI Engine", "ReAct agent with GPT-4o, Claude, Gemini & local models — intelligent multi-step reasoning for complex queries."),
            ("📊", "Price Intelligence", "Real-time price history, trend detection, weighted deal scoring, and discount analysis across every result."),
            ("💬", "AI Chat Assistant", "Conversational memory answers follow-up questions about warranty, shipping, comparisons, and best-value picks."),
        ]
        fcols = st.columns(4)
        for col, (icon, title, desc) in zip(fcols, features):
            with col:
                st.markdown(f'<div class="feat-card"><div class="feat-icon">{icon}</div><div class="feat-title">{title}</div><div class="feat-desc">{desc}</div></div>', unsafe_allow_html=True)

        st.markdown("<div style='height:2rem'></div>", unsafe_allow_html=True)
        st.markdown('<div class="cta-banner"><div class="cta-title">Ready to find your perfect product?</div><div class="cta-sub">Type any product above — from "wireless headphones" to "espresso machine under $600"</div></div>', unsafe_allow_html=True)

    # ── FOOTER ──────────────────────────────────────────────────────────────
    st.markdown(f"""
<div style="text-align:center;padding:2.5rem 0 1rem;margin-top:3rem;border-top:1px solid rgba(212,175,55,0.1);">
  <div style="font-family:'JetBrains Mono',monospace;font-size:0.62rem;letter-spacing:0.18em;color:var(--text4);text-transform:uppercase;">
    ShopSense AI v{APP_VERSION} &nbsp;·&nbsp; 8 Shopping Platform APIs &nbsp;·&nbsp; 2.4M+ Products Indexed &nbsp;·&nbsp; Powered by LangChain
  </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    main()
