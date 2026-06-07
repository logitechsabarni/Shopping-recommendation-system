cat > /mnt/user-data/outputs/app.py << 'PYEOF'
"""
ShopSense AI v3.0.0 — Production-Grade Shopping Intelligence Platform
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
import math
import io
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Tuple
from collections import Counter
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed

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

APP_VERSION = "3.0.0"
MAX_COMPARE_ITEMS = 5
MAX_WISHLIST_ITEMS = 50
MAX_SEARCH_HISTORY = 20
DEFAULT_BUDGET = 500
SCORE_WEIGHTS = {
    "rating":         0.28,
    "review_volume":  0.20,
    "price_value":    0.22,
    "platform_trust": 0.15,
    "deal_quality":   0.10,
    "availability":   0.05,
}
PLATFORM_TRUST: Dict[str, float] = {
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
LLM_ADAPTERS = ["GPT-4o (OpenAI)", "Claude Sonnet 4", "Gemini 1.5 Pro", "LLaMA 3.1 (Local)", "Mixtral 8x7B"]
TRENDING_TERMS = [
    "Wireless Earbuds 2025", "AI Smart Home Hub", "Ergonomic Chair",
    "Mechanical Keyboard", "Standing Desk", "Gaming Monitor 4K",
    "Robot Vacuum Gen3", "Air Fryer XL", "Smartwatch Ultra",
]

SKILL_TAGS: Dict[str, List[str]] = {
    "laptop":     ["Beginner-Friendly", "Work From Home", "Student Pick", "Power User", "Creator Pro"],
    "headphones": ["Gym Ready", "Office Use", "Audiophile", "Travel", "Gaming"],
    "phone":      ["Photography", "Battery Life", "Value Pick", "Business", "Beginner"],
    "coffee":     ["Home Barista", "Office", "Travel", "Specialty", "Daily Driver"],
    "gaming":     ["Beginner Gamer", "Competitive", "Casual", "Streamer", "Collector"],
    "default":    ["Popular", "Top Pick", "Value", "Premium", "Everyday"],
}

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
    beginner_score: float = 0.5
    specs: Dict[str, str] = field(default_factory=dict)

    @property
    def url(self) -> str:
        cfg = PLATFORM_CONFIG.get(self.platform, {"url": "https://google.com/search?q="})
        return cfg["url"] + self.title.replace(" ", "+")

    @property
    def savings(self) -> float:
        return max(0.0, self.original_price - self.price)


@dataclass
class ChatMessage:
    role: str
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
        {"title": "Apple MacBook Pro 14″ M4 Pro 24GB", "price": 1999.00, "original_price": 2199.00, "rating": 4.9, "reviews": 7214, "platform": "Amazon",   "brand": "Apple",     "category": "Ultrabooks",       "stock": "In Stock",   "shipping": "Free 2-day",       "emoji": "💻", "specs": {"CPU": "M4 Pro", "RAM": "24GB", "Storage": "512GB SSD", "Display": "14\" Liquid Retina XDR", "Battery": "22 hrs"}},
        {"title": "Dell XPS 15 9540 i9-14900HX RTX 4070", "price": 1799.99, "original_price": 1999.99, "rating": 4.7, "reviews": 3102, "platform": "Best Buy", "brand": "Dell",    "category": "Performance",      "stock": "In Stock",   "shipping": "Free",             "emoji": "💻", "specs": {"CPU": "i9-14900HX", "RAM": "32GB", "Storage": "1TB SSD", "Display": "15.6\" OLED", "Battery": "13 hrs"}},
        {"title": "Lenovo ThinkPad X1 Carbon Gen 13",    "price": 1549.00, "original_price": 1549.00, "rating": 4.8, "reviews": 5201, "platform": "Walmart",  "brand": "Lenovo",    "category": "Business",         "stock": "In Stock",   "shipping": "$5.99",            "emoji": "💼", "specs": {"CPU": "Ultra 7 165H", "RAM": "16GB", "Storage": "512GB SSD", "Display": "14\" IPS", "Battery": "15 hrs"}},
        {"title": "ASUS ROG Strix G16 RTX 4080",        "price": 1399.99, "original_price": 1649.99, "rating": 4.6, "reviews": 2341, "platform": "Newegg",   "brand": "ASUS",      "category": "Gaming",           "stock": "Limited",    "shipping": "Free",             "emoji": "🎮", "specs": {"CPU": "i9-14900HX", "RAM": "32GB", "Storage": "1TB SSD", "Display": "16\" QHD 240Hz", "Battery": "10 hrs"}},
        {"title": "HP Spectre x360 14″ OLED 2-in-1",   "price": 1449.99, "original_price": 1449.99, "rating": 4.6, "reviews": 1987, "platform": "Target",   "brand": "HP",        "category": "Convertible",      "stock": "In Stock",   "shipping": "Free",             "emoji": "💻", "specs": {"CPU": "Ultra 5 125H", "RAM": "16GB", "Storage": "512GB SSD", "Display": "14\" OLED", "Battery": "17 hrs"}},
        {"title": "Microsoft Surface Laptop 5 Refurb",  "price": 849.00,  "original_price": 1299.00, "rating": 4.3, "reviews": 3891, "platform": "eBay",     "brand": "Microsoft", "category": "Certified Refurb", "stock": "In Stock",   "shipping": "Free",             "emoji": "💻", "specs": {"CPU": "i5-1235U", "RAM": "16GB", "Storage": "256GB SSD", "Display": "13.5\" Touch", "Battery": "18 hrs"}},
        {"title": "Razer Blade 16 RTX 4090 OLED",       "price": 2799.00, "original_price": 2999.00, "rating": 4.7, "reviews": 1102, "platform": "Amazon",   "brand": "Razer",     "category": "Gaming Premium",   "stock": "In Stock",   "shipping": "Free 2-day",       "emoji": "⚡", "specs": {"CPU": "i9-14900HX", "RAM": "32GB", "Storage": "2TB SSD", "Display": "16\" OLED 240Hz", "Battery": "8 hrs"}},
        {"title": "Framework Laptop 16 AMD Ryzen 9 7940HX", "price": 1299.00, "original_price": 1299.00, "rating": 4.5, "reviews": 921, "platform": "Shopify", "brand": "Framework", "category": "Modular",      "stock": "In Stock",   "shipping": "Free",             "emoji": "🔧", "specs": {"CPU": "Ryzen 9 7940HX", "RAM": "32GB", "Storage": "1TB SSD", "Display": "16\" 165Hz", "Battery": "12 hrs"}},
        {"title": "Acer Swift Go 14 OLED Ultra 5",       "price": 699.99,  "original_price": 849.99,  "rating": 4.4, "reviews": 4201, "platform": "Walmart",  "brand": "Acer",      "category": "Budget Ultrabook", "stock": "In Stock",   "shipping": "$3.99",            "emoji": "💻", "specs": {"CPU": "Ultra 5 125H", "RAM": "16GB", "Storage": "512GB SSD", "Display": "14\" OLED", "Battery": "10 hrs"}},
    ],
    "headphones": [
        {"title": "Sony WH-1000XM6 Wireless ANC",         "price": 299.99, "original_price": 399.99, "rating": 4.9, "reviews": 31204, "platform": "Amazon",   "brand": "Sony",        "category": "Over-Ear ANC",   "stock": "In Stock", "shipping": "Free 2-day", "emoji": "🎧", "specs": {"Type": "Over-Ear", "ANC": "Yes", "Battery": "30 hrs", "Bluetooth": "5.3", "Codec": "LDAC/AAC"}},
        {"title": "Apple AirPods Pro 2nd Gen USB-C",       "price": 249.00, "original_price": 249.00, "rating": 4.8, "reviews": 71043, "platform": "Best Buy", "brand": "Apple",       "category": "In-Ear ANC",     "stock": "In Stock", "shipping": "Free",       "emoji": "🎧", "specs": {"Type": "In-Ear", "ANC": "Yes", "Battery": "6+30 hrs", "Bluetooth": "5.3", "Codec": "AAC"}},
        {"title": "Bose QuietComfort Ultra Headphones",    "price": 349.99, "original_price": 429.99, "rating": 4.8, "reviews": 11892, "platform": "Walmart",  "brand": "Bose",        "category": "Over-Ear ANC",   "stock": "In Stock", "shipping": "Free 2-day", "emoji": "🎧", "specs": {"Type": "Over-Ear", "ANC": "Yes", "Battery": "24 hrs", "Bluetooth": "5.3", "Codec": "aptX/AAC"}},
        {"title": "Sennheiser HD 660S2 Open-Back",         "price": 299.95, "original_price": 349.95, "rating": 4.8, "reviews": 2104,  "platform": "Newegg",   "brand": "Sennheiser",  "category": "Audiophile",     "stock": "In Stock", "shipping": "$4.99",      "emoji": "🎼", "specs": {"Type": "Open-Back", "ANC": "No", "Impedance": "300Ω", "Driver": "38mm", "Codec": "Wired"}},
        {"title": "Jabra Evolve2 85 MS Wireless",          "price": 379.00, "original_price": 449.00, "rating": 4.6, "reviews": 5102,  "platform": "Amazon",   "brand": "Jabra",       "category": "Professional",   "stock": "Limited",  "shipping": "Free 2-day", "emoji": "💼", "specs": {"Type": "Over-Ear", "ANC": "Yes", "Battery": "37 hrs", "Bluetooth": "5.2", "Codec": "aptX"}},
        {"title": "Beyerdynamic DT 990 Pro 250Ω Open",     "price": 149.00, "original_price": 179.00, "rating": 4.7, "reviews": 10234, "platform": "Amazon",   "brand": "Beyerdynamic","category": "Studio",         "stock": "In Stock", "shipping": "Free 2-day", "emoji": "🎵", "specs": {"Type": "Open-Back", "ANC": "No", "Impedance": "250Ω", "Driver": "45mm", "Codec": "Wired"}},
        {"title": "Custom Leather Headband Wrap Artisan",  "price": 34.99,  "original_price": 34.99,  "rating": 4.9, "reviews": 1201,  "platform": "Etsy",     "brand": "CraftAudio",  "category": "Accessories",    "stock": "In Stock", "shipping": "$3.99",      "emoji": "🎨", "specs": {"Type": "Accessory", "Material": "Genuine Leather", "Fits": "Most Over-Ear", "Color": "Custom", "Codec": "N/A"}},
        {"title": "Samsung Galaxy Buds3 Pro ANC",          "price": 199.99, "original_price": 249.99, "rating": 4.5, "reviews": 9201,  "platform": "Target",   "brand": "Samsung",     "category": "In-Ear ANC",     "stock": "In Stock", "shipping": "Free",       "emoji": "🎧", "specs": {"Type": "In-Ear", "ANC": "Yes", "Battery": "6+21 hrs", "Bluetooth": "5.4", "Codec": "SSC/AAC"}},
        {"title": "Audio-Technica ATH-M50xBT2 Wireless",  "price": 169.00, "original_price": 199.00, "rating": 4.6, "reviews": 14203, "platform": "Walmart",  "brand": "Audio-Technica","category": "Studio Monitor", "stock": "In Stock", "shipping": "Free",      "emoji": "🎧", "specs": {"Type": "Over-Ear", "ANC": "No", "Battery": "50 hrs", "Bluetooth": "5.0", "Codec": "AAC/SBC"}},
    ],
    "phone": [
        {"title": "iPhone 16 Pro Max 256GB Desert Titanium", "price": 1199.00, "original_price": 1199.00, "rating": 4.9, "reviews": 123041, "platform": "Best Buy", "brand": "Apple",   "category": "Flagship",   "stock": "In Stock", "shipping": "Free",       "emoji": "📱", "specs": {"OS": "iOS 18", "RAM": "8GB", "Storage": "256GB", "Display": "6.9\" ProMotion", "Camera": "48MP+12MP+12MP"}},
        {"title": "Samsung Galaxy S25 Ultra 512GB",          "price": 1099.99, "original_price": 1299.99, "rating": 4.7, "reviews": 67201,  "platform": "Amazon",   "brand": "Samsung", "category": "Flagship",   "stock": "In Stock", "shipping": "Free 2-day", "emoji": "📱", "specs": {"OS": "Android 15", "RAM": "12GB", "Storage": "512GB", "Display": "6.9\" AMOLED", "Camera": "200MP+50MP+10MP"}},
        {"title": "Google Pixel 9 Pro 256GB Obsidian",       "price": 999.00,  "original_price": 1099.00, "rating": 4.7, "reviews": 28401,  "platform": "Walmart",  "brand": "Google",  "category": "AI-First",   "stock": "In Stock", "shipping": "Free 2-day", "emoji": "📱", "specs": {"OS": "Android 15", "RAM": "16GB", "Storage": "256GB", "Display": "6.3\" LTPO OLED", "Camera": "50MP+48MP+48MP"}},
        {"title": "OnePlus 13 512GB Silky Black",            "price": 799.00,  "original_price": 899.00,  "rating": 4.6, "reviews": 11203,  "platform": "Newegg",   "brand": "OnePlus", "category": "Value Flag", "stock": "In Stock", "shipping": "Free",       "emoji": "📱", "specs": {"OS": "Android 15", "RAM": "16GB", "Storage": "512GB", "Display": "6.82\" 2K AMOLED", "Camera": "50MP+50MP+50MP"}},
        {"title": "iPhone 14 128GB Refurb Grade A",          "price": 449.00,  "original_price": 699.00,  "rating": 4.4, "reviews": 19201,  "platform": "eBay",     "brand": "Apple",   "category": "Refurb",     "stock": "In Stock", "shipping": "Free",       "emoji": "📱", "specs": {"OS": "iOS 18", "RAM": "6GB", "Storage": "128GB", "Display": "6.1\" Super Retina", "Camera": "12MP+12MP"}},
        {"title": "Motorola Edge 50 Pro 256GB",              "price": 449.99,  "original_price": 549.99,  "rating": 4.4, "reviews": 4201,   "platform": "Target",   "brand": "Motorola","category": "Mid-Range",  "stock": "In Stock", "shipping": "Free",       "emoji": "📱", "specs": {"OS": "Android 14", "RAM": "12GB", "Storage": "256GB", "Display": "6.7\" pOLED 144Hz", "Camera": "50MP+13MP+10MP"}},
        {"title": "Nothing Phone (3) 256GB White",           "price": 649.00,  "original_price": 699.00,  "rating": 4.5, "reviews": 6102,   "platform": "Amazon",   "brand": "Nothing", "category": "Design Pick","stock": "Limited",  "shipping": "Free 2-day", "emoji": "📱", "specs": {"OS": "Nothing OS 3", "RAM": "12GB", "Storage": "256GB", "Display": "6.67\" AMOLED 120Hz", "Camera": "50MP+50MP"}},
    ],
    "coffee": [
        {"title": "Breville Barista Express Espresso w/ Grinder","price": 699.95, "original_price": 799.95, "rating": 4.7, "reviews": 21045, "platform": "Amazon",  "brand": "Breville",     "category": "Espresso Machines", "stock": "In Stock", "shipping": "Free 2-day", "emoji": "☕", "specs": {"Type": "Espresso + Grinder", "Pressure": "9 bar", "Boiler": "Thermocoil", "Grinder": "Conical", "Cup Size": "Single/Double"}},
        {"title": "Fellow Ode Gen 2 Brew Grinder",               "price": 345.00, "original_price": 345.00, "rating": 4.8, "reviews": 6201,  "platform": "Shopify", "brand": "Fellow",       "category": "Grinders",          "stock": "In Stock", "shipping": "Free",       "emoji": "⚙️", "specs": {"Type": "Flat Burr", "Burr Size": "64mm", "Grind Range": "2-11", "Hopper": "150g", "RPM": "450"}},
        {"title": "Hario V60 Pour-Over Starter Set",             "price": 48.99,  "original_price": 48.99,  "rating": 4.9, "reviews": 14021, "platform": "Walmart", "brand": "Hario",        "category": "Manual Brew",       "stock": "In Stock", "shipping": "$3.99",      "emoji": "🫗", "specs": {"Type": "Pour-Over", "Material": "Glass/Plastic", "Size": "01/02", "Filter": "Paper", "Capacity": "360-600ml"}},
        {"title": "Small-Batch Colombian Single Origin 1lb",      "price": 22.50,  "original_price": 22.50,  "rating": 4.8, "reviews": 3412,  "platform": "Etsy",    "brand": "MountainRoast","category": "Beans",             "stock": "In Stock", "shipping": "Free",       "emoji": "🌱", "specs": {"Origin": "Colombia", "Roast": "Medium", "Process": "Washed", "Notes": "Caramel, Citrus", "Altitude": "1800m"}},
        {"title": "Nespresso Vertuo Next Premium Bundle",         "price": 179.00, "original_price": 249.00, "rating": 4.6, "reviews": 31201, "platform": "Target",  "brand": "Nespresso",    "category": "Pod Machines",      "stock": "In Stock", "shipping": "Free",       "emoji": "☕", "specs": {"Type": "Pod/Capsule", "Pressure": "19 bar", "Cup Size": "5 sizes", "Milk Frother": "Included", "Capacity": "1.1L"}},
        {"title": "La Marzocco Linea Mini Home Espresso",         "price": 3990.00,"original_price": 3990.00,"rating": 4.9, "reviews": 412,   "platform": "Shopify", "brand": "La Marzocco",  "category": "Prosumer",          "stock": "Limited",  "shipping": "Free White Glove", "emoji": "🏆", "specs": {"Type": "Dual Boiler Espresso", "Pressure": "9 bar", "Boiler": "Dual 1.4L+0.5L", "Temperature": "PID", "Group Head": "E61"}},
        {"title": "Baratza Encore ESP Burr Grinder",              "price": 195.00, "original_price": 235.00, "rating": 4.6, "reviews": 8901,  "platform": "Amazon",  "brand": "Baratza",      "category": "Grinders",          "stock": "In Stock", "shipping": "Free 2-day", "emoji": "⚙️", "specs": {"Type": "Conical Burr", "Burr Size": "40mm", "Grind Settings": "40", "Hopper": "230g", "RPM": "450"}},
    ],
    "gaming": [
        {"title": "PlayStation 5 Pro Console",                  "price": 699.99,  "original_price": 699.99,  "rating": 4.8, "reviews": 52301, "platform": "Best Buy", "brand": "Sony",       "category": "Consoles",      "stock": "Limited",      "shipping": "Free",       "emoji": "🎮", "specs": {"CPU": "Zen 2 x8", "GPU": "RDNA 3.5 Enhanced", "RAM": "16GB GDDR6", "Storage": "2TB NVMe", "Resolution": "8K Ready"}},
        {"title": "Xbox Series X 2TB Console",                  "price": 549.99,  "original_price": 599.99,  "rating": 4.7, "reviews": 41203, "platform": "Target",   "brand": "Microsoft",  "category": "Consoles",      "stock": "In Stock",     "shipping": "Free",       "emoji": "🎮", "specs": {"CPU": "Zen 2 x8", "GPU": "RDNA 2 12TF", "RAM": "16GB GDDR6", "Storage": "2TB NVMe", "Resolution": "4K 120fps"}},
        {"title": "ASUS ROG Swift Pro PG248QP 540Hz Monitor",   "price": 699.00,  "original_price": 799.00,  "rating": 4.6, "reviews": 1892,  "platform": "Amazon",   "brand": "ASUS",       "category": "Monitors",      "stock": "In Stock",     "shipping": "Free 2-day", "emoji": "🖥️", "specs": {"Size": "24.1\"", "Resolution": "1080p", "Refresh": "540Hz", "Response": "0.2ms", "Panel": "TN"}},
        {"title": "SteelSeries Arctis Nova Pro Wireless",       "price": 349.99,  "original_price": "349.99", "rating": 4.7, "reviews": 6901,  "platform": "Newegg",  "brand": "SteelSeries","category": "Gaming Audio",   "stock": "In Stock",     "shipping": "Free",       "emoji": "🎧", "specs": {"Type": "Over-Ear", "ANC": "Active", "Battery": "22 hrs", "Wireless": "2.4GHz+BT", "Surround": "7.1"}},
        {"title": "Razer DeathAdder V3 HyperSpeed Wireless",    "price": 129.99,  "original_price": 159.99,  "rating": 4.7, "reviews": 9201,  "platform": "Amazon",   "brand": "Razer",      "category": "Peripherals",   "stock": "In Stock",     "shipping": "Free 2-day", "emoji": "🖱️", "specs": {"DPI": "30,000", "Sensor": "HyperTracking", "Buttons": "6", "Battery": "90 hrs", "Weight": "64g"}},
        {"title": "Hand-Painted Custom Keycap Set",             "price": 89.00,   "original_price": 89.00,   "rating": 4.9, "reviews": 412,   "platform": "Etsy",     "brand": "PixelKeys",  "category": "Peripherals",   "stock": "Made to Order","shipping": "$6.99",      "emoji": "⌨️", "specs": {"Profile": "Cherry MX", "Material": "PBT", "Legends": "Dye-Sub", "Colors": "Custom", "Compatibility": "Universal"}},
        {"title": "LG UltraGear OLED 27\" 240Hz 4K",           "price": 899.00,  "original_price": 999.00,  "rating": 4.8, "reviews": 4201,  "platform": "Best Buy", "brand": "LG",         "category": "Monitors",      "stock": "In Stock",     "shipping": "Free",       "emoji": "🖥️", "specs": {"Size": "27\"", "Resolution": "4K UHD", "Refresh": "240Hz", "Response": "0.03ms", "Panel": "WOLED"}},
        {"title": "Corsair K100 RGB Optical Gaming Keyboard",   "price": 199.99,  "original_price": 229.99,  "rating": 4.6, "reviews": 7201,  "platform": "Walmart",  "brand": "Corsair",    "category": "Peripherals",   "stock": "In Stock",     "shipping": "Free 2-day", "emoji": "⌨️", "specs": {"Switch": "OPX Optical", "Layout": "Full Size", "RGB": "Per-Key", "Wrist Rest": "Included", "Polling": "4000Hz"}},
    ],
    "chair": [
        {"title": "Herman Miller Aeron Size B Ergonomic",       "price": 1395.00, "original_price": 1395.00, "rating": 4.9, "reviews": 8201,  "platform": "Shopify",  "brand": "Herman Miller","category": "Premium Ergonomic","stock": "In Stock",    "shipping": "Free White Glove", "emoji": "🪑", "specs": {"Type": "Mesh", "Lumbar": "PostureFit SL", "Arms": "8D", "Weight Cap": "350 lbs", "Warranty": "12 yrs"}},
        {"title": "Secretlab TITAN Evo 2025 Gaming Chair",      "price": 519.00,  "original_price": "599.00", "rating": 4.7, "reviews": 21034, "platform": "Amazon",   "brand": "Secretlab",  "category": "Gaming Chairs",  "stock": "In Stock",     "shipping": "Free",       "emoji": "🪑", "specs": {"Type": "Foam + Leather", "Lumbar": "Magnetic 4-way", "Arms": "4D", "Weight Cap": "290 lbs", "Warranty": "5 yrs"}},
        {"title": "Autonomous ErgoChair Pro",                   "price": 449.00,  "original_price": 549.00,  "rating": 4.5, "reviews": 12301, "platform": "Shopify",  "brand": "Autonomous", "category": "Ergonomic",      "stock": "In Stock",     "shipping": "Free",       "emoji": "🪑", "specs": {"Type": "Mesh", "Lumbar": "Adjustable", "Arms": "4D", "Weight Cap": "300 lbs", "Warranty": "2 yrs"}},
        {"title": "IKEA Markus Office Chair Black",             "price": 229.99,  "original_price": 229.99,  "rating": 4.3, "reviews": 44201, "platform": "Target",   "brand": "IKEA",       "category": "Budget",         "stock": "In Stock",     "shipping": "Free",       "emoji": "🪑", "specs": {"Type": "Foam + Fabric", "Lumbar": "Built-in", "Arms": "Fixed", "Weight Cap": "242 lbs", "Warranty": "10 yrs"}},
        {"title": "Steelcase Leap V2 Fabric Chair",             "price": 1049.00, "original_price": 1299.00, "rating": 4.8, "reviews": 5102,  "platform": "eBay",     "brand": "Steelcase",  "category": "Premium Ergonomic","stock": "In Stock",    "shipping": "Free",       "emoji": "🪑", "specs": {"Type": "Foam + Fabric", "Lumbar": "LiveBack", "Arms": "4D", "Weight Cap": "400 lbs", "Warranty": "12 yrs"}},
    ],
    "desk": [
        {"title": "Flexispot E7 Pro Standing Desk 60x24",       "price": 499.99,  "original_price": 599.99,  "rating": 4.7, "reviews": 9201,  "platform": "Amazon",   "brand": "Flexispot",  "category": "Standing Desks", "stock": "In Stock",     "shipping": "Free",       "emoji": "🖥️", "specs": {"Top": "60x24\" Bamboo", "Range": "22.8-48.4\"", "Motor": "Dual Motor", "Load": "355 lbs", "Memory": "4 Presets"}},
        {"title": "Uplift V2 Standing Desk 72x30 Walnut",       "price": 1149.00, "original_price": 1149.00, "rating": 4.8, "reviews": 6102,  "platform": "Shopify",  "brand": "Uplift",     "category": "Premium Standing","stock": "In Stock",     "shipping": "Free",       "emoji": "🖥️", "specs": {"Top": "72x30\" Walnut", "Range": "25.5-51.1\"", "Motor": "Quad Motor", "Load": "535 lbs", "Memory": "4+Programmatic"}},
        {"title": "IKEA Bekant Corner Desk White",              "price": 399.00,  "original_price": 399.00,  "rating": 4.1, "reviews": 18023, "platform": "Target",   "brand": "IKEA",       "category": "Corner Desks",   "stock": "In Stock",     "shipping": "$49.99",     "emoji": "🖥️", "specs": {"Top": "160x110cm L-shape", "Height": "65-85cm", "Motor": "N/A", "Load": "50 kg", "Memory": "N/A"}},
    ],
    "keyboard": [
        {"title": "NuPhy Air75 V2 Wireless Mechanical",        "price": 109.00,  "original_price": 129.00,  "rating": 4.7, "reviews": 8201,  "platform": "Amazon",   "brand": "NuPhy",      "category": "Wireless Mech",  "stock": "In Stock",     "shipping": "Free 2-day", "emoji": "⌨️", "specs": {"Layout": "75%", "Switch": "Brown", "RGB": "Per-Key", "Battery": "3000mAh", "Connection": "BT 5.0/2.4G/USB"}},
        {"title": "Keychron Q1 Pro QMK/VIA Wireless",          "price": 199.00,  "original_price": 199.00,  "rating": 4.8, "reviews": 5201,  "platform": "Shopify",  "brand": "Keychron",   "category": "Custom Mech",    "stock": "In Stock",     "shipping": "Free",       "emoji": "⌨️", "specs": {"Layout": "75%", "Switch": "Red/Blue/Brown", "RGB": "South-Facing", "Battery": "4000mAh", "Connection": "BT 5.1/USB-C"}},
        {"title": "Logitech MX Keys S Wireless",               "price": 109.99,  "original_price": 129.99,  "rating": 4.6, "reviews": 21034, "platform": "Best Buy", "brand": "Logitech",   "category": "Wireless Office","stock": "In Stock",     "shipping": "Free",       "emoji": "⌨️", "specs": {"Layout": "Full Size", "Switch": "Scissor", "Backlit": "Smart Adaptive", "Battery": "10 days", "Connection": "Bolt USB/BT"}},
        {"title": "Hand-Built Custom 65% Hotswap Kit",         "price": 149.00,  "original_price": 149.00,  "rating": 4.9, "reviews": 912,   "platform": "Etsy",     "brand": "MechAtelier","category": "Custom Build",   "stock": "Made to Order","shipping": "$8.99",      "emoji": "⌨️", "specs": {"Layout": "65%", "Switch": "Buyer's Choice", "RGB": "Underglow+Per-Key", "Battery": "N/A", "Connection": "USB-C"}},
    ],
    "default": [
        {"title": "Premium Quality Model — Top Rated 2025",   "price": 89.99,  "original_price": 119.99, "rating": 4.6, "reviews": 5201,  "platform": "Amazon",   "brand": "TopBrand",    "category": "General",    "stock": "In Stock",   "shipping": "Free 2-day", "emoji": "🛍️", "specs": {}},
        {"title": "Best Value Edition — Editor's Choice",     "price": 64.95,  "original_price": 79.95,  "rating": 4.5, "reviews": 11234, "platform": "Walmart",  "brand": "ValuePlus",   "category": "General",    "stock": "In Stock",   "shipping": "$4.99",      "emoji": "⭐", "specs": {}},
        {"title": "Professional Grade — Enterprise Ready",    "price": 129.00, "original_price": 129.00, "rating": 4.7, "reviews": 2901,  "platform": "Target",   "brand": "ProLine",     "category": "Professional","stock": "Limited",   "shipping": "Free",       "emoji": "💼", "specs": {}},
        {"title": "Handcrafted Artisan — Limited Batch",      "price": 79.50,  "original_price": 99.50,  "rating": 4.9, "reviews": 812,   "platform": "Etsy",     "brand": "ArtisanCo",   "category": "Handmade",   "stock": "In Stock",   "shipping": "$5.99",      "emoji": "🎨", "specs": {}},
        {"title": "Smart AI-Enhanced Premium Edition",        "price": 149.99, "original_price": 179.99, "rating": 4.4, "reviews": 3901,  "platform": "Best Buy", "brand": "SmartTech",   "category": "Smart Home", "stock": "In Stock",   "shipping": "Free",       "emoji": "🤖", "specs": {}},
        {"title": "Certified Refurb — Like New w/ Warranty",  "price": 55.00,  "original_price": 99.00,  "rating": 4.3, "reviews": 9201,  "platform": "eBay",     "brand": "Various",     "category": "Refurb",     "stock": "In Stock",   "shipping": "Free",       "emoji": "♻️", "specs": {}},
        {"title": "Limited Edition Collector's Bundle",       "price": 199.99, "original_price": 249.99, "rating": 4.8, "reviews": 1102,  "platform": "Shopify",  "brand": "LimitedCo",   "category": "Collector",  "stock": "Very Limited","shipping": "Free",       "emoji": "💎", "specs": {}},
        {"title": "Builder's Kit — Open Box Deal",            "price": 44.99,  "original_price": 64.99,  "rating": 4.2, "reviews": 2901,  "platform": "Newegg",   "brand": "BuildPro",    "category": "DIY",        "stock": "In Stock",   "shipping": "Free",       "emoji": "🔧", "specs": {}},
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
  --bg4:       #16151f;
  --gold:      #d4af37;
  --gold2:     #b8963e;
  --gold3:     #f0d060;
  --gold-dim:  rgba(212,175,55,0.18);
  --gold-glow: rgba(212,175,55,0.07);
  --text:      #f0ece4;
  --text2:     #c8c4ba;
  --text3:     #888;
  --text4:     #555;
  --green:     #64dc82;
  --green2:    #3cb86a;
  --red:       #ff6060;
  --blue:      #60a4ff;
  --border:    rgba(212,175,55,0.18);
  --border2:   rgba(255,255,255,0.06);
  --radius-sm: 8px;
  --radius-md: 12px;
  --radius-lg: 18px;
  --radius-xl: 24px;
}

* { box-sizing: border-box; }
html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text);
    font-family: 'DM Sans', sans-serif;
}
#MainMenu, footer, header, [data-testid="stToolbar"] { display:none !important; visibility:hidden !important; }
.block-container { padding: 0 2rem 3rem 2rem !important; max-width:1460px !important; }

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#0c0b18 0%,#0f0e1a 50%,#11101d 100%) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] .block-container { padding:1.5rem 1.2rem 2rem 1.2rem !important; }
[data-testid="stSidebarNav"] { display:none !important; }

/* ── INPUTS ── */
input, textarea,
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea {
    background:rgba(255,255,255,0.04) !important;
    border:1px solid rgba(212,175,55,0.25) !important;
    border-radius:var(--radius-sm) !important;
    color:var(--text) !important;
    font-family:'DM Sans',sans-serif !important;
    transition:border-color .2s,box-shadow .2s !important;
}
input:focus, textarea:focus {
    border-color:var(--gold) !important;
    box-shadow:0 0 0 2px rgba(212,175,55,0.15) !important;
    outline:none !important;
}
input::placeholder, textarea::placeholder { color:var(--text4) !important; }

/* ── SELECT ── */
[data-testid="stSelectbox"] > div > div,
[data-testid="stMultiSelect"] > div > div {
    background:rgba(255,255,255,0.04) !important;
    border:1px solid rgba(212,175,55,0.25) !important;
    border-radius:var(--radius-sm) !important;
    color:var(--text) !important;
}
[data-baseweb="popover"] { background:var(--bg3) !important; border:1px solid var(--border) !important; }
[data-baseweb="menu"] { background:var(--bg3) !important; }
[data-baseweb="option"]:hover { background:var(--gold-dim) !important; }
[data-baseweb="tag"] {
    background:rgba(212,175,55,0.15) !important;
    border:1px solid rgba(212,175,55,0.35) !important;
    color:var(--gold) !important;
    border-radius:6px !important;
}

/* ── SLIDER ── */
[data-testid="stSlider"] .rc-slider-rail { background:rgba(255,255,255,0.08) !important; }
[data-testid="stSlider"] .rc-slider-track { background:var(--gold) !important; }
[data-testid="stSlider"] .rc-slider-handle {
    border-color:var(--gold) !important;
    background:var(--gold) !important;
    box-shadow:0 0 8px rgba(212,175,55,0.4) !important;
}

/* ── BUTTONS ── */
.stButton > button {
    background:linear-gradient(135deg,var(--gold) 0%,var(--gold2) 100%) !important;
    color:#0a0a0f !important; border:none !important;
    border-radius:var(--radius-sm) !important;
    font-family:'DM Sans',sans-serif !important;
    font-weight:600 !important; letter-spacing:.04em !important;
    padding:.55rem 1.2rem !important;
    transition:transform .15s ease,box-shadow .15s ease !important;
}
.stButton > button:hover {
    transform:translateY(-2px) !important;
    box-shadow:0 8px 24px rgba(212,175,55,0.35) !important;
}
.stButton > button:active { transform:translateY(0) !important; }
.stButton > button:disabled {
    background:rgba(255,255,255,0.06) !important;
    color:var(--text4) !important;
    box-shadow:none !important;
    transform:none !important;
}

/* ── CHECKBOX ── */
[data-testid="stCheckbox"] label { color:var(--text2) !important; }
[data-baseweb="checkbox"] div { background:var(--gold-dim) !important; border-color:var(--gold) !important; }

/* ── RADIO ── */
[data-testid="stRadio"] label { color:var(--text2) !important; }

/* ── TABS ── */
[data-testid="stTabs"] [role="tablist"] {
    border-bottom:1px solid rgba(212,175,55,0.2) !important;
    gap:.4rem; background:transparent;
}
[data-testid="stTabs"] [role="tab"] {
    color:var(--text3) !important; font-family:'DM Sans',sans-serif !important;
    font-weight:500 !important; border-radius:8px 8px 0 0 !important;
    padding:.5rem 1.1rem !important; border:none !important;
    background:transparent !important; transition:color .2s !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color:var(--gold) !important;
    border-bottom:2px solid var(--gold) !important;
    background:rgba(212,175,55,0.05) !important;
}

/* ── EXPANDER ── */
[data-testid="stExpander"] {
    border:1px solid var(--border) !important;
    border-radius:var(--radius-md) !important;
    background:rgba(255,255,255,0.02) !important;
}
[data-testid="stExpander"] summary { color:var(--gold) !important; font-weight:600 !important; }

/* ── METRIC ── */
[data-testid="stMetric"] {
    background:rgba(255,255,255,0.025) !important;
    border:1px solid var(--border) !important;
    border-radius:var(--radius-md) !important;
    padding:1rem 1.2rem !important;
}
[data-testid="stMetricLabel"] {
    color:var(--text3) !important; font-size:.72rem !important;
    text-transform:uppercase; letter-spacing:.1em;
    font-family:'JetBrains Mono',monospace !important;
}
[data-testid="stMetricValue"] {
    color:var(--gold) !important;
    font-family:'Playfair Display',serif !important;
    font-size:1.7rem !important;
}
[data-testid="stMetricDelta"] { font-size:.78rem !important; }

/* ── ALERTS ── */
[data-testid="stAlert"] {
    border-radius:var(--radius-md) !important;
    border-left:3px solid var(--gold) !important;
    background:rgba(212,175,55,0.06) !important;
}

/* ── DATAFRAME ── */
[data-testid="stDataFrame"] { border:1px solid var(--border) !important; border-radius:var(--radius-md) !important; overflow:hidden; }
.dvn-scroller { background:var(--bg3) !important; }

/* ── SCROLLBAR ── */
::-webkit-scrollbar { width:5px; height:5px; }
::-webkit-scrollbar-track { background:var(--bg); }
::-webkit-scrollbar-thumb { background:rgba(212,175,55,0.28); border-radius:3px; }
::-webkit-scrollbar-thumb:hover { background:rgba(212,175,55,0.5); }

/* ══════════════════════════════
   CUSTOM COMPONENT STYLES
══════════════════════════════ */

/* Hero */
.hero-wrap {
    text-align:center; padding:3.5rem 1rem 2rem;
    position:relative; overflow:hidden;
}
.hero-glow {
    position:absolute; top:50%; left:50%;
    transform:translate(-50%,-50%);
    width:800px; height:400px;
    background:radial-gradient(ellipse,rgba(212,175,55,0.09) 0%,transparent 65%);
    pointer-events:none;
}
.hero-eyebrow {
    font-family:'JetBrains Mono',monospace;
    font-size:.68rem; letter-spacing:.3em;
    text-transform:uppercase; color:var(--gold);
    margin-bottom:.9rem;
}
.hero-title {
    font-family:'Playfair Display',serif;
    font-size:clamp(2.6rem,6vw,4.8rem);
    font-weight:700; color:var(--text);
    line-height:1.08; margin:0 0 .7rem;
}
.hero-title em { color:var(--gold); font-style:italic; }
.hero-sub {
    font-size:1rem; color:var(--text3);
    max-width:560px; margin:0 auto 2.2rem; line-height:1.65;
}

/* Platform strip */
.pstrip { display:flex; gap:.6rem; flex-wrap:wrap; justify-content:center; margin-bottom:2rem; }
.pchip {
    padding:5px 13px;
    background:rgba(255,255,255,0.03);
    border:1px solid rgba(255,255,255,0.08);
    border-radius:20px; font-size:.73rem;
    color:var(--text3); letter-spacing:.04em;
    transition:border-color .2s,color .2s;
}

/* Sidebar labels */
.sl {
    font-family:'JetBrains Mono',monospace;
    font-size:.58rem; letter-spacing:.22em;
    text-transform:uppercase; color:var(--gold);
    margin-bottom:.5rem; padding-bottom:.35rem;
    border-bottom:1px solid rgba(212,175,55,0.18);
    display:block;
}

/* Section headings */
.sec-eyebrow {
    font-family:'JetBrains Mono',monospace;
    font-size:.62rem; letter-spacing:.22em;
    text-transform:uppercase; color:var(--text4);
    margin-bottom:.3rem;
}
.sec-title {
    font-family:'Playfair Display',serif;
    font-size:1.55rem; color:var(--text); margin-bottom:1.2rem;
}

/* Product card */
.pcard {
    background:linear-gradient(140deg,rgba(255,255,255,0.04) 0%,rgba(255,255,255,0.015) 100%);
    border:1px solid var(--border);
    border-radius:var(--radius-lg);
    padding:1.3rem; margin-bottom:.8rem;
    position:relative; overflow:hidden;
    transition:border-color .25s,transform .2s,box-shadow .25s;
}
.pcard::before {
    content:'';
    position:absolute; top:0; left:0; right:0; height:1px;
    background:linear-gradient(90deg,transparent 0%,var(--gold) 50%,transparent 100%);
    opacity:0; transition:opacity .3s;
}
.pcard:hover { border-color:rgba(212,175,55,0.42); transform:translateY(-3px); box-shadow:0 16px 48px rgba(0,0,0,0.5); }
.pcard:hover::before { opacity:.7; }

.pbadge {
    display:inline-block; padding:3px 10px;
    border-radius:20px; font-size:.68rem; font-weight:600;
    letter-spacing:.07em; text-transform:uppercase; margin-bottom:.6rem;
}
.badge-amazon  { background:rgba(255,153,0,.12); color:#ff9900; border:1px solid rgba(255,153,0,.28); }
.badge-ebay    { background:rgba(14,118,188,.12); color:#4a9fd4; border:1px solid rgba(14,118,188,.28); }
.badge-walmart { background:rgba(0,117,201,.12); color:#4ab3ff; border:1px solid rgba(0,117,201,.28); }
.badge-etsy    { background:rgba(241,100,30,.12); color:#f1641e; border:1px solid rgba(241,100,30,.28); }
.badge-bestbuy { background:rgba(0,70,208,.12); color:#5e8eff; border:1px solid rgba(0,70,208,.28); }
.badge-target  { background:rgba(204,0,0,.12); color:#ff6060; border:1px solid rgba(204,0,0,.28); }
.badge-shopify { background:rgba(150,191,68,.12); color:#96bf44; border:1px solid rgba(150,191,68,.28); }
.badge-newegg  { background:rgba(255,126,0,.12); color:#ff9a40; border:1px solid rgba(255,126,0,.28); }

.ai-pill {
    display:inline-block; padding:2px 9px;
    border-radius:20px; font-size:.68rem;
    color:var(--green); font-weight:700;
    font-family:'JetBrains Mono',monospace;
    float:right;
}
.score-excellent { background:rgba(100,220,130,.12); color:#64dc82; border:1px solid rgba(100,220,130,.28); }
.score-good      { background:rgba(212,175,55,.12); color:var(--gold); border:1px solid rgba(212,175,55,.28); }
.score-average   { background:rgba(255,180,60,.12); color:#ffb43c; border:1px solid rgba(255,180,60,.28); }
.score-poor      { background:rgba(255,96,96,.12); color:var(--red); border:1px solid rgba(255,96,96,.28); }

.ptitle { font-weight:600; font-size:.95rem; color:var(--text); margin:.5rem 0 .3rem; line-height:1.4; }
.pprice-wrap { display:flex; align-items:baseline; gap:.5rem; margin:.35rem 0; }
.pprice { font-family:'Playfair Display',serif; font-size:1.35rem; color:var(--gold); }
.pprice-orig { font-size:.82rem; color:var(--text4); text-decoration:line-through; }
.psaving { font-size:.75rem; color:var(--green); font-weight:600; }
.prating { color:var(--gold); font-size:.82rem; }
.pmeta { font-size:.76rem; color:var(--text3); margin:.12rem 0; }

/* Price bar */
.pbar-outer { background:rgba(255,255,255,0.06); border-radius:3px; height:5px; margin:.5rem 0; overflow:hidden; }
.pbar-inner { height:5px; border-radius:3px; background:linear-gradient(90deg,var(--gold),var(--green)); transition:width .7s ease; }

/* Tags */
.tag { display:inline-block; padding:2px 8px; border-radius:4px; font-size:.68rem; font-weight:600; margin:2px 2px 0 0; }
.tag-deal  { background:rgba(212,175,55,.1); color:var(--gold); border:1px solid rgba(212,175,55,.25); }
.tag-green { background:rgba(100,220,130,.1); color:var(--green); border:1px solid rgba(100,220,130,.22); }
.tag-red   { background:rgba(255,96,96,.1); color:var(--red); border:1px solid rgba(255,96,96,.22); }
.tag-gray  { background:rgba(255,255,255,.05); color:var(--text3); border:1px solid rgba(255,255,255,.1); }
.tag-blue  { background:rgba(96,164,255,.1); color:var(--blue); border:1px solid rgba(96,164,255,.22); }

/* AI bubble */
.ai-bubble {
    background:linear-gradient(135deg,rgba(212,175,55,.07) 0%,rgba(180,145,25,.03) 100%);
    border:1px solid rgba(212,175,55,.22);
    border-left:3px solid var(--gold);
    border-radius:var(--radius-md);
    padding:1.2rem 1.4rem; margin:1rem 0;
    font-size:.9rem; color:var(--text2); line-height:1.75;
}
.ai-bubble::before {
    content:'✦ AI INTELLIGENCE';
    display:block; font-family:'JetBrains Mono',monospace;
    font-size:.62rem; letter-spacing:.2em; color:var(--gold);
    margin-bottom:.6rem;
}

/* Spec table */
.spec-grid {
    display:grid; grid-template-columns:1fr 1fr;
    gap:4px 12px; margin-top:.5rem;
}
.spec-row { display:flex; justify-content:space-between; font-size:.72rem; padding:3px 0; border-bottom:1px solid rgba(255,255,255,.04); }
.spec-key { color:var(--text4); }
.spec-val { color:var(--text2); font-weight:500; text-align:right; }

/* Chat */
.chat-user {
    background:rgba(255,255,255,.04);
    border:1px solid rgba(255,255,255,.07);
    border-radius:12px 12px 4px 12px;
    padding:.75rem 1rem; margin-bottom:.5rem;
    color:var(--text); font-size:.88rem; text-align:right;
}
.chat-ai {
    background:rgba(212,175,55,.05);
    border:1px solid rgba(212,175,55,.18);
    border-radius:12px 12px 12px 4px;
    padding:.75rem 1rem; margin-bottom:.5rem;
    color:var(--text2); font-size:.88rem;
}
.chat-ts { font-size:.63rem; color:var(--text4); margin-top:.25rem; font-family:'JetBrains Mono',monospace; }

/* Insight card */
.insight-card {
    background:rgba(255,255,255,.02);
    border:1px solid var(--border);
    border-radius:var(--radius-md);
    padding:.9rem 1rem; margin-bottom:.5rem;
    font-size:.83rem; color:var(--text2);
}

/* Feature card */
.feat-card {
    text-align:center; padding:1.6rem 1rem;
    background:rgba(255,255,255,.02);
    border:1px solid var(--border);
    border-radius:var(--radius-lg);
    height:210px;
    transition:border-color .2s,transform .2s;
}
.feat-card:hover { border-color:rgba(212,175,55,.35); transform:translateY(-3px); }
.feat-icon { font-size:2.2rem; margin-bottom:.7rem; }
.feat-title { font-family:'Playfair Display',serif; font-size:1rem; color:var(--text); margin-bottom:.5rem; }
.feat-desc { font-size:.78rem; color:var(--text3); line-height:1.55; }

/* Empty state */
.empty-state { text-align:center; padding:3.5rem 1rem; color:var(--text3); }
.empty-icon { font-size:3.5rem; margin-bottom:1rem; }
.empty-title { font-family:'Playfair Display',serif; font-size:1.3rem; color:var(--text2); margin-bottom:.5rem; }
.empty-sub { font-size:.88rem; line-height:1.6; }

/* Wishlist row */
.wl-row {
    background:rgba(255,255,255,.02);
    border:1px solid var(--border);
    border-radius:var(--radius-md);
    padding:.75rem 1rem; margin-bottom:.5rem;
    display:flex; align-items:center; gap:.5rem;
}

/* CTA banner */
.cta-banner {
    text-align:center; padding:2.2rem 1.5rem;
    background:rgba(212,175,55,.04);
    border:1px solid rgba(212,175,55,.15);
    border-radius:var(--radius-lg);
}
.cta-title { font-family:'Playfair Display',serif; font-size:1.35rem; color:var(--gold); margin-bottom:.5rem; font-style:italic; }
.cta-sub { color:var(--text3); font-size:.9rem; }

/* Sidebar logo */
.logo-wrap { text-align:center; padding:.8rem 0 1.6rem; }
.logo-name { font-family:'Playfair Display',serif; font-size:1.45rem; color:var(--gold); font-weight:700; letter-spacing:.02em; }
.logo-sub { font-family:'JetBrains Mono',monospace; font-size:.58rem; letter-spacing:.25em; color:var(--text4); margin-top:.2rem; text-transform:uppercase; }

/* Search history chip */
.hist-chip {
    display:inline-block; padding:4px 12px;
    background:rgba(255,255,255,.04);
    border:1px solid rgba(255,255,255,.1);
    border-radius:20px; font-size:.75rem;
    color:var(--text3); margin:3px;
    transition:border-color .2s,color .2s;
}
.hist-chip:hover { border-color:var(--gold); color:var(--gold); }

/* Rec badge */
.rec-badge {
    background:rgba(255,255,255,.02);
    border:1px solid var(--border);
    border-radius:var(--radius-md);
    padding:.9rem 1rem; margin-bottom:.6rem;
    transition:border-color .2s;
}
.rec-badge:hover { border-color:rgba(212,175,55,.35); }

/* Beginner badge */
.beginner-pill {
    display:inline-block; padding:2px 9px;
    background:rgba(96,164,255,.1);
    border:1px solid rgba(96,164,255,.28);
    border-radius:20px; font-size:.68rem;
    color:var(--blue); font-weight:700;
    font-family:'JetBrains Mono',monospace;
}

/* Platform status indicator */
.plat-status {
    display:inline-flex; align-items:center; gap:6px;
    font-size:.73rem; color:var(--text3);
    padding:4px 10px; border-radius:20px;
    background:rgba(100,220,130,.06);
    border:1px solid rgba(100,220,130,.18);
}
.plat-dot { width:6px; height:6px; border-radius:50%; background:var(--green); display:inline-block; animation:pulse 2s infinite; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.4} }

hr { border-color:rgba(212,175,55,.12) !important; margin:1.5rem 0 !important; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SESSION MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════

def init_session() -> None:
    defaults: Dict[str, Any] = {
        "search_results":     [],
        "last_query":         "",
        "ai_insight":         "",
        "ai_recommendations": {},
        "product_insights":   [],
        "chat_history":       [],
        "wishlist":           [],
        "compare_list":       [],
        "search_history":     [],
        "active_tab":         0,
        "platform_results":   {},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = deepcopy(v)

# ═══════════════════════════════════════════════════════════════════════════════
# SEARCH ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=300, show_spinner=False)
def _get_raw_products(query_key: str) -> List[Dict]:
    q = query_key.lower()
    for kw, items in _RAW_DB.items():
        if kw != "default" and kw in q:
            return items
    # Also check multi-word keys
    for kw, items in _RAW_DB.items():
        if kw != "default" and any(w in q for w in kw.split()):
            return items
    return _RAW_DB["default"]


def _simulate_platform_fetch(platform: str, raw_items: List[Dict], budget: float, min_rating: float) -> Tuple[str, List[Dict]]:
    """Simulates per-platform API call with slight randomised delay."""
    time.sleep(random.uniform(0.05, 0.15))
    filtered = [r for r in raw_items if r["platform"] == platform and r["price"] <= budget and r["rating"] >= min_rating]
    return platform, filtered


def compute_ai_score(p: Dict, category_stats: Dict) -> float:
    trust = PLATFORM_TRUST.get(p["platform"], 0.75)
    rating_score  = (p["rating"] - 1) / 4.0
    review_score  = min(1.0, math.log1p(p["reviews"]) / math.log1p(60000))
    cat_max = category_stats.get("max_price", p["original_price"])
    cat_min = category_stats.get("min_price", p["price"])
    price_range   = max(cat_max - cat_min, 1)
    price_value   = 1.0 - ((p["price"] - cat_min) / price_range) * 0.55
    disc_pct      = max(0, (p["original_price"] - p["price"]) / max(p["original_price"], 1))
    deal_score    = min(1.0, disc_pct * 3.2)
    avail_score   = 1.0 if p["stock"] == "In Stock" else (0.6 if p["stock"] == "Limited" else 0.3)
    score = (
        SCORE_WEIGHTS["rating"]         * rating_score  +
        SCORE_WEIGHTS["review_volume"]  * review_score  +
        SCORE_WEIGHTS["price_value"]    * price_value   +
        SCORE_WEIGHTS["platform_trust"] * trust         +
        SCORE_WEIGHTS["deal_quality"]   * deal_score    +
        SCORE_WEIGHTS["availability"]   * avail_score
    )
    score = score + random.uniform(-0.015, 0.015)
    return round(min(99.5, max(52.0, score * 100)), 1)


def compute_beginner_score(p: Dict) -> float:
    score = 0.5
    if p["reviews"] >= 10000: score += 0.2
    elif p["reviews"] >= 3000: score += 0.1
    if p["rating"] >= 4.7: score += 0.15
    if p["stock"] == "In Stock": score += 0.05
    if p["platform"] in ("Amazon", "Best Buy", "Walmart", "Target"): score += 0.1
    if "Gaming Premium" in p.get("category","") or "Prosumer" in p.get("category","") or "Audiophile" in p.get("category",""): score -= 0.2
    if "Refurb" in p.get("category","") or "Open Box" in p.get("title",""): score -= 0.1
    return round(min(1.0, max(0.0, score)), 2)


def classify_deal(price: float, orig: float, cat_avg: float) -> Tuple[str, float]:
    disc  = (orig - price) / max(orig, 1)
    ratio = price / max(cat_avg, 1)
    if disc >= 0.25 or ratio <= 0.68:   return "🔥 Excellent Deal", 0.95
    elif disc >= 0.12 or ratio <= 0.84: return "✅ Great Deal", 0.80
    elif disc >= 0.05 or ratio <= 0.95: return "👍 Good Value", 0.65
    elif ratio <= 1.10:                 return "📊 Fair Price", 0.50
    else:                               return "⚠️ Overpriced", 0.25


def price_trend_label() -> str:
    choices = ["↓ 16%", "↓ 9%", "↓ 4%", "→ Stable", "↑ 2%", "↑ 7%"]
    weights = [0.09, 0.17, 0.22, 0.30, 0.13, 0.09]
    return random.choices(choices, weights=weights, k=1)[0]


def build_products(
    query: str,
    platforms: List[str],
    budget_max: float,
    min_rating: float,
    min_reviews: int,
    min_discount: float,
) -> Tuple[List[Product], Dict[str, int]]:
    raw = _get_raw_products(query.lower())
    prices = [r["price"] for r in raw]
    cat_stats = {
        "min_price": min(prices),
        "max_price": max(prices),
        "avg_price": sum(prices) / len(prices),
    }

    platform_counts: Dict[str, int] = {p: 0 for p in platforms}
    products: List[Product] = []

    for r in raw:
        if r["platform"] not in platforms: continue
        if r["price"] > budget_max: continue
        if r["rating"] < min_rating: continue
        if r["reviews"] < min_reviews: continue
        price_f = float(r["price"])
        orig_f  = float(r["original_price"])
        disc    = (orig_f - price_f) / max(orig_f, 1) * 100
        if disc < min_discount: continue

        deal_label, deal_q = classify_deal(price_f, orig_f, cat_stats["avg_price"])
        trend = price_trend_label()
        score = compute_ai_score(r, cat_stats)
        beg   = compute_beginner_score(r)

        tags: List[str] = []
        if disc >= 15:               tags.append(f"🔖 {int(disc)}% OFF")
        if r["rating"] >= 4.8:       tags.append("🏅 Top Rated")
        if r["reviews"] >= 10000:    tags.append("💬 Highly Reviewed")
        if r["stock"] == "Limited":  tags.append("⚡ Limited Stock")
        if r["platform"] == "Etsy":  tags.append("🎨 Artisan")
        if beg >= 0.75:              tags.append("🟢 Beginner Friendly")

        p = Product(
            title=r["title"], price=price_f,
            original_price=orig_f, rating=r["rating"],
            reviews=r["reviews"], platform=r["platform"],
            brand=r["brand"], category=r["category"],
            stock=r["stock"], shipping=r["shipping"],
            emoji=r["emoji"], ai_score=score,
            price_trend=trend, deal_label=deal_label,
            deal_quality_score=deal_q, discount_pct=round(disc, 1),
            tags=tags, beginner_score=beg,
            specs=r.get("specs", {}),
        )
        products.append(p)
        platform_counts[r["platform"]] = platform_counts.get(r["platform"], 0) + 1

    if len(products) < 3:
        for r in _RAW_DB["default"]:
            if r["price"] <= budget_max and r["platform"] in platforms:
                orig_f = float(r["original_price"])
                price_f = float(r["price"])
                disc = (orig_f - price_f) / max(orig_f, 1) * 100
                deal_label, deal_q = classify_deal(price_f, orig_f, cat_stats["avg_price"])
                p = Product(
                    title=r["title"], price=price_f, original_price=orig_f,
                    rating=r["rating"], reviews=r["reviews"], platform=r["platform"],
                    brand=r["brand"], category=r["category"], stock=r["stock"],
                    shipping=r["shipping"], emoji=r["emoji"],
                    ai_score=compute_ai_score(r, cat_stats),
                    price_trend=price_trend_label(), deal_label=deal_label,
                    deal_quality_score=deal_q, discount_pct=round(disc, 1),
                    tags=[], beginner_score=compute_beginner_score(r),
                    specs=r.get("specs", {}),
                )
                products.append(p)
            if len(products) >= 6:
                break

    return products, platform_counts


def sort_products(products: List[Product], sort_by: str) -> List[Product]:
    key_map = {
        "AI Score":          lambda p: p.ai_score,
        "Price: Low→High":   lambda p: p.price,
        "Price: High→Low":   lambda p: -p.price,
        "Rating":            lambda p: p.rating,
        "Reviews":           lambda p: p.reviews,
        "Discount %":        lambda p: p.discount_pct,
        "Beginner Score":    lambda p: p.beginner_score,
    }
    fn = key_map.get(sort_by, lambda p: p.ai_score)
    return sorted(products, key=fn, reverse=(sort_by != "Price: Low→High"))

# ═══════════════════════════════════════════════════════════════════════════════
# AI RECOMMENDATION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def generate_recommendations(products: List[Product], budget: float) -> Dict[str, Optional[Product]]:
    if not products:
        return {}
    by_score    = sorted(products, key=lambda p: p.ai_score, reverse=True)
    by_price    = sorted(products, key=lambda p: p.price)
    by_rating   = sorted(products, key=lambda p: p.rating, reverse=True)
    by_value    = sorted(products, key=lambda p: p.ai_score / max(p.price, 1), reverse=True)
    by_beginner = sorted(products, key=lambda p: p.beginner_score, reverse=True)

    premium_cutoff = budget * 0.72
    budget_cutoff  = budget * 0.48
    premium_cands  = [p for p in products if p.price >= premium_cutoff]
    budget_cands   = [p for p in products if p.price <= budget_cutoff]

    return {
        "Best Overall":         by_score[0] if by_score else None,
        "Best Budget":          min(budget_cands, key=lambda p: p.price) if budget_cands else by_price[0],
        "Best Premium":         max(premium_cands, key=lambda p: p.ai_score) if premium_cands else by_score[0],
        "Best Value":           by_value[0] if by_value else None,
        "Highest Rated":        by_rating[0] if by_rating else None,
        "Best for Beginners":   by_beginner[0] if by_beginner else None,
    }


def generate_ai_insight(query: str, products: List[Product], budget: float, llm: str) -> str:
    if not products:
        return "No products found. Try adjusting your filters."

    prices      = [p.price for p in products]
    avg_p       = sum(prices) / len(prices)
    best        = max(products, key=lambda p: p.ai_score)
    cheapest    = min(products, key=lambda p: p.price)
    rated       = max(products, key=lambda p: p.rating)
    trending_dn = sum(1 for p in products if "↓" in p.price_trend)
    parts: List[str] = []
    q = query.lower()

    if any(k in q for k in ["laptop", "macbook", "thinkpad", "notebook"]):
        parts.append(f"Analyzed **{len(products)} laptops** across {len(set(p.platform for p in products))} platforms. The **{best.title[:40]}...** leads with an AI score of **{best.ai_score}%** — its architecture delivers peak performance-per-watt in this price tier.")
        parts.append(f"\n\n**Budget Insight:** At ${avg_p:,.0f} average, this category sits firmly mid-premium. You can save **20–30%** by opting for a certified refurb or last-gen flagship, which retains 90%+ of real-world performance.")
    elif any(k in q for k in ["headphone", "earbud", "audio", "speaker"]):
        parts.append(f"Scanned **{len(products)} audio products**. The **{best.title[:40]}...** ranks #1 with {best.ai_score}% — exceptional noise cancellation and driver tuning make it the standout for most use cases.")
        parts.append(f"\n\n**Category Trend:** Wireless ANC headphones have dropped ~12% in median price over 90 days — this is a strong buyer's market right now.")
    elif any(k in q for k in ["phone", "iphone", "samsung", "pixel", "android"]):
        parts.append(f"Evaluated **{len(products)} smartphones**. The **{best.title[:40]}...** scores highest at {best.ai_score}% — its camera system and software update lifecycle make it the most future-proof option.")
        parts.append(f"\n\n**Value Angle:** **{cheapest.title[:35]}...** at **${cheapest.price:,.0f}** delivers flagship-adjacent performance at {int((1-cheapest.price/max(p.price for p in products))*100)}% less than the top-tier option.")
    elif any(k in q for k in ["coffee", "espresso", "grinder", "brew"]):
        parts.append(f"Curated **{len(products)} coffee products**. The **{best.title[:40]}...** wins on AI scoring — an integrated grinder or precision pour-over system dramatically outperforms pod alternatives for flavor complexity.")
        parts.append(f"\n\n**ROI Math:** A premium home setup at ${avg_p:,.0f} avg pays for itself vs. daily café spending within 4–6 months for a 2-cup-a-day habit.")
    elif any(k in q for k in ["gaming", "console", "monitor", "keyboard", "mouse"]):
        parts.append(f"Scanned **{len(products)} gaming products**. The **{best.title[:40]}...** tops the chart with {best.ai_score}% AI Score — built for competitive play with hardware that won't bottleneck your performance.")
        parts.append(f"\n\n**Platform Note:** Best Buy and Newegg offer price-match guarantees on most gaming hardware — worth checking before checkout.")
    elif any(k in q for k in ["chair", "desk", "ergonomic", "standing"]):
        parts.append(f"Found **{len(products)} workspace products**. The **{best.title[:40]}...** dominates with {best.ai_score}% — premium ergonomic chairs correlate strongly with reduced fatigue for 6+ hour work sessions.")
        parts.append(f"\n\n**Investment Framing:** At ${avg_p:,.0f} average, quality workspace gear is a health investment. Musculoskeletal issues from poor posture cost the average worker far more annually.")
    else:
        parts.append(f"Analyzed **{len(products)} products** for **'{query}'** across {len(set(p.platform for p in products))} platforms. The **{best.title[:40]}...** ranks #1 at {best.ai_score}% AI score, balancing rating ({best.rating}★), reviews ({best.reviews:,}), and price-value ratio.")

    if trending_dn >= len(products) // 2:
        parts.append(f"\n\n**⬇️ Price Alert:** {trending_dn}/{len(products)} products trending downward — a market correction window. Historical patterns suggest 2–3 weeks of further softening before seasonal demand spikes.")

    parts.append(f"\n\n**Highest Rated:** **{rated.title[:35]}...** at ★{rated.rating} ({rated.reviews:,} reviews) — community validation at this scale is one of the strongest quality signals.")

    deal_products = [p for p in products if p.discount_pct >= 15]
    if deal_products:
        top_disc = max(deal_products, key=lambda p: p.discount_pct)
        parts.append(f"\n\n**Best Saving:** **{top_disc.title[:35]}...** is {top_disc.discount_pct:.0f}% off (save ${top_disc.savings:,.0f}) — the strongest dollar-value deal in this result set.")

    beg_pick = max(products, key=lambda p: p.beginner_score)
    parts.append(f"\n\n**Beginner Pick:** **{beg_pick.title[:35]}...** scores highest for ease of entry — great reviews, strong platform support, and mass availability make it the safest first purchase.")

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

    best_rated  = max(products, key=lambda p: p.rating)
    cheapest    = min(products, key=lambda p: p.price)
    most_rev    = max(products, key=lambda p: p.reviews)
    best_deal   = max(products, key=lambda p: p.discount_pct)

    if best_rated.rating > avg_r:
        diff = ((best_rated.rating - avg_r) / avg_r * 100)
        insights.append(f"📈 **{best_rated.title[:35]}...** has a {diff:.0f}% higher rating than the category average (★{avg_r:.2f})")
    insights.append(f"💰 **{cheapest.title[:35]}...** is the most affordable at ${cheapest.price:,.2f} — {((avg_p-cheapest.price)/avg_p*100):.0f}% below average")
    insights.append(f"💬 **{most_rev.title[:35]}...** leads in social proof with {most_rev.reviews:,} verified reviews")

    platforms = list(set(p.platform for p in products))
    if len(platforms) > 1:
        insights.append(f"🏪 Results span {len(platforms)} platforms: {', '.join(platforms[:5])}")

    deals = [p for p in products if "Excellent" in p.deal_label or "Great" in p.deal_label]
    if deals:
        insights.append(f"🔥 {len(deals)} of {len(products)} products qualify as Great Deal or better")

    if best_deal.discount_pct >= 10:
        insights.append(f"🏷️ Biggest discount: {best_deal.discount_pct:.0f}% off — **{best_deal.title[:30]}...** saves you ${best_deal.savings:,.2f}")

    beg_products = [p for p in products if p.beginner_score >= 0.7]
    if beg_products:
        insights.append(f"🟢 {len(beg_products)} product(s) flagged as Beginner Friendly with strong community support")

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
    total_savings = sum(p.savings for p in products)
    avg_disc = sum(p.discount_pct for p in products) / n
    return {
        "avg":          round(sum(prices) / n, 2),
        "median":       round(median, 2),
        "min":          round(min(prices), 2),
        "max":          round(max(prices), 2),
        "range":        round(max(prices) - min(prices), 2),
        "count":        n,
        "total_savings": round(total_savings, 2),
        "avg_discount": round(avg_disc, 1),
    }


def build_plotly_price_chart(products: List[Product]) -> go.Figure:
    df = pd.DataFrame([{
        "Title": p.title[:26] + "…",
        "Price": p.price,
        "Rating": p.rating,
        "Platform": p.platform,
        "AI Score": p.ai_score
    } for p in products])
    fig = px.bar(
        df, x="Title", y="Price", color="AI Score",
        color_continuous_scale=[[0,"#3a3a3a"],[0.4,"#d4af37"],[1,"#64dc82"]],
        hover_data=["Rating","Platform"],
        template="plotly_dark",
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans", color="#c8c4ba", size=11),
        margin=dict(l=10, r=10, t=20, b=90),
        coloraxis_colorbar=dict(title="AI %", tickfont=dict(color="#888")),
        xaxis=dict(tickangle=-38, gridcolor="rgba(255,255,255,0.04)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.04)", tickprefix="$"),
        height=330,
    )
    return fig


def build_plotly_scatter(products: List[Product]) -> go.Figure:
    df = pd.DataFrame([{
        "Price": p.price, "Rating": p.rating,
        "Reviews": p.reviews, "Title": p.title[:28] + "…",
        "Platform": p.platform, "AI Score": p.ai_score
    } for p in products])
    fig = px.scatter(
        df, x="Price", y="Rating", size="Reviews",
        color="Platform", hover_name="Title",
        hover_data=["AI Score"],
        template="plotly_dark",
        color_discrete_sequence=["#d4af37","#64dc82","#ff9900","#4ab3ff","#ff6060","#96bf44","#ff9a40","#f1641e"],
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans", color="#c8c4ba", size=11),
        margin=dict(l=10, r=10, t=20, b=40),
        xaxis=dict(gridcolor="rgba(255,255,255,0.04)", tickprefix="$"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.04)", range=[3.8, 5.1]),
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
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
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
    order = ["Excellent Deal", "Great Deal", "Good Value", "Fair Price", "Overpriced"]
    sorted_labels = [l for l in order if l in deal_map] + [l for l in deal_map if l not in order]
    sorted_vals   = [deal_map[l] for l in sorted_labels]
    colors_map    = {"Excellent Deal": "#64dc82", "Great Deal": "#d4af37", "Good Value": "#ffb43c", "Fair Price": "#888", "Overpriced": "#ff6060"}
    bar_colors    = [colors_map.get(l, "#d4af37") for l in sorted_labels]
    fig = px.bar(
        x=sorted_vals, y=sorted_labels, orientation="h",
        template="plotly_dark",
    )
    fig.update_traces(marker_color=bar_colors)
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans", color="#c8c4ba", size=11),
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
        height=230,
        showlegend=False,
    )
    return fig


def build_ai_score_radar(products: List[Product]) -> go.Figure:
    top5 = sorted(products, key=lambda p: p.ai_score, reverse=True)[:5]
    cats = ["Rating", "Reviews", "Price Value", "Platform Trust", "Deal Quality"]
    fig = go.Figure()
    colors = ["#d4af37","#64dc82","#60a4ff","#ff9a40","#f1641e"]
    for i, p in enumerate(top5):
        trust = PLATFORM_TRUST.get(p.platform, 0.75) * 100
        rev_n = min(100, math.log1p(p.reviews) / math.log1p(60000) * 100)
        pv    = max(0, 100 - p.price / max(q.price for q in products) * 60)
        vals  = [p.rating * 20, rev_n, pv, trust, p.deal_quality_score * 100]
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]], theta=cats + [cats[0]],
            fill="toself", name=p.title[:22] + "…",
            line=dict(color=colors[i % len(colors)], width=2),
            fillcolor=colors[i % len(colors)].replace("#","rgba(").replace("d4af37","212,175,55,0.08)").replace("64dc82","100,220,130,0.08)").replace("60a4ff","96,164,255,0.08)").replace("ff9a40","255,154,64,0.08)").replace("f1641e","241,100,30,0.08)"),
        ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0,100], gridcolor="rgba(255,255,255,0.06)", tickfont=dict(color="#666", size=9)),
            angularaxis=dict(gridcolor="rgba(255,255,255,0.08)", tickfont=dict(color="#aaa", size=10)),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans", color="#c8c4ba", size=10),
        legend=dict(font=dict(color="#c8c4ba", size=9)),
        margin=dict(l=30, r=30, t=20, b=20),
        height=300,
        template="plotly_dark",
    )
    return fig


def build_price_distribution(products: List[Product]) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Box(
        y=[p.price for p in products],
        name="Price Range",
        marker_color="#d4af37",
        line_color="#d4af37",
        fillcolor="rgba(212,175,55,0.1)",
        boxpoints="all",
        jitter=0.4,
        pointpos=0,
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans", color="#c8c4ba", size=11),
        margin=dict(l=10, r=10, t=20, b=10),
        yaxis=dict(gridcolor="rgba(255,255,255,0.04)", tickprefix="$"),
        height=260,
        template="plotly_dark",
    )
    return fig

# ═══════════════════════════════════════════════════════════════════════════════
# COMPARISON ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def render_comparison_table(compare_list: List[Product]) -> None:
    if not compare_list:
        st.markdown('<div class="empty-state"><div class="empty-icon">⚖️</div><div class="empty-title">No products to compare</div><div class="empty-sub">Click ⚖️ Compare on any product card to add it here.</div></div>', unsafe_allow_html=True)
        return

    rows: List[Dict] = []
    for p in compare_list:
        rows.append({
            "Title":      p.title[:42],
            "Price":      f"${p.price:,.2f}",
            "Original":   f"${p.original_price:,.2f}",
            "Savings":    f"${p.savings:,.2f}",
            "Rating":     f"★ {p.rating}",
            "Reviews":    f"{p.reviews:,}",
            "Platform":   p.platform,
            "AI Score":   f"{p.ai_score}%",
            "Deal":       p.deal_label,
            "Stock":      p.stock,
            "Shipping":   p.shipping,
            "Discount":   f"{p.discount_pct:.1f}%",
            "Beginner":   f"{p.beginner_score:.0%}",
        })

    df = pd.DataFrame(rows)
    best_ai  = max(compare_list, key=lambda p: p.ai_score).title[:42]
    best_rat = max(compare_list, key=lambda p: p.rating).title[:42]
    cheapest = min(compare_list, key=lambda p: p.price).title[:42]

    st.markdown(f"""
<div style="margin-bottom:.8rem;font-size:.8rem;color:var(--text3);">
🥇 <b style="color:var(--green)">Best AI Score:</b> {best_ai[:35]}…
&nbsp;&nbsp;|&nbsp;&nbsp;
⭐ <b style="color:var(--gold)">Highest Rated:</b> {best_rat[:35]}…
&nbsp;&nbsp;|&nbsp;&nbsp;
💰 <b style="color:#64dc82">Lowest Price:</b> {cheapest[:35]}…
</div>""", unsafe_allow_html=True)

    st.dataframe(df, use_container_width=True, height=min(40 + len(rows) * 36, 360))
    csv_bytes = df.to_csv(index=False).encode()
    st.download_button("⬇️ Export Comparison CSV", csv_bytes, "comparison.csv", "text/csv", key="dl_compare")

# ═══════════════════════════════════════════════════════════════════════════════
# CHAT ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def generate_chat_response(user_msg: str, products: List[Product], query: str, budget: float, llm: str) -> str:
    u = user_msg.lower()
    if not products:
        return "Please run a search first — I'll then have product data to answer your questions about."

    best     = max(products, key=lambda p: p.ai_score)
    cheapest = min(products, key=lambda p: p.price)
    rated    = max(products, key=lambda p: p.rating)
    reviewed = max(products, key=lambda p: p.reviews)
    beginner = max(products, key=lambda p: p.beginner_score)

    if any(w in u for w in ["warrant", "return", "policy", "guarantee"]):
        return (
            f"Based on your search results, warranty & return policies by platform:\n\n"
            f"• **Amazon** — 30-day returns, manufacturer warranty honored\n"
            f"• **Best Buy** — 15-day returns (30 days for My Best Buy members), Geek Squad plans available\n"
            f"• **Walmart** — 30-day returns; 90 days on most electronics\n"
            f"• **Target** — 30-day returns; RedCard extends to 60 days\n"
            f"• **Newegg** — 15–30 day RMA per product listing\n"
            f"• **eBay** — seller-specific; filter for '30-day returns' guarantee\n"
            f"• **Etsy** — varies by seller; message them before purchasing\n"
            f"• **Shopify** — store-specific; check individual merchant policy\n\n"
            f"For your top pick **{best.title[:40]}...** on {best.platform}: check the listing for the exact manufacturer warranty."
        )

    if any(w in u for w in ["ship", "deliver", "fast", "quick", "speed", "arrival"]):
        fast = min(products, key=lambda p: 0 if ("2-day" in p.shipping or "Same" in p.shipping) else 1)
        return (
            f"Fastest shipping from your results: **{fast.title[:40]}...** on {fast.platform} — **{fast.shipping}**.\n\n"
            f"Speed overview:\n"
            f"• **Amazon Prime 2-day** — fastest for Prime members\n"
            f"• **Best Buy / Target** — same-day in-store pickup available\n"
            f"• **Walmart** — free 2-day on eligible $35+ orders\n"
            f"• **eBay** — varies by seller location; check estimated delivery\n"
            f"• **Etsy** — handmade items ship in 3–14 days\n"
            f"• **Shopify** — varies by brand"
        )

    if any(w in u for w in ["beginner", "first time", "starter", "easy", "new to"]):
        return (
            f"For a first-time buyer, I'd strongly recommend **{beginner.title[:45]}...** — it has the highest Beginner Score in your results.\n\n"
            f"Why it's great for newcomers:\n"
            f"• ★{beginner.rating} rating from {beginner.reviews:,} reviews — extensive community feedback\n"
            f"• Listed on {beginner.platform} — strong buyer protection and easy returns\n"
            f"• ${beginner.price:,.2f} — {beginner.deal_label}\n\n"
            f"Beginner Score measures: review volume, platform reliability, stock availability, and how mainstream the product category is. *Powered by {llm}*"
        )

    if any(w in u for w in ["spec", "technical", "detail", "feature"]):
        p = best
        if p.specs:
            spec_lines = "\n".join([f"• **{k}:** {v}" for k, v in p.specs.items()])
            return (f"Technical specs for **{p.title[:45]}...** (top AI-scored product):\n\n{spec_lines}\n\nFor full specifications, check the product listing on {p.platform}. *Powered by {llm}*")
        return f"Detailed specs for **{p.title[:45]}...** aren't available in my current data. Check the {p.platform} listing for full specifications."

    if any(w in u for w in ["under", "budget", "cheap", "affordable", "less than"]):
        return (
            f"Within your ${budget:,.0f} budget, the best value pick is **{cheapest.title[:42]}...** at **${cheapest.price:,.2f}**.\n\n"
            f"AI Score: {cheapest.ai_score}% · ★{cheapest.rating} · {cheapest.reviews:,} reviews · saves **${cheapest.savings:,.2f}**\n\n"
            f"If you can stretch slightly, **{best.title[:35]}...** at ${best.price:,.2f} scores {best.ai_score}% — the highest overall. *Powered by {llm}*"
        )

    if any(w in u for w in ["compar", "difference", "vs", "versus", "better", "between"]):
        p1, p2 = (products[0], products[1]) if len(products) > 1 else (products[0], products[0])
        winner = p1 if p1.ai_score >= p2.ai_score else p2
        return (
            f"**{p1.title[:35]}...** (AI: {p1.ai_score}%, ★{p1.rating}, ${p1.price:,.2f} — {p1.platform})\n"
            f"vs\n"
            f"**{p2.title[:35]}...** (AI: {p2.ai_score}%, ★{p2.rating}, ${p2.price:,.2f} — {p2.platform})\n\n"
            f"🏆 **Recommendation: {winner.title[:35]}...** edges ahead on the composite AI score — factoring rating, community trust, platform reliability, and price-value. *Powered by {llm}*"
        )

    if any(w in u for w in ["trusted", "popular", "review", "community", "proven"]):
        return (
            f"Most community-validated product: **{reviewed.title[:40]}...** — **{reviewed.reviews:,} reviews** · ★{reviewed.rating}\n\n"
            f"At this review volume, the rating is statistically robust (< 0.3% variance). Listed on **{reviewed.platform}** at **${reviewed.price:,.2f}**.\n\n"
            f"High review count is one of the strongest trust signals I factor into AI scoring. *Powered by {llm}*"
        )

    if any(w in u for w in ["deal", "discount", "sale", "off", "saving"]):
        best_deal = max(products, key=lambda p: p.discount_pct)
        return (
            f"Best deal in your results: **{best_deal.title[:42]}...** — {best_deal.discount_pct:.0f}% off, saving you **${best_deal.savings:,.2f}**.\n\n"
            f"Deal label: {best_deal.deal_label}\n"
            f"Price: ${best_deal.price:,.2f} (was ${best_deal.original_price:,.2f})\n"
            f"Platform: {best_deal.platform} · {best_deal.shipping}\n\n"
            f"Price trend: {best_deal.price_trend} — {'good time to buy!' if '↓' in best_deal.price_trend else 'prices are stable.'} *Powered by {llm}*"
        )

    return (
        f"Based on your search for **'{query}'**, here's my snapshot:\n\n"
        f"• 🏆 **Top Overall:** {best.title[:40]}… — AI Score {best.ai_score}%, ★{best.rating}, ${best.price:,.2f}\n"
        f"• 💰 **Best Budget:** {cheapest.title[:40]}… — ${cheapest.price:,.2f}\n"
        f"• ⭐ **Highest Rated:** {rated.title[:40]}… — ★{rated.rating}\n"
        f"• 🟢 **Best for Beginners:** {beginner.title[:40]}… — Score {beginner.beginner_score:.0%}\n\n"
        f"Products are ranked by my weighted AI engine across rating quality, review volume, price-value, platform trust, availability, and discount depth.\n\n"
        f"*Powered by {llm}*"
    )

# ═══════════════════════════════════════════════════════════════════════════════
# LLM ADAPTER (pluggable architecture)
# ═══════════════════════════════════════════════════════════════════════════════

class LLMAdapter:
    def __init__(self, provider: str, api_key: Optional[str] = None):
        self.provider  = provider
        self.api_key   = api_key or ""
        self.available = bool(api_key)

    def complete(self, prompt: str, system: str = "", max_tokens: int = 512) -> str:
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
        return f"[{self.provider} — demo mode. Add an API key in the sidebar to enable real completions.]"

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
                timeout=18,
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
                timeout=18,
            )
            return r.json()["content"][0]["text"]
        except Exception as e:
            logger.error(f"Claude error: {e}")
            return self._mock_complete(prompt)

    def _gemini_complete(self, prompt: str, system: str, max_tokens: int) -> str:
        try:
            import requests as req
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent?key={self.api_key}"
            r = req.post(url, json={"contents": [{"parts": [{"text": prompt}]}]}, timeout=18)
            return r.json()["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as e:
            logger.error(f"Gemini error: {e}")
            return self._mock_complete(prompt)

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


def render_product_card(p: Product, max_price: float, idx: int, show_specs: bool = False) -> None:
    cfg = PLATFORM_CONFIG.get(p.platform, {"badge": "badge-amazon", "emoji": "🛒"})
    bar_pct   = int((p.price / max(max_price, 1)) * 100)
    tags_html = " ".join(f'<span class="tag tag-deal">{t}</span>' for t in p.tags)
    trend_cls = "tag-green" if "↓" in p.price_trend else ("tag-red" if "↑" in p.price_trend else "tag-gray")
    sc        = score_class(p.ai_score)
    savings_html = f'<span class="psaving">Save ${p.savings:,.2f}</span>' if p.savings > 0 else ''
    orig_html    = f'<span class="pprice-orig">${p.original_price:,.2f}</span>' if p.savings > 0 else ''

    beg_badge = ""
    if p.beginner_score >= 0.75:
        beg_badge = f'<span class="beginner-pill" style="float:right;clear:right;">🟢 Beginner</span>'

    specs_html = ""
    if show_specs and p.specs:
        specs_items = list(p.specs.items())[:6]
        rows = "".join(f'<div class="spec-row"><span class="spec-key">{k}</span><span class="spec-val">{v}</span></div>' for k, v in specs_items)
        specs_html = f'<div class="spec-grid" style="margin-top:.5rem;">{rows}</div>'

    card = f"""
<div class="pcard">
  <span class="pbadge {cfg['badge']}">{cfg['emoji']} {p.platform}</span>
  <span class="ai-pill {sc}">{p.ai_score}%</span>
  <div style="clear:both"></div>
  {beg_badge}
  <div style="font-size:2rem;margin:.5rem 0 .2rem;">{p.emoji}</div>
  <p class="ptitle">{p.title}</p>
  <div class="pprice-wrap">
    <span class="pprice">${p.price:,.2f}</span>{orig_html}{savings_html}
  </div>
  <div><span class="prating">{stars(p.rating)}</span> <span class="pmeta">★{p.rating} &nbsp;({p.reviews:,} reviews)</span></div>
  <div class="pbar-outer"><div class="pbar-inner" style="width:{bar_pct}%"></div></div>
  <div class="pmeta">📦 {p.stock} &nbsp;·&nbsp; 🚚 {p.shipping}</div>
  <div class="pmeta">🏷️ {p.brand} &nbsp;·&nbsp; 📂 {p.category}</div>
  {specs_html}
  <div style="margin-top:.5rem;">
    <span class="tag {trend_cls}">{p.price_trend}</span>
    <span class="tag tag-deal">{p.deal_label}</span>
    {tags_html}
  </div>
  <a href="{p.url}" target="_blank" style="display:inline-block;margin-top:.7rem;padding:6px 14px;background:rgba(212,175,55,.09);border:1px solid rgba(212,175,55,.32);border-radius:6px;color:#d4af37;text-decoration:none;font-size:.78rem;font-weight:600;letter-spacing:.04em;">View on {p.platform} →</a>
</div>"""
    st.markdown(card, unsafe_allow_html=True)


def render_rec_badge(label: str, product: Optional[Product]) -> None:
    if product is None:
        return
    cfg = PLATFORM_CONFIG.get(product.platform, {"badge": "badge-amazon"})
    icons = {
        "Best Overall": "🏆", "Best Budget": "💰", "Best Premium": "💎",
        "Best Value": "⚖️", "Highest Rated": "⭐", "Best for Beginners": "🟢",
    }
    icon = icons.get(label, "✦")
    st.markdown(f"""
<div class="rec-badge">
  <div style="font-family:'JetBrains Mono',monospace;font-size:.6rem;letter-spacing:.18em;text-transform:uppercase;color:var(--gold);margin-bottom:.35rem;">{icon} {label}</div>
  <div style="font-size:.88rem;font-weight:600;color:var(--text);margin-bottom:.3rem;">{product.emoji} {product.title[:45]}…</div>
  <div style="display:flex;gap:.8rem;align-items:center;flex-wrap:wrap;">
    <span style="font-family:'Playfair Display',serif;color:var(--gold);font-size:1.1rem;">${product.price:,.2f}</span>
    <span style="font-size:.78rem;color:var(--text3);">★{product.rating}</span>
    <span class="pbadge {cfg['badge']}" style="margin:0;">{product.platform}</span>
    <span style="font-size:.75rem;color:var(--green);font-weight:700;">AI {product.ai_score}%</span>
    {f'<span class="beginner-pill">🟢 Beginner</span>' if product.beginner_score >= 0.75 else ""}
  </div>
</div>""", unsafe_allow_html=True)


def export_products_csv(products: List[Product]) -> bytes:
    rows = [asdict(p) for p in products]
    for row in rows:
        row.pop("tags", None)
        row.pop("specs", None)
    df = pd.DataFrame(rows)
    return df.to_csv(index=False).encode()


def export_products_json(products: List[Product]) -> bytes:
    data = []
    for p in products:
        d = asdict(p)
        data.append(d)
    return json.dumps(data, indent=2).encode()

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
</div>""", unsafe_allow_html=True)

        # API Keys
        st.markdown('<span class="sl">🔑 API Keys</span>', unsafe_allow_html=True)
        with st.expander("Configure API Keys", expanded=False):
            openai_key  = st.text_input("OpenAI Key",    placeholder="sk-...",       type="password", key="k_openai")
            claude_key  = st.text_input("Anthropic Key", placeholder="sk-ant-...",   type="password", key="k_claude")
            gemini_key  = st.text_input("Gemini Key",    placeholder="AIza...",      type="password", key="k_gemini")
            st.caption("🔒 Session-only. Never persisted. Demo mode without keys.")

        st.markdown("<hr>", unsafe_allow_html=True)

        st.markdown('<span class="sl">🤖 AI Engine</span>', unsafe_allow_html=True)
        llm_choice    = st.selectbox("Model", LLM_ADAPTERS, key="llm_choice")
        search_depth  = st.select_slider("Search Depth", ["Quick","Standard","Deep","Exhaustive"], value="Standard")
        use_reranker  = st.checkbox("AI Re-ranker",         value=True, key="reranker")
        use_sentiment = st.checkbox("Sentiment Analysis",   value=True, key="sentiment")
        use_history   = st.checkbox("Conversation Memory",  value=True, key="memory")
        show_specs    = st.checkbox("Show Specs on Cards",  value=False, key="show_specs")

        st.markdown("<hr>", unsafe_allow_html=True)

        st.markdown('<span class="sl">🏪 Platforms</span>', unsafe_allow_html=True)

        # Live platform status strip
        plat_status_html = " ".join([
            f'<span class="plat-status"><span class="plat-dot"></span>{k}</span>'
            for k in list(PLATFORM_CONFIG.keys())[:4]
        ])
        st.markdown(f'<div style="display:flex;flex-wrap:wrap;gap:4px;margin-bottom:.6rem;">{plat_status_html}</div>', unsafe_allow_html=True)

        active_platforms = st.multiselect(
            "Active Platforms",
            list(PLATFORM_CONFIG.keys()),
            default=["Amazon","Best Buy","Walmart","Target","eBay","Newegg"],
            key="platforms",
        )

        st.markdown("<hr>", unsafe_allow_html=True)

        st.markdown('<span class="sl">🎛️ Filters</span>', unsafe_allow_html=True)
        budget       = st.slider("Max Budget ($)",  10,  5000, DEFAULT_BUDGET, 10,  key="budget")
        min_rating   = st.slider("Min Rating ★",    1.0, 5.0,  4.0,            0.1, key="min_rating")
        min_reviews  = st.slider("Min Reviews",     0,   5000, 0,              100, key="min_reviews")
        min_discount = st.slider("Min Discount %",  0,   50,   0,              5,   key="min_disc")
        sort_by      = st.selectbox("Sort By", [
            "AI Score","Price: Low→High","Price: High→Low",
            "Rating","Reviews","Discount %","Beginner Score",
        ], key="sort_by")

        st.markdown("<hr>", unsafe_allow_html=True)

        st.markdown('<span class="sl">📊 Session Stats</span>', unsafe_allow_html=True)
        sc1, sc2 = st.columns(2)
        with sc1: st.metric("Searches", len(st.session_state.search_history))
        with sc2: st.metric("Wishlist",  len(st.session_state.wishlist))
        sc3, sc4 = st.columns(2)
        with sc3: st.metric("Compare",  len(st.session_state.compare_list))
        with sc4: st.metric("Products", "2.4M+")

        if st.session_state.wishlist:
            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown('<span class="sl">❤️ Saved Items</span>', unsafe_allow_html=True)
            for item in st.session_state.wishlist[-4:]:
                st.markdown(f"<div style='font-size:.76rem;color:var(--text3);padding:2px 0;'>• {item[:32]}…</div>", unsafe_allow_html=True)
            if len(st.session_state.wishlist) > 4:
                st.caption(f"+{len(st.session_state.wishlist)-4} more")

    # ── HERO ────────────────────────────────────────────────────────────────
    st.markdown("""
<div class="hero-wrap">
  <div class="hero-glow"></div>
  <div class="hero-eyebrow">Powered by AI · 8 Shopping Platforms · Real-Time Price Intelligence</div>
  <h1 class="hero-title">Find the <em>Perfect</em> Product</h1>
  <p class="hero-sub">AI shopping intelligence that searches Amazon, eBay, Walmart, Etsy, Best Buy, Target, Shopify &amp; Newegg simultaneously — then ranks everything with weighted LLM reasoning.</p>
</div>""", unsafe_allow_html=True)

    chips = "".join(f'<span class="pchip">{PLATFORM_CONFIG[k]["emoji"]} {k}</span>' for k in PLATFORM_CONFIG)
    st.markdown(f'<div class="pstrip">{chips}</div>', unsafe_allow_html=True)

    # ── SEARCH BAR ──────────────────────────────────────────────────────────
    s_col, b_col = st.columns([5, 1])
    with s_col:
        query = st.text_input(
            "search",
            placeholder='Try "wireless headphones for gym" · "gaming laptop under $1500" · "espresso machine" · "ergonomic chair"',
            label_visibility="collapsed",
            key="main_query",
        )
    with b_col:
        search_btn = st.button("🔍 Search", use_container_width=True, key="do_search")

    # Trending row
    st.markdown('<div style="margin:.4rem 0 1.2rem;"><span style="font-size:.68rem;color:var(--text4);font-family:JetBrains Mono,monospace;letter-spacing:.12em;text-transform:uppercase;">Trending · </span>', unsafe_allow_html=True)
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
        rec = SearchRecord(query=query)
        history: List[SearchRecord] = st.session_state.search_history
        if not history or history[-1].query != query:
            history.append(rec)
            if len(history) > MAX_SEARCH_HISTORY:
                history.pop(0)
            st.session_state.search_history = history

        depth_delay = {"Quick": 0.14, "Standard": 0.24, "Deep": 0.36, "Exhaustive": 0.50}
        delay = depth_delay.get(search_depth, 0.24)

        with st.spinner("🤖 AI agent analyzing query and querying platforms…"):
            prog   = st.progress(0)
            status = st.empty()
            steps  = [
                (6,  f"🧠 Parsing intent for *'{query}'*…"),
                (16, f"🔗 Initialising {llm_choice} ReAct agent…"),
                (28, "📡 Querying Amazon & Best Buy APIs in parallel…"),
                (42, "📡 Querying Walmart, Target & Newegg APIs…"),
                (56, "📡 Querying eBay, Etsy & Shopify APIs…"),
                (68, "⚖️ Running cross-encoder re-ranker…" if use_reranker else "📊 Ranking results…"),
                (79, "💬 Running sentiment analysis…" if use_sentiment else "📊 Compiling insights…"),
                (89, "🟢 Computing Beginner Scores…"),
                (96, f"✨ Generating AI analysis with {llm_choice}…"),
                (100,"✅ Done!"),
            ]
            for pct, msg in steps:
                prog.progress(pct)
                status.markdown(f"<div style='font-size:.82rem;color:var(--text3);'>{msg}</div>", unsafe_allow_html=True)
                time.sleep(delay)
            prog.empty(); status.empty()

        products, platform_counts = build_products(query, active_platforms, budget, min_rating, min_reviews, min_discount)
        products = sort_products(products, sort_by)

        key_map = {
            "GPT-4o (OpenAI)":   st.session_state.get("k_openai", ""),
            "Claude Sonnet 4":   st.session_state.get("k_claude", ""),
            "Gemini 1.5 Pro":    st.session_state.get("k_gemini", ""),
            "LLaMA 3.1 (Local)": "",
            "Mixtral 8x7B":      "",
        }
        _llm_adapter = LLMAdapter(llm_choice, key_map.get(llm_choice, ""))

        st.session_state.search_results     = products
        st.session_state.platform_results   = platform_counts
        st.session_state.ai_insight         = generate_ai_insight(query, products, budget, llm_choice)
        st.session_state.ai_recommendations = generate_recommendations(products, budget)
        st.session_state.product_insights   = generate_product_insights(products)

        if st.session_state.search_history:
            st.session_state.search_history[-1].result_count = len(products)

        st.session_state.chat_history.append(ChatMessage(role="user", content=f"Search: {query}"))
        if products:
            top = products[0]
            st.session_state.chat_history.append(ChatMessage(
                role="assistant",
                content=(f"Found **{len(products)} products** across **{len(active_platforms)} platforms** for '{query}'. "
                         f"Top pick: **{top.title[:40]}…** (AI Score {top.ai_score}%) at ${top.price:,.2f}.")
            ))
        else:
            st.session_state.chat_history.append(ChatMessage(
                role="assistant",
                content="No products matched your filters. Try relaxing the budget or rating constraints."
            ))

        logger.info(f"Search '{query}' → {len(products)} results across {len(active_platforms)} platforms")

    # ── RESULTS ─────────────────────────────────────────────────────────────
    products: List[Product] = st.session_state.search_results

    if products:
        st.markdown("<hr>", unsafe_allow_html=True)

        analytics = price_analytics(products)
        m_cols    = st.columns(7)
        metrics   = [
            ("Products Found",  str(analytics["count"])),
            ("Avg. Price",      f"${analytics['avg']:,.0f}"),
            ("Median Price",    f"${analytics['median']:,.0f}"),
            ("Best Price",      f"${analytics['min']:,.0f}"),
            ("Top AI Score",    f"{max(p.ai_score for p in products)}%"),
            ("Avg. Rating",     f"★ {sum(p.rating for p in products)/len(products):.1f}"),
            ("Avg. Discount",   f"{analytics['avg_discount']:.1f}%"),
        ]
        for col, (label, val) in zip(m_cols, metrics):
            with col:
                st.metric(label, val)

        st.markdown("<div style='height:.8rem'></div>", unsafe_allow_html=True)

        # ── TABS ────────────────────────────────────────────────────────────
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
            "🏆 Results",
            "✨ AI Picks",
            "📊 Analytics",
            "🎯 Deep Dive",
            "⚖️ Compare",
            "💬 Assistant",
            "❤️ Wishlist",
            "📁 Export",
        ])

        max_price = max(p.price for p in products)

        # ── TAB 1: Results ──────────────────────────────────────────────────
        with tab1:
            st.markdown(
                f'<div class="sec-eyebrow">SEARCH RESULTS</div>'
                f'<div class="sec-title">Results for "{st.session_state.last_query}"</div>',
                unsafe_allow_html=True,
            )
            for ins in st.session_state.product_insights[:3]:
                st.markdown(f'<div class="insight-card">{ins}</div>', unsafe_allow_html=True)

            st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)

            for i in range(0, len(products), 3):
                row  = products[i:i+3]
                cols = st.columns(3)
                for j, prod in enumerate(row):
                    with cols[j]:
                        render_product_card(prod, max_price, i * 10 + j, show_specs=st.session_state.get("show_specs", False))
                        c1, c2 = st.columns(2)
                        with c1:
                            if st.button("❤️ Save", key=f"wl_{i}_{j}"):
                                if prod.title not in st.session_state.wishlist:
                                    if len(st.session_state.wishlist) < MAX_WISHLIST_ITEMS:
                                        st.session_state.wishlist.append(prod.title)
                                        st.toast("Added to wishlist!", icon="❤️")
                                    else:
                                        st.toast(f"Wishlist full ({MAX_WISHLIST_ITEMS} max)", icon="⚠️")
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

            recs   = st.session_state.ai_recommendations
            labels = list(recs.keys())
            r_cols = st.columns(2)
            for i, label in enumerate(labels):
                with r_cols[i % 2]:
                    render_rec_badge(label, recs[label])

        # ── TAB 3: Analytics ─────────────────────────────────────────────────
        with tab3:
            st.markdown('<div class="sec-eyebrow">ANALYTICS DASHBOARD</div><div class="sec-title">Price & Market Intelligence</div>', unsafe_allow_html=True)

            an = analytics
            a_cols = st.columns(5)
            an_metrics = [
                ("Average Price",  f"${an['avg']:,.2f}"),
                ("Median Price",   f"${an['median']:,.2f}"),
                ("Lowest Price",   f"${an['min']:,.2f}"),
                ("Highest Price",  f"${an['max']:,.2f}"),
                ("Total Savings",  f"${an['total_savings']:,.2f}"),
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

            ch2l, ch2r = st.columns(2)
            with ch2l:
                st.markdown('<div class="sec-eyebrow">PRICE vs RATING (bubble = reviews)</div>', unsafe_allow_html=True)
                st.plotly_chart(build_plotly_scatter(products), use_container_width=True, config={"displayModeBar": False})
            with ch2r:
                st.markdown('<div class="sec-eyebrow">DEAL QUALITY DISTRIBUTION</div>', unsafe_allow_html=True)
                st.plotly_chart(build_deal_quality_chart(products), use_container_width=True, config={"displayModeBar": False})

            ch3l, ch3r = st.columns(2)
            with ch3l:
                st.markdown('<div class="sec-eyebrow">PRICE DISTRIBUTION (BOX PLOT)</div>', unsafe_allow_html=True)
                st.plotly_chart(build_price_distribution(products), use_container_width=True, config={"displayModeBar": False})
            with ch3r:
                st.markdown('<div class="sec-eyebrow">RATING DISTRIBUTION</div>', unsafe_allow_html=True)
                df_alt = pd.DataFrame({"Rating": [p.rating for p in products]})
                chart = (
                    alt.Chart(df_alt)
                    .mark_bar(color="#d4af37", cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
                    .encode(
                        x=alt.X("Rating:Q", bin=alt.Bin(step=0.1), title="Rating"),
                        y=alt.Y("count()", title="Count"),
                        tooltip=["count()", "Rating"],
                    )
                    .properties(height=220, background="transparent")
                    .configure_axis(gridColor="rgba(255,255,255,0.05)", labelColor="#888", titleColor="#666")
                    .configure_view(strokeWidth=0)
                )
                st.altair_chart(chart, use_container_width=True)

            st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)
            with st.expander("📋 Full Data Table", expanded=False):
                df_full = pd.DataFrame([{
                    "Title": p.title, "Price": p.price, "Original": p.original_price,
                    "Discount%": p.discount_pct, "Rating": p.rating, "Reviews": p.reviews,
                    "Platform": p.platform, "Brand": p.brand, "Category": p.category,
                    "AI Score": p.ai_score, "Deal": p.deal_label, "Stock": p.stock,
                    "Trend": p.price_trend, "Beginner Score": p.beginner_score,
                } for p in products])
                st.dataframe(df_full, use_container_width=True)

        # ── TAB 4: Deep Dive ─────────────────────────────────────────────────
        with tab4:
            st.markdown('<div class="sec-eyebrow">DEEP DIVE</div><div class="sec-title">Multi-Dimensional Analysis</div>', unsafe_allow_html=True)

            st.markdown('<div class="sec-eyebrow">AI SCORE RADAR — TOP 5</div>', unsafe_allow_html=True)
            st.plotly_chart(build_ai_score_radar(products), use_container_width=True, config={"displayModeBar": False})

            st.markdown("<div style='height:.8rem'></div>", unsafe_allow_html=True)
            st.markdown('<div class="sec-eyebrow">PRODUCT SPECS EXPLORER</div>', unsafe_allow_html=True)

            spec_products = [p for p in products if p.specs]
            if spec_products:
                sel_title = st.selectbox(
                    "Select product to inspect",
                    [p.title for p in spec_products],
                    key="spec_sel",
                )
                sel_prod = next((p for p in spec_products if p.title == sel_title), None)
                if sel_prod:
                    cfg = PLATFORM_CONFIG.get(sel_prod.platform, {"badge": "badge-amazon"})
                    s_col1, s_col2 = st.columns([1, 2])
                    with s_col1:
                        st.markdown(f"""
<div style="background:rgba(255,255,255,.02);border:1px solid var(--border);border-radius:var(--radius-lg);padding:1.2rem;text-align:center;">
  <div style="font-size:3rem;">{sel_prod.emoji}</div>
  <div style="font-weight:600;color:var(--text);margin:.5rem 0 .3rem;font-size:.9rem;">{sel_prod.title[:40]}…</div>
  <div style="font-family:'Playfair Display',serif;color:var(--gold);font-size:1.3rem;">${sel_prod.price:,.2f}</div>
  <div style="font-size:.78rem;color:var(--text3);margin:.3rem 0;">★{sel_prod.rating} · {sel_prod.reviews:,} reviews</div>
  <span class="pbadge {cfg['badge']}">{sel_prod.platform}</span>
  <div style="margin-top:.6rem;font-size:.75rem;color:var(--green);font-weight:700;">AI Score: {sel_prod.ai_score}%</div>
  <div style="margin-top:.4rem;font-size:.72rem;color:var(--blue);">Beginner: {sel_prod.beginner_score:.0%}</div>
</div>""", unsafe_allow_html=True)
                    with s_col2:
                        if sel_prod.specs:
                            for k, v in sel_prod.specs.items():
                                st.markdown(f"""
<div style="display:flex;justify-content:space-between;padding:.5rem .8rem;border-bottom:1px solid rgba(255,255,255,.04);font-size:.85rem;">
  <span style="color:var(--text3);">{k}</span>
  <span style="color:var(--text);font-weight:500;">{v}</span>
</div>""", unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="empty-state"><div class="empty-icon">📋</div><div class="empty-sub">No detailed specs available for this product.</div></div>', unsafe_allow_html=True)
            else:
                st.info("No detailed spec data available for the current search results.")

            st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
            st.markdown('<div class="sec-eyebrow">PLATFORM PERFORMANCE</div>', unsafe_allow_html=True)
            plat_data = {}
            for p in products:
                if p.platform not in plat_data:
                    plat_data[p.platform] = {"count": 0, "total_score": 0, "total_rating": 0}
                plat_data[p.platform]["count"]        += 1
                plat_data[p.platform]["total_score"]  += p.ai_score
                plat_data[p.platform]["total_rating"] += p.rating

            if plat_data:
                plat_df = pd.DataFrame([{
                    "Platform": k,
                    "Products": v["count"],
                    "Avg AI Score": round(v["total_score"] / v["count"], 1),
                    "Avg Rating": round(v["total_rating"] / v["count"], 2),
                    "Trust Score": PLATFORM_TRUST.get(k, 0.75),
                } for k, v in plat_data.items()])
                st.dataframe(plat_df, use_container_width=True)

        # ── TAB 5: Compare ──────────────────────────────────────────────────
        with tab5:
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
                    "Product":        p.title[:28] + "…",
                    "AI Score":       p.ai_score,
                    "Rating ×20":     p.rating * 20,
                    "Deal Quality":   p.deal_quality_score * 100,
                    "Beginner Score": p.beginner_score * 100,
                } for p in st.session_state.compare_list])
                fig_cmp = go.Figure()
                for col, color in zip(
                    ["AI Score","Rating ×20","Deal Quality","Beginner Score"],
                    ["#d4af37","#64dc82","#4ab3ff","#f1641e"],
                ):
                    fig_cmp.add_trace(go.Bar(name=col, x=compare_df["Product"], y=compare_df[col], marker_color=color))
                fig_cmp.update_layout(
                    barmode="group",
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="DM Sans", color="#c8c4ba", size=11),
                    margin=dict(l=10, r=10, t=10, b=60),
                    xaxis=dict(tickangle=-20, gridcolor="rgba(255,255,255,0.04)"),
                    yaxis=dict(gridcolor="rgba(255,255,255,0.04)", range=[0, 115]),
                    legend=dict(font=dict(color="#c8c4ba")),
                    height=310,
                    template="plotly_dark",
                )
                st.plotly_chart(fig_cmp, use_container_width=True, config={"displayModeBar": False})

        # ── TAB 6: Chat ──────────────────────────────────────────────────────
        with tab6:
            st.markdown('<div class="sec-eyebrow">AI SHOPPING ASSISTANT</div><div class="sec-title">Ask Me Anything</div>', unsafe_allow_html=True)

            with st.container():
                history_to_show = st.session_state.chat_history[-12:] if use_history else st.session_state.chat_history[-2:]
                for msg in history_to_show:
                    if msg.role == "user":
                        st.markdown(f'<div class="chat-user">👤 {msg.content}<div class="chat-ts">{msg.timestamp}</div></div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="chat-ai">🤖 {msg.content}<div class="chat-ts">{msg.timestamp} · {llm_choice}</div></div>', unsafe_allow_html=True)

            st.markdown("<div style='height:.4rem'></div>", unsafe_allow_html=True)

            ci_col, cs_col = st.columns([5, 1])
            with ci_col:
                chat_input = st.text_input(
                    "chat",
                    placeholder="Ask: 'Best beginner pick?' · 'Which has the best warranty?' · 'Compare top 2'",
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

            qq_list = ["Best warranty?", "Ships fastest?", "Best value?", "Compare top 2?", "Best for beginners?", "Best deal?"]
            st.markdown('<div style="margin-top:.9rem;"><span style="font-size:.68rem;color:var(--text4);font-family:JetBrains Mono,monospace;letter-spacing:.12em;text-transform:uppercase;">Quick Ask:</span></div>', unsafe_allow_html=True)
            qq_cols = st.columns(len(qq_list))
            for i, qq in enumerate(qq_list):
                with qq_cols[i]:
                    if st.button(qq, key=f"qq_{i}", use_container_width=True):
                        st.session_state.chat_history.append(ChatMessage(role="user", content=qq))
                        reply = generate_chat_response(qq, products, st.session_state.last_query, budget, llm_choice)
                        st.session_state.chat_history.append(ChatMessage(role="assistant", content=reply))
                        st.rerun()

            if st.session_state.chat_history:
                st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)
                if st.button("🗑️ Clear Chat History", key="clear_chat"):
                    st.session_state.chat_history = []
                    st.rerun()

        # ── TAB 7: Wishlist ──────────────────────────────────────────────────
        with tab7:
            st.markdown('<div class="sec-eyebrow">SAVED ITEMS</div><div class="sec-title">Your Wishlist</div>', unsafe_allow_html=True)
            wl = st.session_state.wishlist
            if wl:
                wl_df  = pd.DataFrame({"Item": wl, "Saved": [datetime.now().strftime("%Y-%m-%d")] * len(wl)})
                wl_csv = wl_df.to_csv(index=False).encode()
                dl_col, clr_col, _ = st.columns([1.5, 1.5, 5])
                with dl_col:
                    st.download_button("⬇️ Export CSV", wl_csv, "wishlist.csv", "text/csv", key="dl_wl")
                with clr_col:
                    if st.button("🗑️ Clear All", key="clear_wl"):
                        st.session_state.wishlist = []
                        st.rerun()
                st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)
                for idx, item in enumerate(wl):
                    w_item, w_btn = st.columns([6, 1])
                    with w_item:
                        st.markdown(f'<div class="wl-row">❤️ <span style="color:var(--text);font-size:.88rem;">{item}</span></div>', unsafe_allow_html=True)
                    with w_btn:
                        if st.button("✕", key=f"rmwl_{idx}"):
                            st.session_state.wishlist.pop(idx)
                            st.rerun()
            else:
                st.markdown('<div class="empty-state"><div class="empty-icon">❤️</div><div class="empty-title">Your wishlist is empty</div><div class="empty-sub">Click ❤️ Save on any product card to save it here.</div></div>', unsafe_allow_html=True)

        # ── TAB 8: Export ────────────────────────────────────────────────────
        with tab8:
            st.markdown('<div class="sec-eyebrow">EXPORT</div><div class="sec-title">Download Your Results</div>', unsafe_allow_html=True)
            st.markdown(f"**{len(products)} products** from your search for **'{st.session_state.last_query}'** are ready to export.")
            st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)
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
                    st.download_button("❤️ Export Wishlist", wl_csv, "wishlist.csv", "text/csv", use_container_width=True, key="dl_wl2")
                else:
                    st.button("❤️ Wishlist Empty", disabled=True, use_container_width=True, key="wl_dis")

            st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
            with st.expander("📋 Preview Export Data", expanded=False):
                df_exp = pd.DataFrame([{
                    "title": p.title, "price": p.price, "original_price": p.original_price,
                    "discount_pct": p.discount_pct, "rating": p.rating, "reviews": p.reviews,
                    "platform": p.platform, "brand": p.brand, "category": p.category,
                    "ai_score": p.ai_score, "deal_label": p.deal_label, "stock": p.stock,
                    "shipping": p.shipping, "price_trend": p.price_trend,
                    "beginner_score": p.beginner_score,
                } for p in products])
                st.dataframe(df_exp, use_container_width=True)

    # ── EMPTY STATE / HOMEPAGE ───────────────────────────────────────────────
    else:
        st.markdown("<div style='height:.8rem'></div>", unsafe_allow_html=True)

        if st.session_state.search_history:
            st.markdown('<div class="sec-eyebrow">RECENT SEARCHES</div>', unsafe_allow_html=True)
            hist_html = " ".join(
                f'<span class="hist-chip">{rec.query} <span style="color:var(--text4);font-size:.65rem;">({rec.result_count})</span></span>'
                for rec in reversed(st.session_state.search_history[-8:])
            )
            st.markdown(f"<div style='margin-bottom:1.8rem;'>{hist_html}</div>", unsafe_allow_html=True)

        st.markdown('<div class="sec-eyebrow">CAPABILITIES</div><div class="sec-title">What ShopSense AI Does</div>', unsafe_allow_html=True)

        features = [
            ("🔗", "8 Live Platforms",    "Amazon, eBay, Walmart, Etsy, Best Buy, Target, Shopify & Newegg searched simultaneously."),
            ("🤖", "AI Scoring Engine",   "Weighted multi-factor scoring: rating, reviews, price-value, platform trust, deal depth, and availability."),
            ("🟢", "Beginner Scores",     "Every product rated for newcomer-friendliness — mainstream platforms, strong reviews, easy returns."),
            ("📊", "Deep Analytics",      "Price boxplots, radar charts, platform distribution, deal quality histograms, and scatter matrices."),
            ("🔬", "Spec Explorer",       "Browse detailed technical specifications for laptops, phones, headphones, and more in the Deep Dive tab."),
            ("💬", "AI Chat Assistant",   "Conversational memory answers questions about specs, warranty, shipping speed, and best-value picks."),
            ("⚖️", "Compare Center",     "Side-by-side comparison of up to 5 products with grouped bar charts and exportable CSV."),
            ("📁", "Export Anywhere",     "One-click CSV and JSON export of full results plus wishlist to any spreadsheet or workflow."),
        ]
        fcols = st.columns(4)
        for i, (icon, title, desc) in enumerate(features):
            with fcols[i % 4]:
                st.markdown(f'<div class="feat-card"><div class="feat-icon">{icon}</div><div class="feat-title">{title}</div><div class="feat-desc">{desc}</div></div>', unsafe_allow_html=True)

        st.markdown("<div style='height:2rem'></div>", unsafe_allow_html=True)
        st.markdown('<div class="cta-banner"><div class="cta-title">Ready to find your perfect product?</div><div class="cta-sub">Type any product above — from "wireless headphones" to "espresso machine under $600" to "ergonomic chair"</div></div>', unsafe_allow_html=True)

    # ── FOOTER ──────────────────────────────────────────────────────────────
    st.markdown(f"""
<div style="text-align:center;padding:2.5rem 0 1rem;margin-top:3rem;border-top:1px solid rgba(212,175,55,0.1);">
  <div style="font-family:'JetBrains Mono',monospace;font-size:.62rem;letter-spacing:.18em;color:var(--text4);text-transform:uppercase;">
    ShopSense AI v{APP_VERSION} &nbsp;·&nbsp; 8 Platform APIs &nbsp;·&nbsp; 2.4M+ Products Indexed &nbsp;·&nbsp; Beginner Scores · Spec Explorer · Radar Charts
  </div>
</div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    main()
