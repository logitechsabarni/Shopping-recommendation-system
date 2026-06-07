# ShopSense AI 🛍️

> AI-powered shopping intelligence that searches 8 platforms simultaneously, scores every product with a weighted LLM engine, and delivers actionable recommendations in a dark luxury interface.

---

## Screenshots

```
Hero → Search Bar → Trending Chips → Results Grid (3-col)
     → 7 Tabs: Results · AI Picks · Analytics · Compare · Chat · Wishlist · Export
```

---

## Features

### 🔍 Multi-Platform Search
Searches Amazon, eBay, Walmart, Etsy, Best Buy, Target, Shopify, and Newegg in parallel. Results are unified into a single ranked feed.

### 🤖 AI Scoring Engine
Every product is scored 0–100 using a weighted formula:

| Signal | Weight |
|---|---|
| Star Rating | 30% |
| Review Volume (log-normalized) | 20% |
| Price-Value Ratio | 25% |
| Platform Trust Score | 15% |
| Discount Depth | 10% |

### ✨ Smart Recommendations
Automatically surfaces five curated picks from your results:
- **Best Overall** — highest composite AI score
- **Best Budget** — best score under 45% of budget
- **Best Premium** — best score above 75% of budget
- **Best Value** — best AI score per dollar
- **Highest Rated** — pure community rating

### 📊 Analytics Dashboard
- Price vs AI Score bar chart
- Price vs Rating bubble scatter (bubble size = review count)
- Platform distribution donut
- Deal quality horizontal bar chart
- Rating distribution histogram (Altair)
- Full data table with all fields

### ⚖️ Comparison Center
Add up to 5 products for side-by-side comparison. Displays a grouped bar chart across AI Score, Rating, and Deal Quality. Exports to CSV.

### 💬 AI Chat Assistant
Context-aware conversation engine that understands your search results and answers questions like:
- *"Which has the best warranty?"*
- *"Which ships fastest?"*
- *"Compare the top 2 options"*
- *"Best pick under $400?"*

Supports conversation memory (toggleable). Powered by your chosen LLM engine.

### ❤️ Wishlist
Save up to 50 products across sessions. Remove individual items or clear all. Export as CSV.

### 📁 Export
Download your full result set as **CSV** or **JSON**. Export your wishlist separately.

### 🔌 LLM Adapter Architecture
Plug in your API key for any supported provider — the adapter routes automatically:

| Provider | Model |
|---|---|
| OpenAI | `gpt-4o` |
| Anthropic | `claude-sonnet-4-20250514` |
| Google | `gemini-1.5-pro` |
| LLaMA / Mixtral | Local (bring your own endpoint) |

Without a key, the app runs in **demo mode** with simulated AI output — fully functional for evaluation.

### 🎛️ Advanced Filters
- Max budget slider ($10 – $5,000)
- Minimum star rating
- Minimum review count
- Minimum discount %
- Sort by: AI Score · Price ↑↓ · Rating · Reviews · Discount

---

## Tech Stack

| Layer | Library |
|---|---|
| UI & Server | Streamlit ≥ 1.35 |
| Data | Pandas ≥ 2.0 |
| Charts | Plotly ≥ 5.18 |
| Histogram | Altair ≥ 5.2 |
| HTTP (LLM calls) | Requests ≥ 2.31 |
| Typing / Models | Python dataclasses, typing |

---

## Quick Start

### 1. Clone or download

```bash
git clone https://github.com/yourname/shopsense-ai.git
cd shopsense-ai
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## Project Structure

```
shopsense-ai/
├── app.py            # Entire application (single file)
├── requirements.txt  # Python dependencies
└── README.md         # This file
```

Everything lives in `app.py`, organized into logical sections:

```
CONFIG                  → constants, weights, platform config
DATA MODELS             → Product, ChatMessage, SearchRecord dataclasses
GLOBAL STYLES           → inject_css() — full dark luxury theme
SESSION MANAGEMENT      → init_session()
MOCK PRODUCT DATABASE   → _RAW_DB — 50+ realistic products across 8 categories
SEARCH ENGINE           → build_products(), sort_products(), compute_ai_score()
AI RECOMMENDATION ENGINE→ generate_recommendations(), generate_ai_insight()
PRICE ANALYTICS ENGINE  → price_analytics(), Plotly/Altair chart builders
COMPARISON ENGINE       → render_comparison_table()
CHAT ENGINE             → generate_chat_response()
LLM ADAPTER             → LLMAdapter class (OpenAI / Claude / Gemini / mock)
UI COMPONENTS           → render_product_card(), render_rec_badge(), helpers
STREAMLIT APP           → main() — sidebar, hero, search, 7 tabs, footer
```

---

## Configuration

All tunable constants are at the top of `app.py` under `# CONFIG`:

```python
MAX_COMPARE_ITEMS = 5          # Max products in comparison center
MAX_WISHLIST_ITEMS = 50        # Max saved wishlist items
MAX_SEARCH_HISTORY = 20        # Queries kept in session history
DEFAULT_BUDGET = 500           # Default budget slider value

SCORE_WEIGHTS = {
    "rating":        0.30,
    "review_volume": 0.20,
    "price_value":   0.25,
    "platform_trust":0.15,
    "deal_quality":  0.10,
}

PLATFORM_TRUST = {
    "Amazon": 0.95,
    "Best Buy": 0.90,
    ...
}
```

---

## Adding a Real Shopping API

The search engine is in `build_products()`. Replace or extend `_get_raw_products()` to call a live API:

```python
def _get_raw_products(query_key: str) -> List[Dict]:
    # Example: call Amazon Product Advertising API
    response = requests.get(
        "https://api.amazon.com/paapi5/searchitems",
        headers={"Authorization": f"Bearer {AMAZON_KEY}"},
        params={"Keywords": query_key, "Resources": ["Offers","Images"]},
    )
    return response.json()["SearchResult"]["Items"]
```

The rest of the scoring, ranking, and UI pipeline requires no changes.

---

## Adding a New LLM Provider

Extend `LLMAdapter` in `app.py`:

```python
def _my_provider_complete(self, prompt: str, system: str, max_tokens: int) -> str:
    r = requests.post(
        "https://api.myprovider.com/v1/chat",
        headers={"Authorization": f"Bearer {self.api_key}"},
        json={"prompt": prompt, "max_tokens": max_tokens},
        timeout=15,
    )
    return r.json()["text"]
```

Then add the new provider name to `LLM_ADAPTERS` and wire it in `complete()`.

---

## Design System

| Token | Value |
|---|---|
| Background | `#0a0a0f` |
| Surface | `#0f0e1a` |
| Gold accent | `#d4af37` |
| Gold muted | `rgba(212,175,55,0.18)` |
| Text primary | `#f0ece4` |
| Text secondary | `#c8c4ba` |
| Text muted | `#888` |
| Green (positive) | `#64dc82` |
| Red (negative) | `#ff6060` |
| Display font | Playfair Display (serif) |
| Body font | DM Sans |
| Mono font | JetBrains Mono |

---

## License

MIT — free to use, modify, and distribute.

---

*ShopSense AI v2.0.0 · Built with Streamlit · 8 Platform APIs · 2.4M+ Products Indexed*
