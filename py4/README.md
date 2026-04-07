# مانسا · MANSA — Gold Intelligence Platform

<p align="center">
  <img src="https://img.shields.io/badge/version-4.0.0-gold" />
  <img src="https://img.shields.io/badge/python-3.11-blue" />
  <img src="https://img.shields.io/badge/streamlit-1.35+-red" />
  <img src="https://img.shields.io/badge/languages-5-green" />
  <img src="https://img.shields.io/badge/pages-27-purple" />
</p>

A professional multi-page gold-trading intelligence dashboard — the **only platform** serving Arab gold markets in native Arabic, with Zakat calculation, 16-country price coverage, and AI predictions.

---

## ✨ Features

| Category | Features |
|----------|----------|
| **Live Prices** | 16 Arab & international markets · All purities (14K–24K) · Real-time via Twelve Data or Yahoo fallback |
| **AI Predictions** | Linear Regression · Random Forest · Gradient Boosting · XGBoost · LSTM · Prophet |
| **Languages** | Arabic (RTL) · English · Français · Türkçe · Urdu (RTL) |
| **Themes** | Islamic & Arab Civilization · Ancient Gold Coin · Trading Floor |
| **Pages** | Dashboard · Markets · Charts · Simulator · AI Predictions · Data Explorer · AI Advisor · Portfolio · Calculator · Economic Calendar · Sentiment · Sessions · Price Alerts · Heatmap · MANSA Score · Zakat · Asset Comparison · Trade Journal · Gold Map · Stress Test · Supply & Demand · Currency Converter · Trading Signals · News Sentiment · Game · About · Settings |
| **Unique** | Zakat calculator · Arab market prices · Islamic UI themes · RTL layout · Multi-purity matrix |

---

## 🚀 Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/mansa-gold.git
cd mansa-gold

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure secrets (copy template and fill in values)
mkdir -p .streamlit
cp secrets.toml.template .streamlit/secrets.toml
# → Edit .streamlit/secrets.toml with your API keys

# 5. Run
streamlit run mansa_dashboard.py
```

---

## ⚙️ Configuration

Copy `secrets.toml.template` to `.streamlit/secrets.toml` and fill in:

| Key | Purpose | Where to get it |
|-----|---------|-----------------|
| `TWELVE_DATA_API_KEY` | Real-time prices (<1 min latency) | [twelvedata.com/pricing](https://twelvedata.com/pricing) — free tier available |
| `SUPABASE_URL` + `SUPABASE_KEY` | Cross-session persistence (portfolio, alerts, journal) | [supabase.com](https://supabase.com) — free tier available |
| `ANTHROPIC_API_KEY` | AI Gold Advisor chatbot | [console.anthropic.com](https://console.anthropic.com) |

All three are **optional** — the dashboard works without any of them using Yahoo Finance and session-only storage.

---

## 🗄️ Supabase Setup

Run this SQL once in your Supabase project's SQL Editor:

```sql
create table if not exists mansa_user_data (
  user_id      text primary key,
  portfolio    jsonb default '[]'::jsonb,
  alerts       jsonb default '[]'::jsonb,
  journal      jsonb default '[]'::jsonb,
  profile      jsonb default '{}'::jsonb,
  updated_at   timestamptz default now()
);

alter table mansa_user_data enable row level security;

create policy "Users can manage own data"
  on mansa_user_data for all
  using  (user_id = current_user)
  with check (user_id = current_user);
```

---

## 📊 Data Sources

| Source | Latency | Activation |
|--------|---------|------------|
| **Twelve Data** | < 1 minute | Set `TWELVE_DATA_API_KEY` |
| **Yahoo Finance** | ~15 minutes | Automatic fallback |
| **Static fallback** | — | When market is closed or both sources fail |

---

## 🚢 Deploy to Streamlit Cloud

1. Push to GitHub (`.streamlit/secrets.toml` is in `.gitignore` — never committed)
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app
3. Select your repo and `mansa_dashboard.py`
4. Click **Advanced settings** → paste the contents of your `secrets.toml`
5. Deploy

---

## 🏗️ Project Structure

```
mansa-gold/
├── mansa_dashboard.py          # Main application (9,700+ lines)
├── requirements.txt            # Production dependencies
├── secrets.toml.template       # Configuration template (safe to commit)
├── .gitignore                  # Excludes secrets, models, cache
├── .streamlit/
│   └── config.toml             # Streamlit server configuration
├── models/                     # ML model files (git-ignored)
│   ├── model1_lr.pkl
│   ├── model2_rf.pkl
│   ├── model3_gb.pkl
│   ├── model4_xgb.pkl
│   ├── model5_lstm.keras
│   ├── model6_prophet.pkl
│   └── r2_scores.csv
└── merged_financial_data.csv   # Training data (1990–present)
```

---

## 📄 License

© 2025 Own Al Ansari — All rights reserved.

---

*Inspired by Mansa Musa · The Golden King of Mali · 1312 CE*
