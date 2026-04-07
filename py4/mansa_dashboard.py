"""
MANSA — Gold Intelligence Platform
====================================
A professional multi-page gold-trading dashboard built with Streamlit.

Features
--------
- Real-time gold prices across 16 Arab and international markets
- Five AI/ML prediction models (Linear Regression, Random Forest,
  Gradient Boosting, XGBoost, LSTM, Prophet)
- Full multi-language support: Arabic · English · Français · Türkçe · Urdu
- Three visual themes: Islamic & Arab Civilization, Ancient Gold Coin,
  Trading Floor
- 27 interactive pages: Dashboard, Markets, Charts, Simulator, AI Predictions,
  Data Explorer, AI Advisor, Portfolio, Calculator, Economic Calendar,
  Sentiment Index, Market Sessions, Price Alerts, Correlation Heatmap,
  MANSA Score, Zakat Calculator, Asset Comparison, Trade Journal,
  Gold Map, Stress Test, Supply & Demand, Currency Converter,
  Trading Signals, News Sentiment, Game, About, Settings
- Live ticker strip, auto-refresh, mobile-responsive layout
- RTL/LTR layout switching per language

Usage
-----
    streamlit run mansa_dashboard.py

Configuration
-------------
Set ANTHROPIC_API_KEY in .streamlit/secrets.toml or as an environment
variable to enable the AI Gold Advisor chatbot.

Author
------
Own Al Ansari — © 2025 MANSA Gold Intelligence Platform
"""

# ── Standard library ──────────────────────────────────────────────────────────
import logging
import os
import datetime
import warnings
from typing import Optional

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import streamlit.components.v1
import yfinance as yf

# ── Suppress noisy third-party warnings ───────────────────────────────────────
warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
_log = logging.getLogger("mansa")

# ── Application metadata ──────────────────────────────────────────────────────
__version__  = "4.0.0"
__author__   = "Own Al Ansari"
__license__  = "Proprietary"

# Mobile-friendly viewport
st.set_page_config(
    page_title="مانسا · ذكاء الذهب",
    page_icon="☽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Anthropic API key ──────────────────────────────────────────────────────────
# Priority: st.secrets["ANTHROPIC_API_KEY"] → env var → session state (user input)
def get_api_key() -> str:
    """Resolve Anthropic API key from secrets → env var → session state."""
    # 1. Streamlit secrets (add to .streamlit/secrets.toml)
    try:
        return st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        _log.debug("Suppressed %s", exc_info=True)
    # 2. Environment variable
    k = os.environ.get("ANTHROPIC_API_KEY", "")
    if k:
        return k
    # 3. Session state (entered by user in sidebar)
    return st.session_state.get("_api_key", "")

# ═══════════════════════════════════════════════════════════════════════════════
# LANGUAGE STRINGS
# ═══════════════════════════════════════════════════════════════════════════════
LANGS = {
    "العربية 🇸🇦": {
        "dir": "rtl",
        "app_name": "مانسا",
        "tagline": "ذكاء أسعار الذهب",
        "nav_dashboard": "🏠  لوحة التحكم",
        "nav_markets":   "🌍  الأسواق",
        "nav_charts":    "📈  الرسوم البيانية",
        "nav_simulator": "🔭  المحاكي",
        "nav_predictions":"🤖  توقعات الذكاء الاصطناعي",
        "nav_data":      "📂  مستكشف البيانات",
        "nav_advisor":   "💬  مستشار الذهب الذكي",
        "nav_settings":  "⚙️  الإعدادات",

        "nav_signals":  "📡  إشارات التداول",
        "nav_newssent": "🧠  تحليل الأخبار",
        "nav_supply":  "⛏️  العرض والطلب",
        "nav_currency":"💱  محول العملات",
        "nav_alerts":     "🔔  تنبيهات الأسعار",
        "nav_heatmap":    "📊  خريطة الترابط",
        "nav_mansa_score":"🏆  مؤشر مانسا",
        "nav_zakat":      "☪️  حاسبة الزكاة",
        "nav_compare":    "📈  مقارنة الأصول",
        "nav_journal":    "📓  سجل الصفقات",
        "nav_goldmap":    "🌍  خريطة الذهب",
        "nav_drawdown":   "📉  اختبار الضغط",
        "nav_portfolio":  "💼  المحفظة",
        "nav_calculator": "🧮  الحاسبة",
        "nav_calendar":   "📅  التقويم الاقتصادي",
        "nav_sentiment":  "🌡️  مؤشر المشاعر",
        "nav_sessions":   "🕐  جلسات السوق",
        "nav_game":   "🎮  لعبة مانسا",
        "nav_about":  "🏛️  عن مانسا",        "live": "مباشر",
        "not_financial": "ليس نصيحة مالية",
        "quick_settings": "إعدادات سريعة",
        "weight_unit": "وحدة الوزن",
        "purity": "عيار الذهب",
        "theme": "المظهر",
        "auto_refresh": "تحديث تلقائي (60 ث)",
        "primary_market": "السوق الرئيسي",
        "gold_intelligence": "منصة ذكاء الذهب",
        "real_time": "أسعار مباشرة · الأسواق العربية والعالمية",
        "spot_price": "السعر الفوري العالمي",
        "key_stats": "إحصائيات الذهب الرئيسية",
        "wk52_high": "أعلى سعر 52 أسبوع",
        "wk52_low": "أدنى سعر 52 أسبوع",
        "wk52_avg": "متوسط 52 أسبوع",
        "ytd_return": "العائد منذ بداية العام",
        "gold_silver": "نسبة ذهب/فضة",
        "gold_oil": "نسبة ذهب/نفط",
        "market_overview": "نظرة عامة على السوق",
        "stocks_indices": "الأسهم والمؤشرات",
        "all_purities": "جميع العيارات",
        "prediction_date": "تاريخ التوقع",
        "tomorrow": "غد",
        "best_model": "أفضل نموذج",
        "algo_predictions": "توقعات الخوارزميات",
        "consensus": "التوافق",
        "model_high": "أعلى توقع",
        "model_low": "أدنى توقع",
        "spread": "الفارق",
        "vs_spot": "مقابل السعر الحالي",
        "pred_multi_unit": "التوقع بعدة وحدات وعملات",
        "no_models": "لم يتم العثور على نماذج. الرجاء تشغيل train_models.py أولاً",
        "r2_scores": "دقة النماذج · معامل R²",
        "prophet_forecast": "توقع Prophet للأيام الـ 30 القادمة",
        "select_model": "اختر النموذج",
        "disclaimer": "تنبيه: هذه المعلومات للأغراض التعليمية فقط وليست نصيحة مالية.",
        "arab_markets": "الأسواق العربية",
        "intl_markets": "الأسواق الدولية",
        "purity_matrix": "مصفوفة العيارات والأسواق",
        "data_explorer": "مستكشف البيانات",
        "training_data": "بيانات التدريب المستخدمة لبناء النماذج",
        "total_rows": "إجمالي الصفوف",
        "features": "المؤشرات",
        "from_date": "من",
        "to_date": "إلى",
        "filter_year": "تصفية بالسنة",
        "columns_display": "الأعمدة المعروضة",
        "rows_show": "عدد الصفوف",
        "feat_desc": "وصف المؤشرات",
        "gold_history": "تاريخ أسعار الذهب",
        "trading_advisor": "مستشار التداول",
        "live_signals": "إشارات مباشرة",
        "price_levels": "مستويات الأسعار",
        "trade_setup": "إعداد الصفقة",
        "personalised_advice": "نصائح مخصصة",
        "golden_rules": "القواعد الذهبية العشر للتداول",
        "macro_env": "البيئة الاقتصادية الكلية للذهب",
        "overall_signal": "الإشارة الإجمالية للسوق",
        "bullish": "صاعد",
        "bearish": "هابط",
        "neutral": "محايد",
        "support": "دعم",
        "resistance": "مقاومة",
        "entry": "نقطة الدخول",
        "stop_loss": "وقف الخسارة",
        "take_profit": "جني الأرباح",
        "settings": "الإعدادات",
        "design_theme": "سمة التصميم",
        "display_prefs": "تفضيلات العرض",
        "show_purity_table": "عرض جدول العيارات في لوحة التحكم",
        "default_period": "الفترة الزمنية الافتراضية للرسم",
        "active_markets": "الأسواق النشطة",
        "active_stocks": "الأسهم والمؤشرات النشطة",
        "conversion_ref": "مرجع تحويل الوحدات والعيارات",
        "not_found_data": "لم يتم العثور على بيانات التدريب.",
        "profile": "ملف التاجر",
        "profile_opts": ["مبتدئ 🌱", "متوسط 📊", "متقدم ⚡", "مستثمر طويل الأجل 🏛️"],
        "chatbot_title": "💬  مستشار الذهب الذكي",
        "chatbot_sub": "اسألني أي شيء عن تداول الذهب · مدعوم بالذكاء الاصطناعي",
        "chatbot_placeholder": "اكتب سؤالك هنا... مثل: هل يجب أن أشتري الذهب الآن؟",
        "chatbot_send": "إرسال",
        "chatbot_clear": "مسح المحادثة",
        "chatbot_thinking": "جاري التفكير...",
        "chatbot_welcome": "مرحباً! أنا مستشار الذهب الذكي. يمكنني مساعدتك في: متى تشتري/تبيع الذهب، تحليل الأسعار، استراتيجيات التداول، وكل ما يتعلق بسوق الذهب. اسألني!",
        "chatbot_error": "عذراً، حدث خطأ. يرجى المحاولة مرة أخرى.",
        "models_demo_note": "⚠️ نماذج AI غير موجودة. يتم عرض توقعات تجريبية بناءً على بيانات التدريب.",
        "no_csv": "لم يتم العثور على ملف البيانات.",
        "indicators": "مؤشرات",
        "period": "الفترة",
        "chart_type": "نوع الرسم",
        "live_prices": "أسعار مباشرة",
        "yf_delay": "Yahoo Finance · ~15 دقيقة تأخير",
        "live_snapshot": "📊 جدول الأسعار المباشرة",
        "col_asset": "الأصل",
        "col_price": "السعر",
        "col_change": "التغيير",
        "col_status": "حالة",
        "data_up_to": "آخر بيانات:",
        "added": "أُضيف",
        "new_rows_to": "سطر جديد · البيانات محدّثة حتى",
        "update_error": "خطأ أثناء التحديث:",
        "latest": "آخر",
        "rows_added": "سطور مضافة",
        "rows_filtered": "صف مرشح",
        "total_lc": "إجمالي",
        "sparklines": "مخططات صغيرة · آخر سنة",
        "quick_questions": "أسئلة سريعة",
        "brand_logos": "الشعارات · واحدة لكل تصميم",
        "developer": "المطوّر والمنشئ",
        "why_mansa": "لماذا مانسا؟ · قصة مانسا موسى",
        "mansa_title": "مانسا موسى — أغنى رجل في التاريخ",
        "hist_highlights": "لمحات تاريخية",
        "mansa_caption": "مانسا موسى · إمبراطور مالي",
        "mansa_dates": "1280 – 1337 م",
        "mosque_caption": "مسجد جنغريبر · تمبكتو · مالي",
        "mosque_built": "بناه مانسا موسى عام 1327م",
        "empire_lbl": "الإمبراطورية",
        "trade_routes": "طرق التجارة",
        "empire_peak": "إمبراطورية مالي في أوجها",
        "empire_century": "القرن الرابع عشر الميلادي · 2 مليون كم²",
        "about_platform": "عن المنصة · من تطوير عون الانصاري",
        "copyright": "© 2025 عون الانصاري",
        "platform_name": "منصة مانسا لذكاء الذهب",
        "position_details": "الصفقات التفصيلية",
        "results": "النتائج",
        "risk_amount": "المخاطرة بالدولار",
        "position_size_lbl": "حجم الصفقة",
        "position_cost": "تكلفة الصفقة",
        "risk_reward": "نسبة المكافأة/المخاطرة",
        "calc_pnl_sub": "احسب الربح أو الخسارة المتوقعة",
        "cost_lbl": "التكلفة",
        "value_lbl": "القيمة",
        "calc_margin_sub": "احسب الهامش المطلوب للتداول بالرافعة المالية",
        "notional": "القيمة الاسمية",
        "req_margin": "الهامش المطلوب",
        "per_dollar": "قيمة النقطة",
        "leverage_lbl": "الرافعة",
        "calc_be_sub": "احسب سعر التعادل مع تكاليف التداول",
        "breakeven": "سعر التعادل",
        "above_entry": "فوق سعر الدخول",
        "spread_lbl": "الفارق",
        "commission_lbl": "العمولة",
        "swap_lbl": "تكلفة الترحيل",
        "today_lbl": "اليوم ⚡",
        "forecast_lbl": "التوقع:",
        "actual_lbl": "الفعلي:",
        "fear_lbl": "خوف",
        "greed_lbl": "جشع",
        "what_means_gold": "ماذا يعني هذا للذهب؟",
        "index_components": "مكونات المؤشر",
        "market_status_now": "حالة الأسواق الآن",
        "arab_mkts_primary": "⭐ الأسواق العربية · الأردن السوق الرئيسي",
        "intl_mkts_lbl": "🌐 الأسواق الدولية",
        "timeline_24h": "الجدول الزمني 24 ساعة (UTC)",
        "gold_corr_assets": "🥇 ارتباط الذهب مع كل أصل",
        "five_pillars": "الركائز الخمس",
        "enter_holdings": "إدخال ممتلكاتك الذهبية",
        "gold24k": "ذهب عيار 24 (جرام)",
        "gold22k": "ذهب عيار 22 (جرام)",
        "gold21k": "ذهب عيار 21 (جرام)",
        "gold18k": "ذهب عيار 18 (جرام)",
        "gold14k": "ذهب عيار 14 (جرام)",
        "zakat_result": "نتيجة حساب الزكاة",
        "total_gold_lbl": "مجموع الذهب",
        "nisab_lbl": "النصاب",
        "total_value_lbl": "القيمة الإجمالية",
        "zakat_due": "الزكاة الواجبة (2.5%)",
        "returns_lb": "🏆 ترتيب العوائد",
        "trade_log": "سجل الصفقات",
        "nisab_based": "بناءً على سعر الفضة المباشر: الذهب 24K ",
        "current_price_lbl": "السعر الحالي",
        "interactive_sim": "محاكاة تفاعلية",
        "cb_reserves": "🏦 احتياطيات الذهب لدى البنوك المركزية (أطنان) · بيانات 2025",
        "mine_prod": "⛏️ إنتاج مناجم الذهب (أطنان/سنة 2024)",
        "global_demand": "📊 الطلب العالمي على الذهب (أطنان 2024)",
        "major_etfs": "📊 صناديق ETF الذهبية الكبرى",
        "gram_arab": "سعر الجرام الآن بعملات المنطقة العربية",
        "auto_refresh_60": "أسعار محدثة تلقائياً كل دقيقة",
        "overall_signal_ts": "إشارة مانسا الإجمالية · ",
        "confidence_lbl": "ثقة",
        "buy_lbl": "شراء",
        "sell_lbl": "بيع",
        "all_signals": "📋 تفاصيل جميع الإشارات",
        "action_buy": "الإجراء المقترح: شراء",
        "entry_lbl": "دخول",
        "action_sell": "الإجراء المقترح: بيع أو انتظار",
        "exit_sell": "خروج/بيع عند",
        "action_wait": "الإجراء المقترح: انتظار إشارة أوضح",
        "bullish_lbl": "صاعد",
        "neutral_lbl": "محايد",
        "bearish_lbl": "هابط",
        "very_bearish": "هابط جداً",
        "average_lbl": "المتوسط",
        "very_bullish": "صاعد جداً",
        "article_analysis": "📰 تحليل كل خبر",
        "session_score": "النقاط في هذه الجلسة",
        "high_score_lbl": "أعلى نقاط",
        "start_game": "ابدأ اللعبة",
        "amazing_10": "رائع! عشر قطع ذهب!",
        "game_over": "انتهت اللعبة · Game Over",
        "gold_collected": "قطع الذهب التي جمعتها",
        "play_again": "إعادة اللعب",
        "restart_lbl": "إعادة",
        "calc_pos_sub": "احسب حجم الصفقة المناسب بناءً على المخاطرة",
        "tomorrow_lbl": "غداً",
        "days_lbl": " يوم",
        "neutral_sig": "محايد",
        "stop_loss_lbl": "وقف الخسارة",
        "news_sources": "يتجدد تلقائياً كل 15 دقيقة · المصادر: Reuters · Kitco · FXStreet · MarketWatch",
        "game_instructions": "⬆ اضغط SPACE أو انقر للقفز · اجمع قطع الذهب · تحدَّ نفسك وحقق أعلى نقاط!",
        "update_hs": "🏆 تحديث أعلى نقاط",
        "feat_info": "معلومات المؤشرات وعلاقتها بالذهب",
        "history_lbl": "📖 التاريخ",
        "relation_gold": "🔗 العلاقة بالذهب",
        "closes_in": "يغلق خلال",
        "opens_in": "يفتح خلال",
        "neutral_lc": "محايد",
        "add_new_alert": "إضافة تنبيه جديد",
        "target_price": "السعر المستهدف (USD/oz)",
        "direction_lbl": "الاتجاه",
        "alert_label": "وصف التنبيه",
        "why_level": "لماذا هذا السعر مهم؟",
        "save_alert": "حفظ التنبيه",
        "alert_saved": "🔔 تم حفظ التنبيه!",
        "alert_triggered_lbl": "تنبيه مُفعَّل!",
        "current_lbl": "السعر الحالي",
        "no_alerts_msg": "لا توجد تنبيهات نشطة — أضف تنبيهاً بسعر مستهدف أعلاه",
        "clear_triggered": "مسح المُفعَّلة",
        "alert_at": "إنذار عند",
        "distance_lbl": "المسافة",
        "nav_demo": "🎯  التداول التجريبي",
        "nav_savings": "🪙  خطة ادخار الذهب",
        "nav_report": "📋  التقرير الأسبوعي",
        "nav_widget": "🔌  أداة التضمين",
        "nav_cb": "🏦  احتياطيات البنوك المركزية",
        "demo_title": "التداول التجريبي",
        "demo_sub": "تداول بأموال وهمية · بدون مخاطر حقيقية",
        "demo_balance": "الرصيد التجريبي",
        "demo_buy": "شراء ذهب",
        "demo_sell": "بيع ذهب",
        "demo_reset": "إعادة تعيين الحساب",
        "demo_pnl": "الربح / الخسارة",
        "demo_holdings": "المحفظة التجريبية",
        "demo_trades": "سجل الصفقات التجريبية",
        "demo_note": "💡 هذا حساب تجريبي — لا تُستخدم أموال حقيقية",
        "demo_qty": "الكمية (جرام)",
        "demo_cleared": "تم إعادة تعيين الحساب التجريبي",
        "savings_title": "خطة ادخار الذهب",
        "savings_sub": "ادخر بالذهب شهرياً · حماية من التضخم",
        "savings_monthly": "المبلغ الشهري",
        "savings_since": "تاريخ البدء",
        "savings_curr": "العملة",
        "savings_add": "إضافة خطة",
        "savings_total": "إجمالي المدخرات",
        "savings_gold": "الذهب المتراكم",
        "savings_value": "القيمة الحالية",
        "savings_gain": "الربح مقابل الادخار النقدي",
        "savings_vs": "مقارنة: الذهب مقابل الادخار النقدي",
        "savings_note": "💡 الادخار بالذهب يحميك من تآكل قيمة العملة",
        "savings_plan": "خطتي الادخارية",
        "savings_delete": "حذف الخطة",
        "report_title": "التقرير الأسبوعي",
        "report_sub": "ملخص أسواق الذهب · يُولَّد تلقائياً",
        "report_gen": "توليد التقرير",
        "report_dl": "تحميل PDF",
        "report_week": "تقرير أسبوع",
        "report_perf": "الأداء الأسبوعي",
        "report_outlook": "التوقعات للأسبوع القادم",
        "report_events": "الأحداث الاقتصادية القادمة",
        "report_signals": "ملخص الإشارات",
        "widget_title": "أداة التضمين",
        "widget_sub": "أضف سعر الذهب إلى موقعك بسطر واحد",
        "widget_copy": "نسخ الكود",
        "widget_preview": "معاينة الأداة",
        "widget_style": "النمط",
        "widget_curr2": "العملة",
        "widget_copied": "✅ تم النسخ!",
        "widget_instruct": "انسخ هذا الكود والصقه في موقعك",
        "cb_title": "احتياطيات البنوك المركزية",
        "cb_sub": "احتياطيات الذهب لدول العالم · بيانات WGC 2025",
        "cb_arab": "الدول العربية",
        "cb_world": "أكبر دول العالم",
        "cb_rank": "الترتيب",
        "cb_country": "الدولة",
        "cb_tonnes": "الكمية (طن)",
        "cb_pct": "% من الاحتياطيات",
        "cb_trend": "الاتجاه",
        "cb_insight": "تحليل",
        "persona_select": "اختر نوع المستخدم",
        "persona_trader": "📈 متداول",
        "persona_investor": "💰 مستثمر",
        "persona_shop": "🏪 صاحب محل ذهب",
        "persona_factory": "🏭 مصنع مجوهرات",
        "persona_designer": "💍 مصمم / حرفي",
        "persona_buyer": "🛍️ مشتري",
        "persona_lbl": "الشخصية",
        "nav_shopboard": "🏪  لوحة المحل",
        "nav_invoice": "🧾  حاسبة الفاتورة",
        "nav_production": "🏭  تكلفة الإنتاج",
        "nav_fairprice": "⚖️  فحص السعر العادل",
        "nav_piecepricing": "💍  تسعير المقطوعة",
        "shop_buy": "سعر الشراء",
        "shop_sell": "سعر البيع",
        "shop_spread": "هامش المحل",
        "shop_display": "وضع العرض",
        "shop_currency": "عملة العرض",
        "shop_name": "اسم المحل",
        "shop_purity": "عرض جميع العيارات",
        "shop_fullscreen": "ملء الشاشة",
        "shop_embed": "تضمين في الموقع",
        "inv_weight": "الوزن (جرام)",
        "inv_purity": "العيار",
        "inv_making": "أجر الصنعة (لكل جرام)",
        "inv_vat": "ضريبة القيمة المضافة %",
        "inv_gold_val": "قيمة الذهب",
        "inv_making_val": "إجمالي الصنعة",
        "inv_vat_val": "الضريبة",
        "inv_total": "الإجمالي",
        "inv_save": "حفظ الفاتورة",
        "inv_history": "سجل الفواتير",
        "inv_clear": "مسح السجل",
        "prod_recipe": "وصفة السبيكة",
        "prod_gold_pct": "نسبة الذهب %",
        "prod_weight": "وزن القطعة المطلوب (جرام)",
        "prod_wastage": "نسبة الهدر %",
        "prod_labour": "تكلفة العمالة (لكل جرام)",
        "prod_overhead": "المصاريف العامة",
        "prod_gold_cost": "تكلفة الذهب الخام",
        "prod_total_cost": "إجمالي تكلفة الإنتاج",
        "prod_min_price": "الحد الأدنى للبيع",
        "prod_suggest": "السعر المقترح (ربح 30%)",
        "prod_batch": "تكلفة الدفعة",
        "prod_qty": "عدد القطع",
        "fp_quoted": "السعر المعروض عليك",
        "fp_weight": "الوزن (جرام)",
        "fp_purity": "العيار",
        "fp_currency": "العملة",
        "fp_spot_val": "القيمة بسعر الصرف",
        "fp_making_est": "الصنعة المقدرة",
        "fp_verdict_fair": "سعر عادل ✅",
        "fp_verdict_high": "سعر مرتفع ⚠️",
        "fp_verdict_low": "سعر منخفض جداً 🤔",
        "fp_saving": "توفيرك",
        "fp_overpay": "زيادة تدفعها",
        "piece_gold_g": "وزن الذهب (جرام)",
        "piece_purity": "العيار",
        "piece_gems": "أحجار كريمة",
        "piece_gem_add": "إضافة حجر",
        "piece_labour_hrs": "ساعات العمل",
        "piece_hourly": "أجر الساعة",
        "piece_overhead": "المصاريف العامة",
        "piece_margin": "هامش الربح المطلوب %",
        "piece_cost": "إجمالي التكلفة",
        "piece_price": "السعر المقترح",
        "piece_save": "حفظ التصميم",
        "piece_designs": "تصاميمي المحفوظة",
        "geo_news": "آخر الأخبار الجيوسياسية",
        "geo_timeline": "الجدول الزمني للأحداث",
        "geo_filter": "تصفية حسب التأثير",
        "geo_all": "الكل",
        "date_from": "من تاريخ",
        "date_to": "إلى تاريخ",
        "quick_range": "فترة سريعة",
        "nav_geo": "🌍  الخريطة الجيوسياسية",
        "nav_oilgold": "🛢️  النفط والذهب",
        "geo_title": "الخريطة الجيوسياسية",
        "geo_sub": "مناطق التوتر حول العالم وتأثيرها على أسعار الذهب",
        "geo_risk": "مؤشر المخاطر الجيوسياسية",
        "geo_impact": "التأثير على الذهب",
        "geo_events": "الأحداث النشطة",
        "geo_bullish": "داعم للذهب 🟢",
        "geo_bearish": "ضاغط على الذهب 🔴",
        "geo_neutral": "محايد 🟡",
        "og_title": "النفط والذهب",
        "og_sub": "الارتباط بين النفط والذهب · مؤشر حي",
        "og_corr": "معامل الارتباط",
        "og_diverge": "تحذير: انفصال غير معتاد",
        "og_signal": "إشارة الارتباط",
        "sig_entry": "سعر الدخول",
        "sig_target": "الهدف",
        "sig_stop": "وقف الخسارة",
        "sig_rr": "نسبة المخاطرة/العائد",
        "sig_trade": "تفاصيل الصفقة",
    },
    "English 🇬🇧": {
        "dir": "ltr",
        "app_name": "MANSA",
        "tagline": "Gold Intelligence",
        "nav_dashboard": "🏠  Dashboard",
        "nav_markets":   "🌍  Markets",
        "nav_charts":    "📈  Charts & Analysis",
        "nav_simulator": "🔭  Simulator",
        "nav_predictions":"🤖  AI Predictions",
        "nav_data":      "📂  Data Explorer",
        "nav_advisor":   "💬  AI Gold Advisor",
        "nav_settings":  "⚙️  Settings",

        "nav_signals":  "📡  Trading Signals",
        "nav_newssent": "🧠  News Sentiment",
        "nav_supply":  "⛏️  Supply & Demand",
        "nav_currency":"💱  Currency Converter",
        "nav_alerts":     "🔔  Price Alerts",
        "nav_heatmap":    "📊  Correlation Heatmap",
        "nav_mansa_score":"🏆  MANSA Score",
        "nav_zakat":      "☪️  Zakat Calculator",
        "nav_compare":    "📈  Asset Comparison",
        "nav_journal":    "📓  Trade Journal",
        "nav_goldmap":    "🌍  Gold Map",
        "nav_drawdown":   "📉  Stress Test",
        "nav_portfolio":  "💼  Portfolio",
        "nav_calculator": "🧮  Calculator",
        "nav_calendar":   "📅  Economic Calendar",
        "nav_sentiment":  "🌡️  Sentiment",
        "nav_sessions":   "🕐  Market Sessions",
        "nav_game":   "🎮  Mansa Game",
        "nav_about":  "🏛️  About Mansa",        "live": "Live",
        "not_financial": "Not financial advice",
        "quick_settings": "Quick Settings",
        "weight_unit": "Weight Unit",
        "purity": "Gold Purity",
        "theme": "Design Theme",
        "auto_refresh": "Auto-refresh (60s)",
        "primary_market": "Primary Market",
        "gold_intelligence": "Gold Intelligence Platform",
        "real_time": "Real-time · Arab & global markets",
        "spot_price": "Global Spot Price",
        "key_stats": "Gold Key Statistics",
        "wk52_high": "52W High",
        "wk52_low": "52W Low",
        "wk52_avg": "52W Average",
        "ytd_return": "YTD Return",
        "gold_silver": "Gold/Silver Ratio",
        "gold_oil": "Gold/Oil Ratio",
        "market_overview": "Market Overview",
        "stocks_indices": "Stocks & Indices",
        "all_purities": "All Purities",
        "prediction_date": "Prediction Date",
        "tomorrow": "Tomorrow",
        "best_model": "Best Model",
        "algo_predictions": "Algorithm Predictions",
        "consensus": "Consensus",
        "model_high": "Model High",
        "model_low": "Model Low",
        "spread": "Spread",
        "vs_spot": "vs Spot",
        "pred_multi_unit": "Prediction in Multiple Units & Currencies",
        "no_models": "No models found. Run train_models.py first.",
        "r2_scores": "Model Accuracy · R² Scores",
        "prophet_forecast": "Prophet 30-Day Forecast",
        "select_model": "Select Model",
        "disclaimer": "Disclaimer: For educational purposes only. Not financial advice.",
        "arab_markets": "Arab Markets",
        "intl_markets": "International Markets",
        "purity_matrix": "Full Purity × Market Matrix",
        "data_explorer": "Data Explorer",
        "training_data": "Training dataset used to build prediction models",
        "total_rows": "Total Rows",
        "features": "Features",
        "from_date": "From",
        "to_date": "To",
        "filter_year": "Filter by year",
        "columns_display": "Columns to display",
        "rows_show": "Rows to show",
        "feat_desc": "Feature Descriptions",
        "gold_history": "Gold Price History",
        "trading_advisor": "Trading Advisor",
        "live_signals": "Live Technical Signals",
        "price_levels": "Key Price Levels",
        "trade_setup": "Trade Setup",
        "personalised_advice": "Personalised Advice",
        "golden_rules": "The 10 Golden Rules of Gold Trading",
        "macro_env": "Current Macro Environment",
        "overall_signal": "Overall Market Signal",
        "bullish": "Bullish",
        "bearish": "Bearish",
        "neutral": "Neutral",
        "support": "Support",
        "resistance": "Resistance",
        "entry": "Entry",
        "stop_loss": "Stop Loss",
        "take_profit": "Take Profit",
        "settings": "Settings",
        "design_theme": "Design Theme",
        "display_prefs": "Display Preferences",
        "show_purity_table": "Show purity table on Dashboard",
        "default_period": "Default Chart Period",
        "active_markets": "Active Markets",
        "active_stocks": "Active Stocks & Indices",
        "conversion_ref": "Unit × Purity Conversion Reference",
        "not_found_data": "Training data not found.",
        "profile": "Trader Profile",
        "profile_opts": ["Beginner 🌱", "Intermediate 📊", "Advanced ⚡", "Long-Term Investor 🏛️"],
        "chatbot_title": "💬  AI Gold Advisor Chat",
        "chatbot_sub": "Ask me anything about gold trading · Powered by AI",
        "chatbot_placeholder": "Type your question... e.g. Should I buy gold now?",
        "chatbot_send": "Send",
        "chatbot_clear": "Clear Chat",
        "chatbot_thinking": "Thinking...",
        "chatbot_welcome": "Hello! I'm your AI Gold Advisor. I can help with: when to buy/sell gold, price analysis, trading strategies, market signals, and anything gold-related. Ask me anything!",
        "chatbot_error": "Sorry, an error occurred. Please try again.",
        "models_demo_note": "⚠️ AI models not found. Showing demo predictions based on training data statistics.",
        "no_csv": "Data file not found.",
        "indicators": "Indicators",
        "period": "Period",
        "chart_type": "Chart Type",
        "live_prices": "Live prices",
        "yf_delay": "Yahoo Finance · ~15 min delay",
        "live_snapshot": "📊 Live Market Snapshot",
        "col_asset": "ASSET",
        "col_price": "PRICE",
        "col_change": "CHANGE",
        "col_status": "STATUS",
        "data_up_to": "Data up to:",
        "added": "Added",
        "new_rows_to": "new rows · Data now up to",
        "update_error": "Update error:",
        "latest": "Latest",
        "rows_added": "rows added",
        "rows_filtered": "rows filtered",
        "total_lc": "total",
        "sparklines": "Sparklines · Last 12 months",
        "quick_questions": "Quick Questions",
        "brand_logos": "Brand Logos · One per Design Theme",
        "developer": "Developer & Creator",
        "why_mansa": "Why Mansa? · The Story of Mansa Musa",
        "mansa_title": "Mansa Musa — The Richest Man in History",
        "hist_highlights": "Historical Highlights",
        "mansa_caption": "Mansa Musa · Emperor of Mali",
        "mansa_dates": "1280 – 1337 CE",
        "mosque_caption": "Djinguereber Mosque · Timbuktu",
        "mosque_built": "Built by Mansa Musa · 1327 CE",
        "empire_lbl": "Empire",
        "trade_routes": "Trade Routes",
        "empire_peak": "Mali Empire at its Peak",
        "empire_century": "14th Century CE · 2 Million km²",
        "about_platform": "About the Platform · By Own Al Ansari",
        "copyright": "© 2025 Own Al Ansari",
        "platform_name": "MANSA Gold Intelligence Platform",
        "position_details": "Position Details",
        "results": "Results",
        "risk_amount": "Risk Amount",
        "position_size_lbl": "Position Size",
        "position_cost": "Position Cost",
        "risk_reward": "Risk/Reward",
        "calc_pnl_sub": "Calculate expected profit or loss",
        "cost_lbl": "Cost",
        "value_lbl": "Value",
        "calc_margin_sub": "Calculate required margin for leveraged trading",
        "notional": "Notional Value",
        "req_margin": "Required Margin",
        "per_dollar": "Value per $1 move",
        "leverage_lbl": "Leverage",
        "calc_be_sub": "Calculate break-even price including trading costs",
        "breakeven": "Break-even Price",
        "above_entry": "above entry",
        "spread_lbl": "Spread",
        "commission_lbl": "Commission",
        "swap_lbl": "Swap cost",
        "today_lbl": "TODAY ⚡",
        "forecast_lbl": "Forecast:",
        "actual_lbl": "Actual:",
        "fear_lbl": "FEAR",
        "greed_lbl": "GREED",
        "what_means_gold": "What does this mean for gold?",
        "index_components": "Index Components",
        "market_status_now": "Market Status Right Now",
        "arab_mkts_primary": "⭐ Arab Markets · Jordan is Primary",
        "intl_mkts_lbl": "🌐 International Markets",
        "timeline_24h": "24-Hour Timeline (UTC)",
        "gold_corr_assets": "🥇 Gold Correlation with Each Asset",
        "five_pillars": "Five Pillars Breakdown",
        "enter_holdings": "Enter Your Gold Holdings",
        "gold24k": "24K Gold (grams)",
        "gold22k": "22K Gold (grams)",
        "gold21k": "21K Gold (grams)",
        "gold18k": "18K Gold (grams)",
        "gold14k": "14K Gold (grams)",
        "zakat_result": "Zakat Calculation Result",
        "total_gold_lbl": "Total Gold",
        "nisab_lbl": "Nisab",
        "total_value_lbl": "Total Value",
        "zakat_due": "Zakat Due (2.5%)",
        "returns_lb": "🏆 Returns Leaderboard",
        "trade_log": "Trade Log",
        "nisab_based": "Based on live spot price: Gold 24K ",
        "current_price_lbl": "Current price",
        "interactive_sim": "Interactive Simulation",
        "cb_reserves": "🏦 Central Bank Gold Reserves (tonnes) · 2025 Data",
        "mine_prod": "⛏️ Gold Mine Production (tonnes/year 2024)",
        "global_demand": "📊 Global Gold Demand Breakdown (tonnes 2024)",
        "major_etfs": "📊 Major Gold ETFs",
        "gram_arab": "Gold Gram Price in Arab Currencies Right Now",
        "auto_refresh_60": "Rates auto-refresh every 60s",
        "overall_signal_ts": "MANSA OVERALL SIGNAL · ",
        "confidence_lbl": "Confidence",
        "buy_lbl": "BUY",
        "sell_lbl": "SELL",
        "all_signals": "📋 All Signal Details",
        "action_buy": "Suggested Action: BUY",
        "entry_lbl": "Entry",
        "action_sell": "Suggested Action: SELL or WAIT",
        "exit_sell": "Exit/Sell at",
        "action_wait": "Suggested Action: WAIT for clearer signal",
        "bullish_lbl": "Bullish",
        "neutral_lbl": "Neutral",
        "bearish_lbl": "Bearish",
        "very_bearish": "Very Bearish",
        "average_lbl": "Average",
        "very_bullish": "Very Bullish",
        "article_analysis": "📰 Individual Article Analysis",
        "session_score": "Session Score",
        "high_score_lbl": "High Score",
        "start_game": "Start Game",
        "amazing_10": "Amazing! 10 gold coins!",
        "game_over": "Game Over ☽",
        "gold_collected": "Gold collected",
        "play_again": "Play Again",
        "restart_lbl": "Restart",
        "calc_pos_sub": "Calculate the right position size based on your risk tolerance",
        "tomorrow_lbl": "Tomorrow",
        "days_lbl": "in ",
        "neutral_sig": "NEUTRAL",
        "stop_loss_lbl": "Stop Loss",
        "news_sources": "Auto-refreshed every 15 min · Sources: Reuters · Kitco · FXStreet · MarketWatch",
        "game_instructions": "⬆ SPACE or click to jump · Collect gold coins · Beat your high score!",
        "update_hs": "🏆 Update High Score",
        "feat_info": "Feature Information · History · Relationships · Signal",
        "history_lbl": "📖 History",
        "relation_gold": "🔗 Relation to Gold",
        "closes_in": "Closes in",
        "opens_in": "Opens in",
        "neutral_lc": "NEUTRAL",
        "add_new_alert": "Add New Alert",
        "target_price": "Target Price (USD/oz)",
        "direction_lbl": "Direction",
        "alert_label": "Alert Label",
        "why_level": "Why is this level important?",
        "save_alert": "Save Alert",
        "alert_saved": "🔔 Alert saved!",
        "alert_triggered_lbl": "ALERT TRIGGERED!",
        "current_lbl": "Current",
        "no_alerts_msg": "No active alerts — add a target price alert above",
        "clear_triggered": "Clear triggered",
        "alert_at": "Alert at",
        "distance_lbl": "Distance",
        "nav_demo": "🎯  Demo Trading",
        "nav_savings": "🪙  Gold Savings Plan",
        "nav_report": "📋  Weekly Report",
        "nav_widget": "🔌  Embed Widget",
        "nav_cb": "🏦  Central Bank Reserves",
        "demo_title": "Demo Trading",
        "demo_sub": "Trade with virtual money · Zero real risk",
        "demo_balance": "Demo Balance",
        "demo_buy": "Buy Gold",
        "demo_sell": "Sell Gold",
        "demo_reset": "Reset Account",
        "demo_pnl": "Profit / Loss",
        "demo_holdings": "Demo Portfolio",
        "demo_trades": "Demo Trade Log",
        "demo_note": "💡 This is a demo account — no real money is used",
        "demo_qty": "Quantity (grams)",
        "demo_cleared": "Demo account reset",
        "savings_title": "Gold Savings Plan",
        "savings_sub": "Save in gold monthly · Inflation protection",
        "savings_monthly": "Monthly Amount",
        "savings_since": "Start Date",
        "savings_curr": "Currency",
        "savings_add": "Add Plan",
        "savings_total": "Total Saved",
        "savings_gold": "Gold Accumulated",
        "savings_value": "Current Value",
        "savings_gain": "Gain vs Cash Savings",
        "savings_vs": "Comparison: Gold vs Cash Savings",
        "savings_note": "💡 Saving in gold protects you from currency devaluation",
        "savings_plan": "My Savings Plan",
        "savings_delete": "Delete Plan",
        "report_title": "Weekly Report",
        "report_sub": "Gold market summary · Auto-generated",
        "report_gen": "Generate Report",
        "report_dl": "Download PDF",
        "report_week": "Week Report",
        "report_perf": "Weekly Performance",
        "report_outlook": "Outlook for Next Week",
        "report_events": "Upcoming Economic Events",
        "report_signals": "Signal Summary",
        "widget_title": "Embed Widget",
        "widget_sub": "Add live gold price to your website in one line",
        "widget_copy": "Copy Code",
        "widget_preview": "Widget Preview",
        "widget_style": "Style",
        "widget_curr2": "Currency",
        "widget_copied": "✅ Copied!",
        "widget_instruct": "Copy this code and paste it into your website",
        "cb_title": "Central Bank Reserves",
        "cb_sub": "World gold reserves · WGC 2025 data",
        "cb_arab": "Arab Countries",
        "cb_world": "World's Largest Holders",
        "cb_rank": "Rank",
        "cb_country": "Country",
        "cb_tonnes": "Tonnes",
        "cb_pct": "% of Reserves",
        "cb_trend": "Trend",
        "cb_insight": "Insight",
        "persona_select": "Select User Type",
        "persona_trader": "📈 Trader",
        "persona_investor": "💰 Investor",
        "persona_shop": "🏪 Gold Shop Owner",
        "persona_factory": "🏭 Jewellery Factory",
        "persona_designer": "💍 Designer / Craftsperson",
        "persona_buyer": "🛍️ Retail Buyer",
        "persona_lbl": "Persona",
        "nav_shopboard": "🏪  Shop Price Board",
        "nav_invoice": "🧾  Invoice Calculator",
        "nav_production": "🏭  Production Cost",
        "nav_fairprice": "⚖️  Fair Price Checker",
        "nav_piecepricing": "💍  Piece Pricing Studio",
        "shop_buy": "Buy Price",
        "shop_sell": "Sell Price",
        "shop_spread": "Shop Margin",
        "shop_display": "Display Mode",
        "shop_currency": "Display Currency",
        "shop_name": "Shop Name",
        "shop_purity": "Show All Purities",
        "shop_fullscreen": "Full Screen",
        "shop_embed": "Embed in Website",
        "inv_weight": "Weight (grams)",
        "inv_purity": "Purity",
        "inv_making": "Making Charge (per gram)",
        "inv_vat": "VAT %",
        "inv_gold_val": "Gold Value",
        "inv_making_val": "Total Making",
        "inv_vat_val": "Tax",
        "inv_total": "Total",
        "inv_save": "Save Invoice",
        "inv_history": "Invoice History",
        "inv_clear": "Clear History",
        "prod_recipe": "Alloy Recipe",
        "prod_gold_pct": "Gold %",
        "prod_weight": "Finished Weight (grams)",
        "prod_wastage": "Wastage %",
        "prod_labour": "Labour Cost (per gram)",
        "prod_overhead": "Overhead",
        "prod_gold_cost": "Raw Gold Cost",
        "prod_total_cost": "Total Production Cost",
        "prod_min_price": "Break-even Price",
        "prod_suggest": "Suggested Price (30% margin)",
        "prod_batch": "Batch Cost",
        "prod_qty": "Number of Pieces",
        "fp_quoted": "Quoted Price",
        "fp_weight": "Weight (grams)",
        "fp_purity": "Purity",
        "fp_currency": "Currency",
        "fp_spot_val": "Spot Value",
        "fp_making_est": "Estimated Making",
        "fp_verdict_fair": "Fair Price ✅",
        "fp_verdict_high": "Overpriced ⚠️",
        "fp_verdict_low": "Suspiciously Low 🤔",
        "fp_saving": "Your Saving",
        "fp_overpay": "You're Overpaying",
        "piece_gold_g": "Gold Weight (grams)",
        "piece_purity": "Purity",
        "piece_gems": "Gemstones",
        "piece_gem_add": "Add Gemstone",
        "piece_labour_hrs": "Labour Hours",
        "piece_hourly": "Hourly Rate",
        "piece_overhead": "Overhead",
        "piece_margin": "Target Margin %",
        "piece_cost": "Total Cost",
        "piece_price": "Suggested Price",
        "piece_save": "Save Design",
        "piece_designs": "My Saved Designs",
        "geo_news": "Latest Geopolitical News",
        "geo_timeline": "Event Timeline",
        "geo_filter": "Filter by Impact",
        "geo_all": "All",
        "date_from": "From Date",
        "date_to": "To Date",
        "quick_range": "Quick Range",
        "nav_geo": "🌍  Geopolitical Map",
        "nav_oilgold": "🛢️  Oil & Gold",
        "geo_title": "Geopolitical Map",
        "geo_sub": "Active conflict & tension zones and their impact on gold",
        "geo_risk": "Geopolitical Risk Index",
        "geo_impact": "Gold Impact",
        "geo_events": "Active Events",
        "geo_bullish": "Gold Bullish 🟢",
        "geo_bearish": "Gold Bearish 🔴",
        "geo_neutral": "Neutral 🟡",
        "og_title": "Oil & Gold",
        "og_sub": "Oil–Gold correlation tracker · Live index",
        "og_corr": "Correlation Coefficient",
        "og_diverge": "Warning: Unusual divergence detected",
        "og_signal": "Correlation Signal",
        "sig_entry": "Entry Price",
        "sig_target": "Target",
        "sig_stop": "Stop Loss",
        "sig_rr": "Risk/Reward Ratio",
        "sig_trade": "Trade Details",
    },
    "Français 🇫🇷": {
        "dir": "ltr",
        "app_name": "MANSA",
        "tagline": "Intelligence Or",
        "nav_dashboard": "🏠  Tableau de bord",
        "nav_markets":   "🌍  Marchés",
        "nav_charts":    "📈  Graphiques",
        "nav_simulator": "🔭  Simulateur",
        "nav_predictions":"🤖  Prédictions IA",
        "nav_data":      "📂  Explorateur de données",
        "nav_advisor":   "💬  Conseiller Or IA",
        "nav_settings":  "⚙️  Paramètres",

        "nav_signals":  "📡  Signaux",
        "nav_newssent": "🧠  Sentiment Actualités",
        "nav_supply":  "⛏️  Offre & Demande",
        "nav_currency":"💱  Convertisseur",
        "nav_alerts":     "🔔  Alertes Prix",
        "nav_heatmap":    "📊  Corrélations",
        "nav_mansa_score":"🏆  Score MANSA",
        "nav_zakat":      "☪️  Calculateur Zakat",
        "nav_compare":    "📈  Comparaison",
        "nav_journal":    "📓  Journal de Trade",
        "nav_goldmap":    "🌍  Carte Or",
        "nav_drawdown":   "📉  Test de Stress",
        "nav_portfolio":  "💼  Portefeuille",
        "nav_calculator": "🧮  Calculatrice",
        "nav_calendar":   "📅  Calendrier économique",
        "nav_sentiment":  "🌡️  Sentiment",
        "nav_sessions":   "🕐  Sessions de marché",
        "nav_game":   "🎮  Jeu Mansa",
        "nav_about":  "🏛️  À propos de Mansa",        "live": "En direct",
        "not_financial": "Pas un conseil financier",
        "quick_settings": "Paramètres rapides",
        "weight_unit": "Unité de poids",
        "purity": "Pureté de l'or",
        "theme": "Thème",
        "auto_refresh": "Actualisation auto (60s)",
        "primary_market": "Marché principal",
        "gold_intelligence": "Plateforme Intelligence Or",
        "real_time": "Temps réel · Marchés arabes & mondiaux",
        "spot_price": "Prix spot mondial",
        "key_stats": "Statistiques clés de l'or",
        "wk52_high": "Plus haut 52 sem.",
        "wk52_low": "Plus bas 52 sem.",
        "wk52_avg": "Moyenne 52 sem.",
        "ytd_return": "Rendement YTD",
        "gold_silver": "Ratio Or/Argent",
        "gold_oil": "Ratio Or/Pétrole",
        "market_overview": "Vue d'ensemble du marché",
        "stocks_indices": "Actions & Indices",
        "all_purities": "Toutes les puretés",
        "prediction_date": "Date de prédiction",
        "tomorrow": "Demain",
        "best_model": "Meilleur modèle",
        "algo_predictions": "Prédictions algorithmiques",
        "consensus": "Consensus",
        "model_high": "Prédiction haute",
        "model_low": "Prédiction basse",
        "spread": "Écart",
        "vs_spot": "vs spot",
        "pred_multi_unit": "Prédiction en plusieurs unités et devises",
        "no_models": "Aucun modèle trouvé. Lancez train_models.py.",
        "r2_scores": "Précision des modèles · R²",
        "prophet_forecast": "Prévision Prophet 30 jours",
        "select_model": "Choisir un modèle",
        "disclaimer": "Avertissement : à titre éducatif uniquement. Pas un conseil financier.",
        "arab_markets": "Marchés arabes",
        "intl_markets": "Marchés internationaux",
        "purity_matrix": "Matrice pureté × marché",
        "data_explorer": "Explorateur de données",
        "training_data": "Données d'entraînement utilisées pour les modèles",
        "total_rows": "Nombre de lignes",
        "features": "Variables",
        "from_date": "De",
        "to_date": "À",
        "filter_year": "Filtrer par année",
        "columns_display": "Colonnes à afficher",
        "rows_show": "Lignes à afficher",
        "feat_desc": "Description des variables",
        "gold_history": "Historique du prix de l'or",
        "trading_advisor": "Conseiller de trading",
        "live_signals": "Signaux techniques en direct",
        "price_levels": "Niveaux de prix clés",
        "trade_setup": "Configuration de la transaction",
        "personalised_advice": "Conseils personnalisés",
        "golden_rules": "Les 10 règles d'or du trading",
        "macro_env": "Environnement macro actuel",
        "overall_signal": "Signal global du marché",
        "bullish": "Haussier",
        "bearish": "Baissier",
        "neutral": "Neutre",
        "support": "Support",
        "resistance": "Résistance",
        "entry": "Entrée",
        "stop_loss": "Stop loss",
        "take_profit": "Take profit",
        "settings": "Paramètres",
        "design_theme": "Thème de conception",
        "display_prefs": "Préférences d'affichage",
        "show_purity_table": "Afficher tableau de pureté",
        "default_period": "Période graphique par défaut",
        "active_markets": "Marchés actifs",
        "active_stocks": "Actions & indices actifs",
        "conversion_ref": "Référence de conversion unité × pureté",
        "not_found_data": "Données d'entraînement introuvables.",
        "profile": "Profil du trader",
        "profile_opts": ["Débutant 🌱", "Intermédiaire 📊", "Avancé ⚡", "Investisseur long terme 🏛️"],
        "chatbot_title": "💬  Conseiller Or IA",
        "chatbot_sub": "Posez-moi n'importe quelle question sur le trading de l'or · Propulsé par IA",
        "chatbot_placeholder": "Tapez votre question... ex: Devrais-je acheter de l'or maintenant?",
        "chatbot_send": "Envoyer",
        "chatbot_clear": "Effacer",
        "chatbot_thinking": "Réflexion...",
        "chatbot_welcome": "Bonjour! Je suis votre conseiller or IA. Demandez-moi n'importe quoi sur l'or!",
        "chatbot_error": "Erreur. Veuillez réessayer.",
        "models_demo_note": "⚠️ Modèles IA introuvables. Prédictions de démonstration affichées.",
        "no_csv": "Fichier de données introuvable.",
        "indicators": "Indicateurs",
        "period": "Période",
        "chart_type": "Type de graphique",
        "live_prices": "Prix en direct",
        "yf_delay": "Yahoo Finance · ~15 min de délai",
        "live_snapshot": "📊 Snapshot marché en direct",
        "col_asset": "ACTIF",
        "col_price": "PRIX",
        "col_change": "VARIATION",
        "col_status": "STATUT",
        "data_up_to": "Données jusqu'à :",
        "added": "Ajouté",
        "new_rows_to": "nouvelles lignes · Données à jour jusqu'au",
        "update_error": "Erreur de mise à jour :",
        "latest": "Derniers",
        "rows_added": "lignes ajoutées",
        "rows_filtered": "lignes filtrées",
        "total_lc": "total",
        "sparklines": "Graphiques rapides · 12 derniers mois",
        "quick_questions": "Questions rapides",
        "brand_logos": "Logos · Un par thème",
        "developer": "Développeur & Créateur",
        "why_mansa": "Pourquoi Mansa ? · L'histoire de Mansa Musa",
        "mansa_title": "Mansa Musa — L'homme le plus riche de l'histoire",
        "hist_highlights": "Faits historiques",
        "mansa_caption": "Mansa Musa · Empereur du Mali",
        "mansa_dates": "1280 – 1337 CE",
        "mosque_caption": "Mosquée Djinguereber · Tombouctou",
        "mosque_built": "Construite par Mansa Musa · 1327 CE",
        "empire_lbl": "Empire",
        "trade_routes": "Routes commerciales",
        "empire_peak": "Empire du Mali à son apogée",
        "empire_century": "XIVe siècle · 2 millions de km²",
        "about_platform": "À propos de la plateforme · Par Own Al Ansari",
        "copyright": "© 2025 Own Al Ansari",
        "platform_name": "MANSA — Plateforme Intelligence Or",
        "position_details": "Détails des positions",
        "results": "Résultats",
        "risk_amount": "Montant risqué",
        "position_size_lbl": "Taille de position",
        "position_cost": "Coût de position",
        "risk_reward": "Risque/Récompense",
        "calc_pnl_sub": "Calculer le profit ou la perte attendue",
        "cost_lbl": "Coût",
        "value_lbl": "Valeur",
        "calc_margin_sub": "Calculer la marge requise pour le trading à effet de levier",
        "notional": "Valeur nominale",
        "req_margin": "Marge requise",
        "per_dollar": "Valeur par $1",
        "leverage_lbl": "Levier",
        "calc_be_sub": "Calculer le seuil de rentabilité avec frais",
        "breakeven": "Prix seuil",
        "above_entry": "au-dessus du prix d'entrée",
        "spread_lbl": "Écart",
        "commission_lbl": "Commission",
        "swap_lbl": "Coût swap",
        "today_lbl": "AUJOURD'HUI ⚡",
        "forecast_lbl": "Prévision :",
        "actual_lbl": "Réel :",
        "fear_lbl": "PEUR",
        "greed_lbl": "AVIDITÉ",
        "what_means_gold": "Qu'est-ce que cela signifie pour l'or ?",
        "index_components": "Composantes de l'indice",
        "market_status_now": "État du marché maintenant",
        "arab_mkts_primary": "⭐ Marchés arabes · Jordanie principal",
        "intl_mkts_lbl": "🌐 Marchés internationaux",
        "timeline_24h": "Calendrier 24h (UTC)",
        "gold_corr_assets": "🥇 Corrélation de l'or avec chaque actif",
        "five_pillars": "Les cinq piliers",
        "enter_holdings": "Entrez vos avoirs en or",
        "gold24k": "Or 24K (grammes)",
        "gold22k": "Or 22K (grammes)",
        "gold21k": "Or 21K (grammes)",
        "gold18k": "Or 18K (grammes)",
        "gold14k": "Or 14K (grammes)",
        "zakat_result": "Résultat du calcul de la Zakat",
        "total_gold_lbl": "Or total",
        "nisab_lbl": "Nissab",
        "total_value_lbl": "Valeur totale",
        "zakat_due": "Zakat due (2,5%)",
        "returns_lb": "🏆 Classement des rendements",
        "trade_log": "Journal des trades",
        "nisab_based": "Basé sur le prix spot argent : Or 24K ",
        "current_price_lbl": "Prix actuel",
        "interactive_sim": "Simulation interactive",
        "cb_reserves": "🏦 Réserves or des banques centrales (tonnes) · 2025",
        "mine_prod": "⛏️ Production minière or (tonnes/an 2024)",
        "global_demand": "📊 Demande mondiale en or (tonnes 2024)",
        "major_etfs": "📊 Principaux ETF or",
        "gram_arab": "Prix du gramme d'or en devises arabes",
        "auto_refresh_60": "Tarifs actualisés toutes les 60s",
        "overall_signal_ts": "SIGNAL GLOBAL MANSA · ",
        "confidence_lbl": "Confiance",
        "buy_lbl": "ACHETER",
        "sell_lbl": "VENDRE",
        "all_signals": "📋 Détail de tous les signaux",
        "action_buy": "Action suggérée : ACHETER",
        "entry_lbl": "Entrée",
        "action_sell": "Action suggérée : VENDRE ou ATTENDRE",
        "exit_sell": "Sortie/Vente à",
        "action_wait": "Action suggérée : ATTENDRE signal plus clair",
        "bullish_lbl": "Haussier",
        "neutral_lbl": "Neutre",
        "bearish_lbl": "Baissier",
        "very_bearish": "Très baissier",
        "average_lbl": "Moyenne",
        "very_bullish": "Très haussier",
        "article_analysis": "📰 Analyse article par article",
        "session_score": "Score de session",
        "high_score_lbl": "Meilleur score",
        "start_game": "Démarrer le jeu",
        "amazing_10": "Incroyable ! 10 pièces d'or !",
        "game_over": "Partie terminée ☽",
        "gold_collected": "Or collecté",
        "play_again": "Rejouer",
        "restart_lbl": "Recommencer",
        "calc_pos_sub": "Calculez la taille de position adaptée à votre tolérance au risque",
        "tomorrow_lbl": "Demain",
        "days_lbl": "dans ",
        "neutral_sig": "NEUTRE",
        "stop_loss_lbl": "Stop loss",
        "news_sources": "Actualisé toutes les 15 min · Sources : Reuters · Kitco · FXStreet · MarketWatch",
        "game_instructions": "⬆ ESPACE ou clic pour sauter · Collectez les pièces d'or · Battez votre record !",
        "update_hs": "🏆 Nouveau record",
        "feat_info": "Informations · Histoire · Relations · Signal",
        "history_lbl": "📖 Histoire",
        "relation_gold": "🔗 Relation avec l'or",
        "closes_in": "Ferme dans",
        "opens_in": "Ouvre dans",
        "neutral_lc": "NEUTRE",
        "add_new_alert": "Ajouter une alerte",
        "target_price": "Prix cible (USD/oz)",
        "direction_lbl": "Direction",
        "alert_label": "Libellé de l'alerte",
        "why_level": "Pourquoi ce niveau est-il important ?",
        "save_alert": "Enregistrer l'alerte",
        "alert_saved": "🔔 Alerte enregistrée !",
        "alert_triggered_lbl": "ALERTE DÉCLENCHÉE !",
        "current_lbl": "Actuel",
        "no_alerts_msg": "Aucune alerte active — ajoutez une alerte ci-dessus",
        "clear_triggered": "Effacer déclenchées",
        "alert_at": "Alerte à",
        "distance_lbl": "Distance",
        "nav_demo": "🎯  Trading démo",
        "nav_savings": "🪙  Plan d'épargne or",
        "nav_report": "📋  Rapport hebdomadaire",
        "nav_widget": "🔌  Widget intégrable",
        "nav_cb": "🏦  Réserves des banques centrales",
        "demo_title": "Trading Démo",
        "demo_sub": "Tradez avec de l'argent virtuel · Zéro risque réel",
        "demo_balance": "Solde démo",
        "demo_buy": "Acheter de l'or",
        "demo_sell": "Vendre de l'or",
        "demo_reset": "Réinitialiser le compte",
        "demo_pnl": "Profit / Perte",
        "demo_holdings": "Portefeuille démo",
        "demo_trades": "Journal démo",
        "demo_note": "💡 Compte démo — aucun argent réel n'est utilisé",
        "demo_qty": "Quantité (grammes)",
        "demo_cleared": "Compte démo réinitialisé",
        "savings_title": "Plan d'épargne or",
        "savings_sub": "Épargnez en or chaque mois · Protection contre l'inflation",
        "savings_monthly": "Montant mensuel",
        "savings_since": "Date de début",
        "savings_curr": "Devise",
        "savings_add": "Ajouter un plan",
        "savings_total": "Total épargné",
        "savings_gold": "Or accumulé",
        "savings_value": "Valeur actuelle",
        "savings_gain": "Gain vs épargne en cash",
        "savings_vs": "Comparaison : or vs épargne cash",
        "savings_note": "💡 L'épargne en or vous protège de la dévaluation",
        "savings_plan": "Mon plan d'épargne",
        "savings_delete": "Supprimer le plan",
        "report_title": "Rapport hebdomadaire",
        "report_sub": "Résumé du marché de l'or · Généré automatiquement",
        "report_gen": "Générer le rapport",
        "report_dl": "Télécharger PDF",
        "report_week": "Rapport semaine",
        "report_perf": "Performance hebdomadaire",
        "report_outlook": "Perspectives pour la semaine prochaine",
        "report_events": "Événements économiques à venir",
        "report_signals": "Résumé des signaux",
        "widget_title": "Widget intégrable",
        "widget_sub": "Ajoutez le prix de l'or à votre site en une ligne",
        "widget_copy": "Copier le code",
        "widget_preview": "Aperçu du widget",
        "widget_style": "Style",
        "widget_curr2": "Devise",
        "widget_copied": "✅ Copié !",
        "widget_instruct": "Copiez ce code et collez-le dans votre site web",
        "cb_title": "Réserves des banques centrales",
        "cb_sub": "Réserves d'or mondiales · Données WGC 2025",
        "cb_arab": "Pays arabes",
        "cb_world": "Plus grands détenteurs mondiaux",
        "cb_rank": "Rang",
        "cb_country": "Pays",
        "cb_tonnes": "Tonnes",
        "cb_pct": "% des réserves",
        "cb_trend": "Tendance",
        "cb_insight": "Analyse",
        "persona_select": "Sélectionner le type d'utilisateur",
        "persona_trader": "📈 Trader",
        "persona_investor": "💰 Investisseur",
        "persona_shop": "🏪 Propriétaire de bijouterie",
        "persona_factory": "🏭 Manufacture de bijoux",
        "persona_designer": "💍 Designer / Artisan",
        "persona_buyer": "🛍️ Acheteur",
        "persona_lbl": "Profil",
        "nav_shopboard": "🏪  Tableau des prix",
        "nav_invoice": "🧾  Calculateur de facture",
        "nav_production": "🏭  Coût de production",
        "nav_fairprice": "⚖️  Vérificateur de prix",
        "nav_piecepricing": "💍  Studio de tarification",
        "shop_buy": "Prix d'achat",
        "shop_sell": "Prix de vente",
        "shop_spread": "Marge boutique",
        "shop_display": "Mode d'affichage",
        "shop_currency": "Devise d'affichage",
        "shop_name": "Nom de la boutique",
        "shop_purity": "Afficher toutes les puretés",
        "shop_fullscreen": "Plein écran",
        "shop_embed": "Intégrer au site",
        "inv_weight": "Poids (grammes)",
        "inv_purity": "Pureté",
        "inv_making": "Frais de fabrication (par g)",
        "inv_vat": "TVA %",
        "inv_gold_val": "Valeur or",
        "inv_making_val": "Fabrication totale",
        "inv_vat_val": "Taxe",
        "inv_total": "Total",
        "inv_save": "Sauvegarder la facture",
        "inv_history": "Historique des factures",
        "inv_clear": "Effacer l'historique",
        "prod_recipe": "Recette d'alliage",
        "prod_gold_pct": "Or %",
        "prod_weight": "Poids fini (grammes)",
        "prod_wastage": "Déchets %",
        "prod_labour": "Main-d'œuvre (par g)",
        "prod_overhead": "Frais généraux",
        "prod_gold_cost": "Coût or brut",
        "prod_total_cost": "Coût de production total",
        "prod_min_price": "Prix de revient",
        "prod_suggest": "Prix suggéré (marge 30%)",
        "prod_batch": "Coût du lot",
        "prod_qty": "Nombre de pièces",
        "fp_quoted": "Prix proposé",
        "fp_weight": "Poids (grammes)",
        "fp_purity": "Pureté",
        "fp_currency": "Devise",
        "fp_spot_val": "Valeur spot",
        "fp_making_est": "Fabrication estimée",
        "fp_verdict_fair": "Prix équitable ✅",
        "fp_verdict_high": "Prix excessif ⚠️",
        "fp_verdict_low": "Prix suspicieusement bas 🤔",
        "fp_saving": "Votre économie",
        "fp_overpay": "Vous payez trop",
        "piece_gold_g": "Poids or (grammes)",
        "piece_purity": "Pureté",
        "piece_gems": "Pierres précieuses",
        "piece_gem_add": "Ajouter une pierre",
        "piece_labour_hrs": "Heures de travail",
        "piece_hourly": "Taux horaire",
        "piece_overhead": "Frais généraux",
        "piece_margin": "Marge cible %",
        "piece_cost": "Coût total",
        "piece_price": "Prix suggéré",
        "piece_save": "Sauvegarder le design",
        "piece_designs": "Mes designs sauvegardés",
        "geo_news": "Dernières actualités géopolitiques",
        "geo_timeline": "Chronologie des événements",
        "geo_filter": "Filtrer par impact",
        "geo_all": "Tout",
        "date_from": "Date de début",
        "date_to": "Date de fin",
        "quick_range": "Période rapide",
        "nav_geo": "🌍  Carte géopolitique",
        "nav_oilgold": "🛢️  Pétrole & Or",
        "geo_title": "Carte géopolitique",
        "geo_sub": "Zones de tension actives et leur impact sur l'or",
        "geo_risk": "Indice de risque géopolitique",
        "geo_impact": "Impact sur l'or",
        "geo_events": "Événements actifs",
        "geo_bullish": "Haussier pour l'or 🟢",
        "geo_bearish": "Baissier pour l'or 🔴",
        "geo_neutral": "Neutre 🟡",
        "og_title": "Pétrole & Or",
        "og_sub": "Corrélation pétrole-or · Indice en direct",
        "og_corr": "Coefficient de corrélation",
        "og_diverge": "Avertissement: divergence inhabituelle",
        "og_signal": "Signal de corrélation",
        "sig_entry": "Prix d'entrée",
        "sig_target": "Objectif",
        "sig_stop": "Stop-loss",
        "sig_rr": "Ratio risque/rendement",
        "sig_trade": "Détails du trade",
    },
    "Türkçe 🇹🇷": {
        "dir": "ltr",
        "app_name": "MANSA",
        "tagline": "Altın Zekası",
        "nav_dashboard": "🏠  Panel",
        "nav_markets":   "🌍  Piyasalar",
        "nav_charts":    "📈  Grafikler",
        "nav_simulator": "🔭  Simülatör",
        "nav_predictions":"🤖  YZ Tahminleri",
        "nav_data":      "📂  Veri Gezgini",
        "nav_advisor":   "💬  YZ Altın Danışmanı",
        "nav_settings":  "⚙️  Ayarlar",

        "nav_signals":  "📡  Sinyal Tablosu",
        "nav_newssent": "🧠  Haber Analizi",
        "nav_supply":  "⛏️  Arz & Talep",
        "nav_currency":"💱  Döviz Çevirici",
        "nav_alerts":     "🔔  Fiyat Uyarıları",
        "nav_heatmap":    "📊  Korelasyon",
        "nav_mansa_score":"🏆  MANSA Skoru",
        "nav_zakat":      "☪️  Zekat Hesaplayıcı",
        "nav_compare":    "📈  Varlık Karşılaştırma",
        "nav_journal":    "📓  İşlem Günlüğü",
        "nav_goldmap":    "🌍  Altın Haritası",
        "nav_drawdown":   "📉  Stres Testi",
        "nav_portfolio":  "💼  Portföy",
        "nav_calculator": "🧮  Hesaplayıcı",
        "nav_calendar":   "📅  Ekonomik Takvim",
        "nav_sentiment":  "🌡️  Duygu Endeksi",
        "nav_sessions":   "🕐  Piyasa Seansları",
        "nav_game":   "🎮  Mansa Oyunu",
        "nav_about":  "🏛️  Mansa Hakkında",        "live": "Canlı",
        "not_financial": "Finansal tavsiye değil",
        "quick_settings": "Hızlı Ayarlar",
        "weight_unit": "Ağırlık Birimi",
        "purity": "Altın Saflığı",
        "theme": "Tema",
        "auto_refresh": "Otomatik yenile (60s)",
        "primary_market": "Ana Piyasa",
        "gold_intelligence": "Altın Zekâ Platformu",
        "real_time": "Canlı · Arap & küresel piyasalar",
        "spot_price": "Global Spot Fiyatı",
        "key_stats": "Temel Altın İstatistikleri",
        "wk52_high": "52H En Yüksek",
        "wk52_low": "52H En Düşük",
        "wk52_avg": "52H Ortalama",
        "ytd_return": "YTD Getiri",
        "gold_silver": "Altın/Gümüş Oranı",
        "gold_oil": "Altın/Petrol Oranı",
        "market_overview": "Piyasa Genel Bakış",
        "stocks_indices": "Hisse & Endeksler",
        "all_purities": "Tüm Saflıklar",
        "prediction_date": "Tahmin Tarihi",
        "tomorrow": "Yarın",
        "best_model": "En İyi Model",
        "algo_predictions": "Algoritma Tahminleri",
        "consensus": "Konsensüs",
        "model_high": "Model Yüksek",
        "model_low": "Model Düşük",
        "spread": "Fark",
        "vs_spot": "spot'a karşı",
        "pred_multi_unit": "Çoklu Birim & Para Birimi Tahmini",
        "no_models": "Model bulunamadı. Önce train_models.py çalıştırın.",
        "r2_scores": "Model Doğruluğu · R²",
        "prophet_forecast": "Prophet 30 Günlük Tahmin",
        "select_model": "Model Seç",
        "disclaimer": "Uyarı: Yalnızca eğitim amaçlıdır. Finansal tavsiye değildir.",
        "arab_markets": "Arap Piyasaları",
        "intl_markets": "Uluslararası Piyasalar",
        "purity_matrix": "Saflık × Piyasa Matrisi",
        "data_explorer": "Veri Gezgini",
        "training_data": "Model eğitiminde kullanılan veri seti",
        "total_rows": "Toplam Satır",
        "features": "Değişkenler",
        "from_date": "Başlangıç",
        "to_date": "Bitiş",
        "filter_year": "Yıla Göre Filtrele",
        "columns_display": "Gösterilecek Sütunlar",
        "rows_show": "Gösterilecek Satır Sayısı",
        "feat_desc": "Değişken Açıklamaları",
        "gold_history": "Altın Fiyat Geçmişi",
        "trading_advisor": "İşlem Danışmanı",
        "live_signals": "Canlı Teknik Sinyaller",
        "price_levels": "Temel Fiyat Seviyeleri",
        "trade_setup": "İşlem Kurulumu",
        "personalised_advice": "Kişiselleştirilmiş Tavsiyeler",
        "golden_rules": "Altın Ticaretinin 10 Altın Kuralı",
        "macro_env": "Mevcut Makro Ortam",
        "overall_signal": "Genel Piyasa Sinyali",
        "bullish": "Yükseliş",
        "bearish": "Düşüş",
        "neutral": "Nötr",
        "support": "Destek",
        "resistance": "Direnç",
        "entry": "Giriş",
        "stop_loss": "Stop Loss",
        "take_profit": "Kâr Al",
        "settings": "Ayarlar",
        "design_theme": "Tasarım Teması",
        "display_prefs": "Görüntüleme Tercihleri",
        "show_purity_table": "Panelde saflık tablosunu göster",
        "default_period": "Varsayılan Grafik Dönemi",
        "active_markets": "Aktif Piyasalar",
        "active_stocks": "Aktif Hisse & Endeksler",
        "conversion_ref": "Birim × Saflık Dönüşüm Referansı",
        "not_found_data": "Eğitim verisi bulunamadı.",
        "profile": "Trader Profili",
        "profile_opts": ["Başlangıç 🌱", "Orta Düzey 📊", "İleri Düzey ⚡", "Uzun Vadeli Yatırımcı 🏛️"],
        "chatbot_title": "💬  YZ Altın Danışmanı",
        "chatbot_sub": "Altın ticareti hakkında her şeyi sorun · YZ destekli",
        "chatbot_placeholder": "Sorunuzu yazın... örn: Şimdi altın almalı mıyım?",
        "chatbot_send": "Gönder",
        "chatbot_clear": "Sohbeti Temizle",
        "chatbot_thinking": "Düşünüyor...",
        "chatbot_welcome": "Merhaba! Ben YZ Altın Danışmanınızım. Altın hakkında her şeyi sorabilirsiniz!",
        "chatbot_error": "Hata oluştu. Lütfen tekrar deneyin.",
        "models_demo_note": "⚠️ YZ modelleri bulunamadı. Demo tahminler gösteriliyor.",
        "no_csv": "Veri dosyası bulunamadı.",
        "indicators": "Göstergeler",
        "period": "Dönem",
        "chart_type": "Grafik Türü",
        "live_prices": "Canlı fiyatlar",
        "yf_delay": "Yahoo Finance · ~15 dk gecikme",
        "live_snapshot": "📊 Canlı Piyasa Özeti",
        "col_asset": "VARLIK",
        "col_price": "FİYAT",
        "col_change": "DEĞİŞİM",
        "col_status": "DURUM",
        "data_up_to": "Veri tarihi:",
        "added": "Eklendi",
        "new_rows_to": "yeni satır · Veri güncel:",
        "update_error": "Güncelleme hatası:",
        "latest": "Son",
        "rows_added": "satır eklendi",
        "rows_filtered": "filtreli satır",
        "total_lc": "toplam",
        "sparklines": "Mini grafikler · Son 12 ay",
        "quick_questions": "Hızlı Sorular",
        "brand_logos": "Logolar · Her tema için",
        "developer": "Geliştirici & Yaratıcı",
        "why_mansa": "Neden Mansa? · Mansa Musa'nın Hikayesi",
        "mansa_title": "Mansa Musa — Tarihin En Zengin İnsanı",
        "hist_highlights": "Tarihî Öne Çıkanlar",
        "mansa_caption": "Mansa Musa · Mali İmparatoru",
        "mansa_dates": "1280 – 1337",
        "mosque_caption": "Djinguereber Camii · Timbuktu",
        "mosque_built": "Mansa Musa tarafından yapıldı · 1327",
        "empire_lbl": "İmparatorluk",
        "trade_routes": "Ticaret Yolları",
        "empire_peak": "Mali İmparatorluğu Zirvede",
        "empire_century": "14. Yüzyıl · 2 Milyon km²",
        "about_platform": "Platform Hakkında · Own Al Ansari",
        "copyright": "© 2025 Own Al Ansari",
        "platform_name": "MANSA Altın Zekâ Platformu",
        "position_details": "Pozisyon Detayları",
        "results": "Sonuçlar",
        "risk_amount": "Risk Tutarı",
        "position_size_lbl": "Pozisyon Büyüklüğü",
        "position_cost": "Pozisyon Maliyeti",
        "risk_reward": "Risk/Ödül",
        "calc_pnl_sub": "Beklenen kâr veya zararı hesaplayın",
        "cost_lbl": "Maliyet",
        "value_lbl": "Değer",
        "calc_margin_sub": "Kaldıraçlı işlem için gerekli marjı hesaplayın",
        "notional": "Nominal Değer",
        "req_margin": "Gerekli Marj",
        "per_dollar": "$1 başına değer",
        "leverage_lbl": "Kaldıraç",
        "calc_be_sub": "İşlem maliyetleri dahil başabaş fiyatı hesaplayın",
        "breakeven": "Başabaş Fiyatı",
        "above_entry": "giriş üzerinde",
        "spread_lbl": "Fark",
        "commission_lbl": "Komisyon",
        "swap_lbl": "Swap maliyeti",
        "today_lbl": "BUGÜN ⚡",
        "forecast_lbl": "Tahmin:",
        "actual_lbl": "Gerçek:",
        "fear_lbl": "KORKU",
        "greed_lbl": "AÇGÖZLÜLÜK",
        "what_means_gold": "Bu altın için ne anlama gelir?",
        "index_components": "Endeks Bileşenleri",
        "market_status_now": "Piyasa Durumu Şu An",
        "arab_mkts_primary": "⭐ Arap Piyasaları · Ürdün Ana Piyasa",
        "intl_mkts_lbl": "🌐 Uluslararası Piyasalar",
        "timeline_24h": "24 Saatlik Zaman Çizelgesi (UTC)",
        "gold_corr_assets": "🥇 Altının Her Varlıkla Korelasyonu",
        "five_pillars": "Beş Sütun Analizi",
        "enter_holdings": "Altın Varlıklarınızı Girin",
        "gold24k": "24K Altın (gram)",
        "gold22k": "22K Altın (gram)",
        "gold21k": "21K Altın (gram)",
        "gold18k": "18K Altın (gram)",
        "gold14k": "14K Altın (gram)",
        "zakat_result": "Zekat Hesaplama Sonucu",
        "total_gold_lbl": "Toplam Altın",
        "nisab_lbl": "Nisap",
        "total_value_lbl": "Toplam Değer",
        "zakat_due": "Ödenecek Zekat (2,5%)",
        "returns_lb": "🏆 Getiri Sıralaması",
        "trade_log": "İşlem Günlüğü",
        "nisab_based": "Canlı gümüş fiyatına göre: 24K Altın ",
        "current_price_lbl": "Mevcut fiyat",
        "interactive_sim": "Etkileşimli Simülasyon",
        "cb_reserves": "🏦 Merkez Bankası Altın Rezervleri (ton) · 2025",
        "mine_prod": "⛏️ Altın Maden Üretimi (ton/yıl 2024)",
        "global_demand": "📊 Küresel Altın Talebi (ton 2024)",
        "major_etfs": "📊 Büyük Altın ETF'leri",
        "gram_arab": "Altın Gram Fiyatı Arap Para Birimlerinde",
        "auto_refresh_60": "Fiyatlar her 60 saniyede güncellenir",
        "overall_signal_ts": "MANSA GENEL SİNYAL · ",
        "confidence_lbl": "Güven",
        "buy_lbl": "AL",
        "sell_lbl": "SAT",
        "all_signals": "📋 Tüm Sinyal Detayları",
        "action_buy": "Önerilen Eylem: AL",
        "entry_lbl": "Giriş",
        "action_sell": "Önerilen Eylem: SAT veya BEKLE",
        "exit_sell": "Çıkış/Sat:",
        "action_wait": "Önerilen Eylem: Net sinyal için BEKLE",
        "bullish_lbl": "Yükseliş",
        "neutral_lbl": "Nötr",
        "bearish_lbl": "Düşüş",
        "very_bearish": "Çok Düşüş",
        "average_lbl": "Ortalama",
        "very_bullish": "Çok Yükseliş",
        "article_analysis": "📰 Makale Bazlı Analiz",
        "session_score": "Oturum Puanı",
        "high_score_lbl": "En Yüksek Puan",
        "start_game": "Oyunu Başlat",
        "amazing_10": "Harika! 10 altın sikke!",
        "game_over": "Oyun Bitti ☽",
        "gold_collected": "Toplanan altın",
        "play_again": "Tekrar Oyna",
        "restart_lbl": "Yeniden Başlat",
        "calc_pos_sub": "Risk toleransınıza göre doğru pozisyon büyüklüğünü hesaplayın",
        "tomorrow_lbl": "Yarın",
        "days_lbl": "içinde ",
        "neutral_sig": "NÖTR",
        "stop_loss_lbl": "Stop Loss",
        "news_sources": "Her 15 dakikada güncellenir · Kaynaklar: Reuters · Kitco · FXStreet · MarketWatch",
        "game_instructions": "⬆ Zıplamak için BOŞLUK ya da tıklayın · Altın toplayın · Rekor kırın!",
        "update_hs": "🏆 Skoru Güncelle",
        "feat_info": "Değişken Bilgisi · Tarih · İlişkiler · Sinyal",
        "history_lbl": "📖 Tarih",
        "relation_gold": "🔗 Altınla İlişki",
        "closes_in": "Kapanıyor",
        "opens_in": "Açılıyor",
        "neutral_lc": "NÖTR",
        "add_new_alert": "Yeni Uyarı Ekle",
        "target_price": "Hedef Fiyat (USD/oz)",
        "direction_lbl": "Yön",
        "alert_label": "Uyarı Etiketi",
        "why_level": "Bu seviye neden önemli?",
        "save_alert": "Uyarıyı Kaydet",
        "alert_saved": "🔔 Uyarı kaydedildi!",
        "alert_triggered_lbl": "UYARI TETİKLENDİ!",
        "current_lbl": "Mevcut",
        "no_alerts_msg": "Aktif uyarı yok — yukarıdan hedef fiyat ekleyin",
        "clear_triggered": "Tetiklenenleri temizle",
        "alert_at": "Uyarı:",
        "distance_lbl": "Mesafe",
        "nav_demo": "🎯  Demo Ticaret",
        "nav_savings": "🪙  Altın Birikim Planı",
        "nav_report": "📋  Haftalık Rapor",
        "nav_widget": "🔌  Gömme Widget",
        "nav_cb": "🏦  Merkez Bankası Rezervleri",
        "demo_title": "Demo Ticaret",
        "demo_sub": "Sanal para ile işlem yapın · Gerçek risk yok",
        "demo_balance": "Demo Bakiye",
        "demo_buy": "Altın Al",
        "demo_sell": "Altın Sat",
        "demo_reset": "Hesabı Sıfırla",
        "demo_pnl": "Kâr / Zarar",
        "demo_holdings": "Demo Portföy",
        "demo_trades": "Demo İşlem Günlüğü",
        "demo_note": "💡 Demo hesap — gerçek para kullanılmaz",
        "demo_qty": "Miktar (gram)",
        "demo_cleared": "Demo hesap sıfırlandı",
        "savings_title": "Altın Birikim Planı",
        "savings_sub": "Her ay altın biriktirin · Enflasyon koruması",
        "savings_monthly": "Aylık Tutar",
        "savings_since": "Başlangıç Tarihi",
        "savings_curr": "Para Birimi",
        "savings_add": "Plan Ekle",
        "savings_total": "Toplam Birikim",
        "savings_gold": "Biriken Altın",
        "savings_value": "Güncel Değer",
        "savings_gain": "Nakit birikime kıyasla kazanç",
        "savings_vs": "Karşılaştırma: Altın vs Nakit Birikim",
        "savings_note": "💡 Altında birikim, kur değer kaybına karşı koruma sağlar",
        "savings_plan": "Birikim Planım",
        "savings_delete": "Planı Sil",
        "report_title": "Haftalık Rapor",
        "report_sub": "Altın piyasası özeti · Otomatik oluşturulur",
        "report_gen": "Rapor Oluştur",
        "report_dl": "PDF İndir",
        "report_week": "Hafta Raporu",
        "report_perf": "Haftalık Performans",
        "report_outlook": "Gelecek Hafta Görünümü",
        "report_events": "Yaklaşan Ekonomik Olaylar",
        "report_signals": "Sinyal Özeti",
        "widget_title": "Gömme Widget",
        "widget_sub": "Web sitenize tek satırla canlı altın fiyatı ekleyin",
        "widget_copy": "Kodu Kopyala",
        "widget_preview": "Widget Önizleme",
        "widget_style": "Stil",
        "widget_curr2": "Para Birimi",
        "widget_copied": "✅ Kopyalandı!",
        "widget_instruct": "Bu kodu kopyalayıp web sitenize yapıştırın",
        "cb_title": "Merkez Bankası Rezervleri",
        "cb_sub": "Dünya altın rezervleri · WGC 2025 verileri",
        "cb_arab": "Arap Ülkeleri",
        "cb_world": "Dünyanın En Büyük Sahipleri",
        "cb_rank": "Sıra",
        "cb_country": "Ülke",
        "cb_tonnes": "Ton",
        "cb_pct": "Rezervlerin %'si",
        "cb_trend": "Trend",
        "cb_insight": "Analiz",
        "persona_select": "Kullanıcı tipini seçin",
        "persona_trader": "📈 Trader",
        "persona_investor": "💰 Yatırımcı",
        "persona_shop": "🏪 Kuyumcu",
        "persona_factory": "🏭 Mücevher Fabrikası",
        "persona_designer": "💍 Tasarımcı / Zanaatkar",
        "persona_buyer": "🛍️ Alıcı",
        "persona_lbl": "Profil",
        "nav_shopboard": "🏪  Dükkan Fiyat Panosu",
        "nav_invoice": "🧾  Fatura Hesaplayıcı",
        "nav_production": "🏭  Üretim Maliyeti",
        "nav_fairprice": "⚖️  Adil Fiyat Kontrolü",
        "nav_piecepricing": "💍  Parça Fiyatlandırma",
        "shop_buy": "Alış Fiyatı",
        "shop_sell": "Satış Fiyatı",
        "shop_spread": "Dükkan Marjı",
        "shop_display": "Görüntüleme Modu",
        "shop_currency": "Görüntüleme Para Birimi",
        "shop_name": "Dükkan Adı",
        "shop_purity": "Tüm Ayarları Göster",
        "shop_fullscreen": "Tam Ekran",
        "shop_embed": "Web Sitesine Göm",
        "inv_weight": "Ağırlık (gram)",
        "inv_purity": "Ayar",
        "inv_making": "İşçilik (gram başına)",
        "inv_vat": "KDV %",
        "inv_gold_val": "Altın Değeri",
        "inv_making_val": "Toplam İşçilik",
        "inv_vat_val": "Vergi",
        "inv_total": "Toplam",
        "inv_save": "Faturayı Kaydet",
        "inv_history": "Fatura Geçmişi",
        "inv_clear": "Geçmişi Temizle",
        "prod_recipe": "Alaşım Tarifi",
        "prod_gold_pct": "Altın %",
        "prod_weight": "Bitmiş Ağırlık (gram)",
        "prod_wastage": "Fire %",
        "prod_labour": "İşçilik (gram başına)",
        "prod_overhead": "Genel Giderler",
        "prod_gold_cost": "Ham Altın Maliyeti",
        "prod_total_cost": "Toplam Üretim Maliyeti",
        "prod_min_price": "Başabaş Fiyatı",
        "prod_suggest": "Önerilen Fiyat (%30 marj)",
        "prod_batch": "Parti Maliyeti",
        "prod_qty": "Parça Sayısı",
        "fp_quoted": "Teklif Edilen Fiyat",
        "fp_weight": "Ağırlık (gram)",
        "fp_purity": "Ayar",
        "fp_currency": "Para Birimi",
        "fp_spot_val": "Spot Değer",
        "fp_making_est": "Tahmini İşçilik",
        "fp_verdict_fair": "Adil Fiyat ✅",
        "fp_verdict_high": "Fazla Fiyat ⚠️",
        "fp_verdict_low": "Şüpheli Düşük Fiyat 🤔",
        "fp_saving": "Tasarrufunuz",
        "fp_overpay": "Fazla Ödüyorsunuz",
        "piece_gold_g": "Altın Ağırlığı (gram)",
        "piece_purity": "Ayar",
        "piece_gems": "Değerli Taşlar",
        "piece_gem_add": "Taş Ekle",
        "piece_labour_hrs": "Çalışma Saati",
        "piece_hourly": "Saatlik Ücret",
        "piece_overhead": "Genel Giderler",
        "piece_margin": "Hedef Marj %",
        "piece_cost": "Toplam Maliyet",
        "piece_price": "Önerilen Fiyat",
        "piece_save": "Tasarımı Kaydet",
        "piece_designs": "Kayıtlı Tasarımlarım",
        "geo_news": "Son Jeopolitik Haberler",
        "geo_timeline": "Olay Zaman Çizelgesi",
        "geo_filter": "Etkiye Göre Filtrele",
        "geo_all": "Tümü",
        "date_from": "Başlangıç Tarihi",
        "date_to": "Bitiş Tarihi",
        "quick_range": "Hızlı Aralık",
        "nav_geo": "🌍  Jeopolitik Harita",
        "nav_oilgold": "🛢️  Petrol & Altın",
        "geo_title": "Jeopolitik Harita",
        "geo_sub": "Aktif çatışma bölgeleri ve altın fiyatına etkileri",
        "geo_risk": "Jeopolitik Risk Endeksi",
        "geo_impact": "Altına Etkisi",
        "geo_events": "Aktif Olaylar",
        "geo_bullish": "Altın için yükseliş 🟢",
        "geo_bearish": "Altın için düşüş 🔴",
        "geo_neutral": "Nötr 🟡",
        "og_title": "Petrol & Altın",
        "og_sub": "Petrol-altın korelasyon takipçisi · Canlı endeks",
        "og_corr": "Korelasyon Katsayısı",
        "og_diverge": "Uyarı: Olağandışı ıraksama",
        "og_signal": "Korelasyon Sinyali",
        "sig_entry": "Giriş Fiyatı",
        "sig_target": "Hedef",
        "sig_stop": "Stop-Loss",
        "sig_rr": "Risk/Ödül Oranı",
        "sig_trade": "İşlem Detayları",
    },
    "اردو 🇵🇰": {
        "dir": "rtl",
        "app_name": "مانسا",
        "tagline": "سونے کی ذہانت",
        "nav_dashboard": "🏠  ڈیش بورڈ",
        "nav_markets":   "🌍  مارکیٹیں",
        "nav_charts":    "📈  چارٹس",
        "nav_simulator": "🔭  سمیولیٹر",
        "nav_predictions":"🤖  اے آئی پیشگوئی",
        "nav_data":      "📂  ڈیٹا ایکسپلورر",
        "nav_advisor":   "💬  اے آئی گولڈ مشیر",
        "nav_settings":  "⚙️  ترتیبات",

        "nav_signals":  "📡  ٹریڈنگ سگنل",
        "nav_newssent": "🧠  خبر تجزیہ",
        "nav_supply":  "⛏️  طلب و رسد",
        "nav_currency":"💱  کرنسی کنورٹر",
        "nav_alerts":     "🔔  قیمت الرٹ",
        "nav_heatmap":    "📊  ارتباط نقشہ",
        "nav_mansa_score":"🏆  مانسا سکور",
        "nav_zakat":      "☪️  زکوٰۃ کیلکولیٹر",
        "nav_compare":    "📈  اثاثہ موازنہ",
        "nav_journal":    "📓  ٹریڈ جرنل",
        "nav_goldmap":    "🌍  سونے کا نقشہ",
        "nav_drawdown":   "📉  اسٹریس ٹیسٹ",
        "nav_portfolio":  "💼  پورٹ فولیو",
        "nav_calculator": "🧮  حساب کتاب",
        "nav_calendar":   "📅  اقتصادی کیلنڈر",
        "nav_sentiment":  "🌡️  جذبات انڈیکس",
        "nav_sessions":   "🕐  مارکیٹ سیشن",
        "nav_game":   "🎮  مانسا گیم",
        "nav_about":  "🏛️  مانسا کے بارے میں",        "live": "لائیو",
        "not_financial": "مالی مشورہ نہیں",
        "quick_settings": "فوری ترتیبات",
        "weight_unit": "وزن کی اکائی",
        "purity": "سونے کا عیار",
        "theme": "ڈیزائن تھیم",
        "auto_refresh": "خودکار تازہ کاری (60 سیکنڈ)",
        "primary_market": "بنیادی مارکیٹ",
        "gold_intelligence": "سونے کی ذہانت پلیٹ فارم",
        "real_time": "ریئل ٹائم · عرب اور عالمی مارکیٹیں",
        "spot_price": "عالمی اسپاٹ قیمت",
        "key_stats": "سونے کے اہم اعداد و شمار",
        "wk52_high": "52 ہفتے کی بلند ترین",
        "wk52_low": "52 ہفتے کی کم ترین",
        "wk52_avg": "52 ہفتے کی اوسط",
        "ytd_return": "سال کا منافع",
        "gold_silver": "سونا/چاندی تناسب",
        "gold_oil": "سونا/تیل تناسب",
        "market_overview": "مارکیٹ کا جائزہ",
        "stocks_indices": "اسٹاکس اور انڈیکس",
        "all_purities": "تمام عیار",
        "prediction_date": "پیشگوئی کی تاریخ",
        "tomorrow": "کل",
        "best_model": "بہترین ماڈل",
        "algo_predictions": "الگورتھم پیشگوئیاں",
        "consensus": "اتفاق رائے",
        "model_high": "بلند ترین پیشگوئی",
        "model_low": "کم ترین پیشگوئی",
        "spread": "فرق",
        "vs_spot": "موجودہ قیمت کے مقابلے",
        "pred_multi_unit": "متعدد اکائیوں اور کرنسیوں میں پیشگوئی",
        "no_models": "کوئی ماڈل نہیں ملا۔ پہلے train_models.py چلائیں۔",
        "r2_scores": "ماڈل کی درستگی · R²",
        "prophet_forecast": "Prophet 30 دن کی پیشگوئی",
        "select_model": "ماڈل منتخب کریں",
        "disclaimer": "نوٹ: یہ معلومات صرف تعلیمی مقاصد کے لیے ہے۔ مالی مشورہ نہیں۔",
        "arab_markets": "عرب مارکیٹیں",
        "intl_markets": "بین الاقوامی مارکیٹیں",
        "purity_matrix": "عیار × مارکیٹ میٹرکس",
        "data_explorer": "ڈیٹا ایکسپلورر",
        "training_data": "ماڈل بنانے کے لیے استعمال شدہ ڈیٹا",
        "total_rows": "کل صفیں",
        "features": "خصوصیات",
        "from_date": "سے",
        "to_date": "تک",
        "filter_year": "سال کے مطابق فلٹر کریں",
        "columns_display": "کالم دکھائیں",
        "rows_show": "صفیں دکھائیں",
        "feat_desc": "خصوصیات کی تفصیل",
        "gold_history": "سونے کی قیمتوں کی تاریخ",
        "trading_advisor": "تجارتی مشیر",
        "live_signals": "لائیو تکنیکی اشارے",
        "price_levels": "قیمتوں کی سطحیں",
        "trade_setup": "ٹریڈ سیٹ اپ",
        "personalised_advice": "ذاتی نصیحت",
        "golden_rules": "سونے کی تجارت کے 10 سنہرے اصول",
        "macro_env": "موجودہ میکرو ماحول",
        "overall_signal": "مجموعی مارکیٹ سگنل",
        "bullish": "تیزی",
        "bearish": "مندی",
        "neutral": "غیر جانبدار",
        "support": "سپورٹ",
        "resistance": "مزاحمت",
        "entry": "داخلہ",
        "stop_loss": "اسٹاپ لاس",
        "take_profit": "منافع لیں",
        "settings": "ترتیبات",
        "design_theme": "ڈیزائن تھیم",
        "display_prefs": "ڈسپلے ترجیحات",
        "show_purity_table": "ڈیش بورڈ پر عیار جدول دکھائیں",
        "default_period": "پہلے سے طے شدہ چارٹ کی مدت",
        "active_markets": "فعال مارکیٹیں",
        "active_stocks": "فعال اسٹاکس اور انڈیکس",
        "conversion_ref": "اکائی × عیار تبادلہ حوالہ",
        "not_found_data": "تربیتی ڈیٹا نہیں ملا۔",
        "profile": "تاجر کا پروفائل",
        "profile_opts": ["ابتدائی 🌱", "درمیانی 📊", "اعلی درجے کا ⚡", "طویل مدتی سرمایہ کار 🏛️"],
        "chatbot_title": "💬  اے آئی گولڈ مشیر",
        "chatbot_sub": "سونے کی تجارت کے بارے میں کچھ بھی پوچھیں · اے آئی سے مدد",
        "chatbot_placeholder": "اپنا سوال لکھیں... مثلاً: کیا مجھے ابھی سونا خریدنا چاہیے؟",
        "chatbot_send": "بھیجیں",
        "chatbot_clear": "گفتگو صاف کریں",
        "chatbot_thinking": "سوچ رہا ہے...",
        "chatbot_welcome": "سلام! میں آپ کا اے آئی گولڈ مشیر ہوں۔ سونے کے بارے میں کچھ بھی پوچھیں!",
        "chatbot_error": "خطا ہوئی۔ دوبارہ کوشش کریں۔",
        "models_demo_note": "⚠️ اے آئی ماڈل نہیں ملے۔ ڈیمو پیشگوئیاں دکھائی جا رہی ہیں۔",
        "no_csv": "ڈیٹا فائل نہیں ملی۔",
        "indicators": "اشارے",
        "period": "مدت",
        "chart_type": "چارٹ کی قسم",
        "live_prices": "لائیو قیمتیں",
        "yf_delay": "Yahoo Finance · ~15 منٹ تاخیر",
        "live_snapshot": "📊 لائیو مارکیٹ جائزہ",
        "col_asset": "اثاثہ",
        "col_price": "قیمت",
        "col_change": "تبدیلی",
        "col_status": "حیثیت",
        "data_up_to": "ڈیٹا تاریخ:",
        "added": "شامل کیا",
        "new_rows_to": "نئی قطاریں · ڈیٹا اپ ٹو:",
        "update_error": "اپ ڈیٹ خطا:",
        "latest": "تازہ ترین",
        "rows_added": "قطاریں شامل",
        "rows_filtered": "فلٹر شدہ قطاریں",
        "total_lc": "کل",
        "sparklines": "اسپارک لائنز · آخری 12 ماہ",
        "quick_questions": "فوری سوالات",
        "brand_logos": "لوگو · ہر تھیم کے لیے",
        "developer": "ڈویلپر اور خالق",
        "why_mansa": "مانسا کیوں؟ · مانسا موسیٰ کی کہانی",
        "mansa_title": "مانسا موسیٰ — تاریخ کا سب سے امیر آدمی",
        "hist_highlights": "تاریخی نکات",
        "mansa_caption": "مانسا موسیٰ · مالی کا شہنشاہ",
        "mansa_dates": "1280 – 1337 عیسوی",
        "mosque_caption": "جنگریبر مسجد · ٹمبکٹو",
        "mosque_built": "مانسا موسیٰ نے بنائی · 1327 عیسوی",
        "empire_lbl": "سلطنت",
        "trade_routes": "تجارتی راستے",
        "empire_peak": "مالی سلطنت عروج پر",
        "empire_century": "14ویں صدی · 20 لاکھ کلومیٹر²",
        "about_platform": "پلیٹ فارم کے بارے میں · عون انصاری",
        "copyright": "© 2025 عون انصاری",
        "platform_name": "مانسا گولڈ انٹیلیجنس پلیٹ فارم",
        "position_details": "پوزیشن تفصیلات",
        "results": "نتائج",
        "risk_amount": "خطرے کی رقم",
        "position_size_lbl": "پوزیشن کا حجم",
        "position_cost": "پوزیشن کی لاگت",
        "risk_reward": "خطرہ/انعام",
        "calc_pnl_sub": "متوقع نفع یا نقصان کا حساب کریں",
        "cost_lbl": "لاگت",
        "value_lbl": "قدر",
        "calc_margin_sub": "لیوریج ٹریڈنگ کے لیے مطلوبہ مارجن کا حساب کریں",
        "notional": "برائے نام قدر",
        "req_margin": "مطلوبہ مارجن",
        "per_dollar": "فی $1 قدر",
        "leverage_lbl": "لیوریج",
        "calc_be_sub": "تجارتی اخراجات سمیت بریک ایون قیمت کا حساب کریں",
        "breakeven": "بریک ایون قیمت",
        "above_entry": "داخلے سے اوپر",
        "spread_lbl": "فرق",
        "commission_lbl": "کمیشن",
        "swap_lbl": "سویپ لاگت",
        "today_lbl": "آج ⚡",
        "forecast_lbl": "پیشگوئی:",
        "actual_lbl": "حقیقی:",
        "fear_lbl": "خوف",
        "greed_lbl": "لالچ",
        "what_means_gold": "یہ سونے کے لیے کیا مطلب رکھتا ہے؟",
        "index_components": "انڈیکس کے اجزاء",
        "market_status_now": "ابھی مارکیٹ کی صورتحال",
        "arab_mkts_primary": "⭐ عرب مارکیٹیں · اردن بنیادی",
        "intl_mkts_lbl": "🌐 بین الاقوامی مارکیٹیں",
        "timeline_24h": "24 گھنٹے کا ٹائم لائن (UTC)",
        "gold_corr_assets": "🥇 سونے کا ہر اثاثے سے تعلق",
        "five_pillars": "پانچ ستون",
        "enter_holdings": "اپنا سونا درج کریں",
        "gold24k": "24K سونا (گرام)",
        "gold22k": "22K سونا (گرام)",
        "gold21k": "21K سونا (گرام)",
        "gold18k": "18K سونا (گرام)",
        "gold14k": "14K سونا (گرام)",
        "zakat_result": "زکوٰۃ حساب کا نتیجہ",
        "total_gold_lbl": "کل سونا",
        "nisab_lbl": "نصاب",
        "total_value_lbl": "کل قدر",
        "zakat_due": "واجب الادا زکوٰۃ (2.5%)",
        "returns_lb": "🏆 منافع کی درجہ بندی",
        "trade_log": "تجارتی لاگ",
        "nisab_based": "لائیو چاندی کی قیمت پر: 24K سونا ",
        "current_price_lbl": "موجودہ قیمت",
        "interactive_sim": "انٹرایکٹو سمیولیشن",
        "cb_reserves": "🏦 مرکزی بینک سونے کے ذخائر (ٹن) · 2025",
        "mine_prod": "⛏️ سونے کی کان کنی (ٹن/سال 2024)",
        "global_demand": "📊 عالمی سونے کی طلب (ٹن 2024)",
        "major_etfs": "📊 بڑے گولڈ ETFs",
        "gram_arab": "عرب کرنسیوں میں سونے کی گرام قیمت",
        "auto_refresh_60": "قیمتیں ہر 60 سیکنڈ میں تازہ ہوتی ہیں",
        "overall_signal_ts": "مانسا مجموعی سگنل · ",
        "confidence_lbl": "اعتماد",
        "buy_lbl": "خریدیں",
        "sell_lbl": "بیچیں",
        "all_signals": "📋 تمام سگنل تفصیلات",
        "action_buy": "تجویز کردہ اقدام: خریدیں",
        "entry_lbl": "داخلہ",
        "action_sell": "تجویز کردہ اقدام: بیچیں یا انتظار کریں",
        "exit_sell": "خروج/فروخت",
        "action_wait": "تجویز کردہ اقدام: واضح سگنل کا انتظار",
        "bullish_lbl": "تیزی",
        "neutral_lbl": "غیر جانبدار",
        "bearish_lbl": "مندی",
        "very_bearish": "بہت مندی",
        "average_lbl": "اوسط",
        "very_bullish": "بہت تیزی",
        "article_analysis": "📰 ہر مضمون کا تجزیہ",
        "session_score": "سیشن اسکور",
        "high_score_lbl": "سب سے زیادہ اسکور",
        "start_game": "گیم شروع کریں",
        "amazing_10": "شاندار! 10 سونے کے سکے!",
        "game_over": "گیم ختم ☽",
        "gold_collected": "اکٹھا کیا سونا",
        "play_again": "دوبارہ کھیلیں",
        "restart_lbl": "دوبارہ شروع",
        "calc_pos_sub": "اپنے خطرے کی برداشت کے مطابق صحیح پوزیشن کا حجم معلوم کریں",
        "tomorrow_lbl": "کل",
        "days_lbl": " دن میں",
        "neutral_sig": "غیر جانبدار",
        "stop_loss_lbl": "اسٹاپ لاس",
        "news_sources": "ہر 15 منٹ میں تازہ · ذرائع: Reuters · Kitco · FXStreet · MarketWatch",
        "game_instructions": "⬆ SPACE یا کلک کریں · سونے کے سکے اکٹھے کریں · بہترین اسکور کریں!",
        "update_hs": "🏆 اعلیٰ اسکور اپ ڈیٹ",
        "feat_info": "اشارے کی معلومات · تاریخ · تعلقات · سگنل",
        "history_lbl": "📖 تاریخ",
        "relation_gold": "🔗 سونے سے تعلق",
        "closes_in": "بند ہوتا ہے",
        "opens_in": "کھلتا ہے",
        "neutral_lc": "غیر جانبدار",
        "add_new_alert": "نیا الرٹ شامل کریں",
        "target_price": "ہدف قیمت (USD/oz)",
        "direction_lbl": "سمت",
        "alert_label": "الرٹ کا عنوان",
        "why_level": "یہ سطح کیوں اہم ہے؟",
        "save_alert": "الرٹ محفوظ کریں",
        "alert_saved": "🔔 الرٹ محفوظ ہو گیا!",
        "alert_triggered_lbl": "الرٹ فعال ہو گیا!",
        "current_lbl": "موجودہ",
        "no_alerts_msg": "کوئی فعال الرٹ نہیں — اوپر ہدف قیمت شامل کریں",
        "clear_triggered": "فعال شدہ صاف کریں",
        "alert_at": "الرٹ بقیمت",
        "distance_lbl": "فاصلہ",
        "nav_demo": "🎯  ڈیمو ٹریڈنگ",
        "nav_savings": "🪙  گولڈ بچت پلان",
        "nav_report": "📋  ہفتہ وار رپورٹ",
        "nav_widget": "🔌  ایمبیڈ وجٹ",
        "nav_cb": "🏦  مرکزی بینک ذخائر",
        "demo_title": "ڈیمو ٹریڈنگ",
        "demo_sub": "مجازی رقم سے تجارت کریں · کوئی حقیقی خطرہ نہیں",
        "demo_balance": "ڈیمو بیلنس",
        "demo_buy": "سونا خریدیں",
        "demo_sell": "سونا بیچیں",
        "demo_reset": "اکاؤنٹ ری سیٹ کریں",
        "demo_pnl": "نفع / نقصان",
        "demo_holdings": "ڈیمو پورٹ فولیو",
        "demo_trades": "ڈیمو تجارتی لاگ",
        "demo_note": "💡 یہ ڈیمو اکاؤنٹ ہے — کوئی حقیقی رقم استعمال نہیں ہوتی",
        "demo_qty": "مقدار (گرام)",
        "demo_cleared": "ڈیمو اکاؤنٹ ری سیٹ ہو گیا",
        "savings_title": "گولڈ بچت پلان",
        "savings_sub": "ہر ماہ سونے میں بچت کریں · افراطِ زر سے تحفظ",
        "savings_monthly": "ماہانہ رقم",
        "savings_since": "شروع کی تاریخ",
        "savings_curr": "کرنسی",
        "savings_add": "پلان شامل کریں",
        "savings_total": "کل بچت",
        "savings_gold": "جمع شدہ سونا",
        "savings_value": "موجودہ قیمت",
        "savings_gain": "نقد بچت کے مقابلے فائدہ",
        "savings_vs": "موازنہ: سونا بمقابلہ نقد بچت",
        "savings_note": "💡 سونے میں بچت کرنسی کی قدر میں کمی سے بچاتی ہے",
        "savings_plan": "میرا بچت پلان",
        "savings_delete": "پلان حذف کریں",
        "report_title": "ہفتہ وار رپورٹ",
        "report_sub": "سونے کی مارکیٹ کا خلاصہ · خودکار تیار",
        "report_gen": "رپورٹ بنائیں",
        "report_dl": "PDF ڈاؤنلوڈ",
        "report_week": "ہفتے کی رپورٹ",
        "report_perf": "ہفتہ وار کارکردگی",
        "report_outlook": "اگلے ہفتے کا جائزہ",
        "report_events": "آنے والے اقتصادی واقعات",
        "report_signals": "سگنل خلاصہ",
        "widget_title": "ایمبیڈ وجٹ",
        "widget_sub": "اپنی ویب سائٹ پر ایک لائن سے لائیو سونے کی قیمت شامل کریں",
        "widget_copy": "کوڈ کاپی کریں",
        "widget_preview": "وجٹ پریویو",
        "widget_style": "انداز",
        "widget_curr2": "کرنسی",
        "widget_copied": "✅ کاپی ہو گیا!",
        "widget_instruct": "یہ کوڈ کاپی کریں اور اپنی ویب سائٹ میں پیسٹ کریں",
        "cb_title": "مرکزی بینک ذخائر",
        "cb_sub": "عالمی سونے کے ذخائر · WGC 2025 ڈیٹا",
        "cb_arab": "عرب ممالک",
        "cb_world": "دنیا کے سب سے بڑے ذخائر",
        "cb_rank": "درجہ",
        "cb_country": "ملک",
        "cb_tonnes": "ٹن",
        "cb_pct": "ذخائر کا %",
        "cb_trend": "رجحان",
        "cb_insight": "تجزیہ",
        "persona_select": "صارف کی قسم منتخب کریں",
        "persona_trader": "📈 ٹریڈر",
        "persona_investor": "💰 سرمایہ کار",
        "persona_shop": "🏪 سونے کی دکان کا مالک",
        "persona_factory": "🏭 زیورات فیکٹری",
        "persona_designer": "💍 ڈیزائنر / کاریگر",
        "persona_buyer": "🛍️ خریدار",
        "persona_lbl": "پروفائل",
        "nav_shopboard": "🏪  شاپ پرائس بورڈ",
        "nav_invoice": "🧾  انوائس کیلکولیٹر",
        "nav_production": "🏭  پیداواری لاگت",
        "nav_fairprice": "⚖️  منصفانہ قیمت چیکر",
        "nav_piecepricing": "💍  پیس پرائسنگ اسٹوڈیو",
        "shop_buy": "خرید قیمت",
        "shop_sell": "فروخت قیمت",
        "shop_spread": "دکان کا مارجن",
        "shop_display": "ڈسپلے موڈ",
        "shop_currency": "ڈسپلے کرنسی",
        "shop_name": "دکان کا نام",
        "shop_purity": "تمام خالصیت دکھائیں",
        "shop_fullscreen": "فل اسکرین",
        "shop_embed": "ویب سائٹ میں شامل کریں",
        "inv_weight": "وزن (گرام)",
        "inv_purity": "خالصیت",
        "inv_making": "بنائی چارج (فی گرام)",
        "inv_vat": "VAT %",
        "inv_gold_val": "سونے کی قیمت",
        "inv_making_val": "کل بنائی",
        "inv_vat_val": "ٹیکس",
        "inv_total": "کل",
        "inv_save": "انوائس محفوظ کریں",
        "inv_history": "انوائس کی تاریخ",
        "inv_clear": "تاریخ صاف کریں",
        "prod_recipe": "مرکب نسخہ",
        "prod_gold_pct": "سونا %",
        "prod_weight": "مکمل وزن (گرام)",
        "prod_wastage": "ضیاع %",
        "prod_labour": "محنت لاگت (فی گرام)",
        "prod_overhead": "اوور ہیڈ",
        "prod_gold_cost": "خام سونے کی لاگت",
        "prod_total_cost": "کل پیداواری لاگت",
        "prod_min_price": "بریک ایون قیمت",
        "prod_suggest": "تجویز کردہ قیمت (30% مارجن)",
        "prod_batch": "بیچ لاگت",
        "prod_qty": "ٹکڑوں کی تعداد",
        "fp_quoted": "کوٹ کی گئی قیمت",
        "fp_weight": "وزن (گرام)",
        "fp_purity": "خالصیت",
        "fp_currency": "کرنسی",
        "fp_spot_val": "اسپاٹ ویلیو",
        "fp_making_est": "تخمینی بنائی",
        "fp_verdict_fair": "مناسب قیمت ✅",
        "fp_verdict_high": "زیادہ قیمت ⚠️",
        "fp_verdict_low": "مشکوک طور پر کم 🤔",
        "fp_saving": "آپ کی بچت",
        "fp_overpay": "آپ زیادہ ادا کر رہے ہیں",
        "piece_gold_g": "سونے کا وزن (گرام)",
        "piece_purity": "خالصیت",
        "piece_gems": "قیمتی پتھر",
        "piece_gem_add": "پتھر شامل کریں",
        "piece_labour_hrs": "محنت کے گھنٹے",
        "piece_hourly": "فی گھنٹہ شرح",
        "piece_overhead": "اوور ہیڈ",
        "piece_margin": "ہدف مارجن %",
        "piece_cost": "کل لاگت",
        "piece_price": "تجویز کردہ قیمت",
        "piece_save": "ڈیزائن محفوظ کریں",
        "piece_designs": "میرے محفوظ ڈیزائن",
        "geo_news": "تازہ ترین جغرافیائی خبریں",
        "geo_timeline": "واقعات کا ٹائم لائن",
        "geo_filter": "اثر کے لحاظ سے فلٹر",
        "geo_all": "تمام",
        "date_from": "شروع کی تاریخ",
        "date_to": "ختم کی تاریخ",
        "quick_range": "فوری رینج",
        "nav_geo": "🌍  جغرافیائی نقشہ",
        "nav_oilgold": "🛢️  تیل اور سونا",
        "geo_title": "جغرافیائی سیاسی نقشہ",
        "geo_sub": "دنیا بھر کے تنازعات اور سونے کی قیمت پر اثرات",
        "geo_risk": "جغرافیائی خطرے کا اشاریہ",
        "geo_impact": "سونے پر اثر",
        "geo_events": "فعال واقعات",
        "geo_bullish": "سونے کے لیے مثبت 🟢",
        "geo_bearish": "سونے کے لیے منفی 🔴",
        "geo_neutral": "غیرجانبدار 🟡",
        "og_title": "تیل اور سونا",
        "og_sub": "تیل-سونا ارتباط ٹریکر · لائیو",
        "og_corr": "ارتباط گنجائش",
        "og_diverge": "انتباہ: غیر معمولی انحراف",
        "og_signal": "ارتباط اشارہ",
        "sig_entry": "داخلے کی قیمت",
        "sig_target": "ہدف",
        "sig_stop": "سٹاپ لاس",
        "sig_rr": "خطرہ/انعام تناسب",
        "sig_trade": "تجارتی تفصیلات",
    },
}

# ═══════════════════════════════════════════════════════════════════════════════
# THEME DEFINITIONS  — Islamic & Arab is the default
# ═══════════════════════════════════════════════════════════════════════════════
THEMES = {
    "الحضارة الإسلامية ☽": {
        "bg":"#050408","bg2":"#080610","card":"#0D0A18","card2":"#110E20",
        "border":"#251540","border2":"#321C55",
        "gold":"#D4A017","gold_hi":"#F5C842","gold_pale":"#FAE5A0","gold_dark":"#7A5A00",
        "gold_glow":"rgba(212,160,23,0.27)","text":"#F0E8D5","muted":"#806050","dim":"#4A3828",
        "green":"#5CC86A","red":"#D9534F","blue":"#7DAACC","accent":"#B8860B",
        "brand":"☽","font_h":"Amiri","font_b":"Amiri","font_m":"JetBrains Mono",
        "desc":"الحضارة الإسلامية والعربية",
        "arabesque":"rgba(212,160,23,0.06)",
    },
    "العملة الذهبية القديمة ⬡": {
        "bg":"#080807","bg2":"#0D0C0A","card":"#111109","card2":"#161410",
        "border":"#2C2510","border2":"#3A3018",
        "gold":"#C9960C","gold_hi":"#F2C94C","gold_pale":"#F5DFA0","gold_dark":"#7A5A04",
        "gold_glow":"rgba(201,150,12,0.27)","text":"#EDE8D8","muted":"#7A7060","dim":"#4A4535",
        "green":"#52B788","red":"#E05C5C","blue":"#6EA8C8","accent":"#C9960C",
        "brand":"⬡","font_h":"Cinzel","font_b":"Cormorant Garamond","font_m":"JetBrains Mono",
        "desc":"العملات الذهبية القديمة",
        "arabesque":"rgba(201,150,12,0.04)",
    },
    "قاعة التداول ◈": {
        "bg":"#050A0F","bg2":"#080F16","card":"#0C1520","card2":"#101C28",
        "border":"#162030","border2":"#1E2E40",
        "gold":"#00C9A7","gold_hi":"#00FFD4","gold_pale":"#A0F0E0","gold_dark":"#007A66",
        "gold_glow":"rgba(0,201,167,0.27)","text":"#D0ECE8","muted":"#507060","dim":"#304540",
        "green":"#39D98A","red":"#FF4D6D","blue":"#4DA6FF","accent":"#F2C94C",
        "brand":"◈","font_h":"Share Tech Mono","font_b":"IBM Plex Sans","font_m":"Share Tech Mono",
        "desc":"قاعة التداول / بلومبرغ",
        "arabesque":"rgba(0,201,167,0.04)",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# LOGO SVGs — one per theme, matching the uploaded brand image
# ─────────────────────────────────────────────────────────────────────────────
def get_logo_svg(theme_key, width=180):
    """Return inline SVG logo matching the theme brand image."""

    if "الحضارة الإسلامية" in theme_key:
        # ── Islamic Logo v3 — precisely matching the new uploaded image:
        #   · Purple-violet-to-gold gradient mosque (richer purples)
        #   · Very tall slender minarets with long pointed gold spires
        #   · Large bright purple dome dominating the center
        #   · Gold crescent moon floating above
        #   · Dark shield/badge with angular pentagon outline in gold
        #   · Dark starry ground layer with sparkle dots
        #   · Bold gold diagonal arrow across bottom
        #   · MANSA in wide-spaced gold-gradient bold text
        return f"""<svg width="{width}" height="{int(width*0.85)}" viewBox="0 0 320 272" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <!-- Dark purple background gradient -->
    <linearGradient id="g_bg" x1="0" y1="0" x2="0.6" y2="1">
      <stop offset="0%" stop-color="#1E0840"/>
      <stop offset="100%" stop-color="#07030F"/>
    </linearGradient>
    <!-- Mosque body: deep purple-blue like the image -->
    <linearGradient id="g_body" x1="0" y1="0" x2="0.4" y2="1">
      <stop offset="0%" stop-color="#8B35C8"/>
      <stop offset="55%" stop-color="#6A1FA0"/>
      <stop offset="100%" stop-color="#4A0E78"/>
    </linearGradient>
    <!-- Main dome: bright purple top, deep bottom -->
    <linearGradient id="g_dome" x1="0.2" y1="0" x2="0.8" y2="1">
      <stop offset="0%" stop-color="#C060FF"/>
      <stop offset="35%" stop-color="#9B35D8"/>
      <stop offset="100%" stop-color="#5010A0"/>
    </linearGradient>
    <!-- Dome soft highlight -->
    <radialGradient id="g_dome_hl" cx="38%" cy="28%" r="50%">
      <stop offset="0%" stop-color="#E0A8FF" stop-opacity="0.55"/>
      <stop offset="100%" stop-color="#9B35D8" stop-opacity="0"/>
    </radialGradient>
    <!-- Purple glow behind mosque -->
    <radialGradient id="g_glow" cx="50%" cy="55%" r="50%">
      <stop offset="0%" stop-color="#8B35C8" stop-opacity="0.45"/>
      <stop offset="100%" stop-color="#4A0E78" stop-opacity="0"/>
    </radialGradient>
    <!-- Gold text gradient: yellow-gold left to darker gold right -->
    <linearGradient id="g_text" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0%" stop-color="#FFE050"/>
      <stop offset="40%" stop-color="#F5C830"/>
      <stop offset="100%" stop-color="#C8920A"/>
    </linearGradient>
    <!-- Arrow gradient -->
    <linearGradient id="g_arrow" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0%" stop-color="#B87A08"/>
      <stop offset="50%" stop-color="#F5C830"/>
      <stop offset="100%" stop-color="#FFE050"/>
    </linearGradient>
    <!-- Shield outer gradient (subtle) -->
    <linearGradient id="g_shield_bg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#2A0860" stop-opacity="0.9"/>
      <stop offset="100%" stop-color="#08030E" stop-opacity="0.95"/>
    </linearGradient>
  </defs>

  <!-- ═══ SHIELD / BADGE SHAPE ═══ -->
  <!-- Outer gold border — angular pentagon matching the image -->
  <path d="M160 6 L298 46 L298 148 Q298 222 160 268 Q22 222 22 148 L22 46 Z"
        fill="none" stroke="#C8920A" stroke-width="3"/>
  <!-- Second inner gold line — double border like the image -->
  <path d="M160 14 L290 51 L290 148 Q290 214 160 260 Q30 214 30 148 L30 51 Z"
        fill="url(#g_shield_bg)" stroke="#F5C830" stroke-width="1" stroke-opacity="0.4"/>

  <!-- ═══ BACKGROUND GLOW ═══ -->
  <ellipse cx="160" cy="150" rx="115" ry="105" fill="url(#g_glow)"/>

  <!-- ═══ CITY SILHOUETTE (dark buildings in background) ═══ -->
  <rect x="36"  y="168" width="16" height="52" fill="#180630" opacity="0.8"/>
  <rect x="54"  y="176" width="12" height="44" fill="#180630" opacity="0.65"/>
  <rect x="67"  y="160" width="14" height="60" fill="#180630" opacity="0.55"/>
  <rect x="238" y="160" width="14" height="60" fill="#180630" opacity="0.55"/>
  <rect x="253" y="176" width="12" height="44" fill="#180630" opacity="0.65"/>
  <rect x="266" y="168" width="16" height="52" fill="#180630" opacity="0.8"/>

  <!-- ═══ LEFT TALL MINARET (slender, very tall) ═══ -->
  <!-- Main shaft -->
  <rect x="72" y="140" width="20" height="88" rx="2" fill="url(#g_body)"/>
  <!-- Upper shaft (narrower) -->
  <rect x="75" y="110" width="14" height="34" rx="1" fill="url(#g_body)"/>
  <!-- Balcony 1 (wider band) -->
  <rect x="67" y="155" width="30" height="6" rx="2" fill="#C8920A"/>
  <!-- Balcony 2 -->
  <rect x="70" y="123" width="24" height="4" rx="1" fill="#C8920A" opacity="0.85"/>
  <!-- Spire (long, pointed, gold) -->
  <polygon points="82,76 90,110 74,110" fill="#C8920A"/>
  <!-- Spire tip finial -->
  <circle cx="82" cy="73" r="4.5" fill="#FFE050"/>
  <line x1="82" y1="73" x2="82" y2="64" stroke="#FFE050" stroke-width="1.5"/>
  <circle cx="82" cy="62" r="2" fill="#FFE050"/>
  <!-- Arch windows -->
  <path d="M79 148 Q79 140 82 138 Q85 140 85 148 Z" fill="#F5C830" opacity="0.45"/>
  <path d="M79 170 Q79 162 82 160 Q85 162 85 170 Z" fill="#F5C830" opacity="0.3"/>

  <!-- ═══ RIGHT TALL MINARET ═══ -->
  <rect x="228" y="140" width="20" height="88" rx="2" fill="url(#g_body)"/>
  <rect x="231" y="110" width="14" height="34" rx="1" fill="url(#g_body)"/>
  <rect x="223" y="155" width="30" height="6" rx="2" fill="#C8920A"/>
  <rect x="226" y="123" width="24" height="4" rx="1" fill="#C8920A" opacity="0.85"/>
  <polygon points="238,76 246,110 230,110" fill="#C8920A"/>
  <circle cx="238" cy="73" r="4.5" fill="#FFE050"/>
  <line x1="238" y1="73" x2="238" y2="64" stroke="#FFE050" stroke-width="1.5"/>
  <circle cx="238" cy="62" r="2" fill="#FFE050"/>
  <path d="M235 148 Q235 140 238 138 Q241 140 241 148 Z" fill="#F5C830" opacity="0.45"/>
  <path d="M235 170 Q235 162 238 160 Q241 162 241 170 Z" fill="#F5C830" opacity="0.3"/>

  <!-- ═══ INNER SIDE MINARETS (smaller pair) ═══ -->
  <!-- Left inner -->
  <rect x="114" y="163" width="13" height="62" rx="1" fill="url(#g_body)" opacity="0.9"/>
  <rect x="111" y="174" width="19" height="3.5" rx="1" fill="#C8920A" opacity="0.75"/>
  <polygon points="120,146 126,163 114,163" fill="#C8920A"/>
  <circle cx="120" cy="144" r="3" fill="#FFE050"/>

  <!-- Right inner -->
  <rect x="193" y="163" width="13" height="62" rx="1" fill="url(#g_body)" opacity="0.9"/>
  <rect x="190" y="174" width="19" height="3.5" rx="1" fill="#C8920A" opacity="0.75"/>
  <polygon points="199,146 205,163 193,163" fill="#C8920A"/>
  <circle cx="199" cy="144" r="3" fill="#FFE050"/>

  <!-- ═══ MOSQUE BODY ═══ -->
  <rect x="112" y="188" width="96" height="38" rx="2" fill="url(#g_body)"/>
  <!-- Arch windows (3) -->
  <path d="M124 226 Q124 212 131 209 Q138 212 138 226 Z" fill="#F5C830" opacity="0.4"/>
  <path d="M150 226 Q150 212 157 209 Q164 212 164 226 Z" fill="#F5C830" opacity="0.4"/>
  <path d="M176 226 Q176 212 183 209 Q190 212 190 226 Z" fill="#F5C830" opacity="0.4"/>
  <!-- Body collar -->
  <rect x="108" y="183" width="104" height="9" rx="2" fill="url(#g_body)"/>

  <!-- ═══ SIDE SMALL DOMES ═══ -->
  <path d="M112 186 Q112 165 126 161 Q140 165 140 186 Z" fill="url(#g_body)" opacity="0.88"/>
  <path d="M180 186 Q180 165 194 161 Q208 165 208 186 Z" fill="url(#g_body)" opacity="0.88"/>

  <!-- ═══ MAIN CENTER DOME (large, bright purple) ═══ -->
  <!-- Dome base drum -->
  <rect x="125" y="181" width="70" height="10" rx="3" fill="url(#g_body)"/>
  <!-- Main dome arch -->
  <path d="M112 188 Q112 122 160 116 Q208 122 208 188 Z" fill="url(#g_dome)"/>
  <!-- Dome highlight (no filter — pure SVG) -->
  <path d="M112 188 Q112 122 160 116 Q208 122 208 188 Z" fill="url(#g_dome_hl)"/>
  <!-- Dome sheen line -->
  <path d="M130 162 Q127 136 160 126 Q182 133 184 162"
        fill="none" stroke="#E8B8FF" stroke-width="1.8" opacity="0.45" stroke-linecap="round"/>
  <!-- Finial pole -->
  <line x1="160" y1="116" x2="160" y2="99" stroke="#C8920A" stroke-width="2.5"/>
  <circle cx="160" cy="96" r="4.5" fill="#FFE050"/>

  <!-- ═══ CRESCENT MOON ═══ -->
  <!-- Large gold crescent — outer circle minus offset inner circle -->
  <circle cx="160" cy="70" r="18" fill="#F5C830"/>
  <circle cx="168" cy="65" r="15" fill="#07030F"/>
  <!-- Small 5-point star beside crescent -->
  <g transform="translate(184,60)">
    <polygon points="0,-7 1.7,-2.2 6.7,-2.2 2.8,1 4.2,6 0,3.5 -4.2,6 -2.8,1 -6.7,-2.2 -1.7,-2.2"
             fill="#F5C830" opacity="0.95"/>
  </g>

  <!-- ═══ SCATTERED STARS ═══ -->
  <circle cx="46"  cy="82"  r="2"   fill="#F5C830" opacity="0.9"/>
  <circle cx="274" cy="82"  r="2"   fill="#F5C830" opacity="0.9"/>
  <circle cx="38"  cy="110" r="1.5" fill="#F5C830" opacity="0.7"/>
  <circle cx="282" cy="110" r="1.5" fill="#F5C830" opacity="0.7"/>
  <circle cx="55"  cy="132" r="1.2" fill="#C8920A" opacity="0.8"/>
  <circle cx="265" cy="132" r="1.2" fill="#C8920A" opacity="0.8"/>
  <!-- Tiny sparkle dots on ground area -->
  <circle cx="90"  cy="230" r="1"   fill="#F5C830" opacity="0.5"/>
  <circle cx="130" cy="240" r="1"   fill="#F5C830" opacity="0.4"/>
  <circle cx="160" cy="245" r="1.2" fill="#F5C830" opacity="0.6"/>
  <circle cx="195" cy="240" r="1"   fill="#F5C830" opacity="0.4"/>
  <circle cx="230" cy="230" r="1"   fill="#F5C830" opacity="0.5"/>

  <!-- ═══ GROUND STEPS ═══ -->
  <rect x="94"  y="226" width="132" height="6"  rx="2" fill="#C8920A" opacity="0.75"/>
  <rect x="80"  y="232" width="160" height="5"  rx="2" fill="#C8920A" opacity="0.48"/>
  <rect x="64"  y="237" width="192" height="4"  rx="2" fill="#C8920A" opacity="0.25"/>

  <!-- ═══ UPWARD ARROW (bold, gold, diagonal) ═══ -->
  <line x1="72" y1="258" x2="235" y2="240"
        stroke="url(#g_arrow)" stroke-width="3.5" stroke-linecap="round"/>
  <polygon points="235,240 219,231 224,244" fill="#FFE050"/>
  <!-- Arrow sparkle accents -->
  <circle cx="105" cy="254" r="2.2" fill="#F5C830" opacity="0.65"/>
  <circle cx="150" cy="250" r="2.2" fill="#F5C830" opacity="0.65"/>
  <circle cx="195" cy="246" r="2.2" fill="#F5C830" opacity="0.65"/>

  <!-- ═══ MANSA TEXT (bold, wide, gold gradient) ═══ -->
  <text x="160" y="270" text-anchor="middle"
        font-family="Georgia,'Times New Roman',serif"
        font-size="30" font-weight="900" letter-spacing="7"
        fill="url(#g_text)">MANSA</text>
</svg>"""

    elif "العملة الذهبية" in theme_key:
        # Logo 2: Bar chart + upward curve arrow + gold text (middle logo from image)
        return f"""<svg width="{width}" height="{int(width*0.65)}" viewBox="0 0 180 117" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="gBar1" x1="0%" y1="100%" x2="0%" y2="0%">
      <stop offset="0%" style="stop-color:#E05C5C"/>
      <stop offset="100%" style="stop-color:#FF8C42"/>
    </linearGradient>
    <linearGradient id="gBar2" x1="0%" y1="100%" x2="0%" y2="0%">
      <stop offset="0%" style="stop-color:#FF8C42"/>
      <stop offset="100%" style="stop-color:#F5C842"/>
    </linearGradient>
    <linearGradient id="gBar3" x1="0%" y1="100%" x2="0%" y2="0%">
      <stop offset="0%" style="stop-color:#52B788"/>
      <stop offset="100%" style="stop-color:#00C9A7"/>
    </linearGradient>
    <linearGradient id="gBar4" x1="0%" y1="100%" x2="0%" y2="0%">
      <stop offset="0%" style="stop-color:#52B788"/>
      <stop offset="100%" style="stop-color:#39D98A"/>
    </linearGradient>
    <linearGradient id="gArrow" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" style="stop-color:#6EA8C8"/>
      <stop offset="100%" style="stop-color:#00C9A7"/>
    </linearGradient>
  </defs>
  <!-- Bars -->
  <rect x="28" y="70" width="16" height="22" rx="2" fill="url(#gBar1)"/>
  <rect x="50" y="56" width="16" height="36" rx="2" fill="url(#gBar2)"/>
  <rect x="72" y="44" width="16" height="48" rx="2" fill="url(#gBar3)"/>
  <rect x="94" y="30" width="16" height="62" rx="2" fill="url(#gBar3)"/>
  <rect x="116" y="18" width="16" height="74" rx="2" fill="url(#gBar4)"/>
  <!-- Curve arrow -->
  <path d="M30 80 Q70 40 148 16" fill="none" stroke="url(#gArrow)" stroke-width="3.5" stroke-linecap="round"/>
  <polygon points="148,16 138,22 144,30" fill="#00C9A7"/>
  <!-- MANSA text -->
  <text x="90" y="105" text-anchor="middle" font-family="serif" font-size="20"
        font-weight="900" letter-spacing="5" fill="#F5DFA0">MANSA</text>
  <!-- SMART TRADING subtext -->
  <text x="90" y="115" text-anchor="middle" font-family="sans-serif" font-size="7"
        letter-spacing="3" fill="#C9960C">SMART TRADING</text>
</svg>"""

    else:
        # Logo 3: Rainbow circle + bar chart + arrow (right logo from image)
        return f"""<svg width="{width}" height="{int(width*0.7)}" viewBox="0 0 180 126" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="gRainbow" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#E05C5C"/>
      <stop offset="25%" style="stop-color:#F5C842"/>
      <stop offset="50%" style="stop-color:#39D98A"/>
      <stop offset="75%" style="stop-color:#4DA6FF"/>
      <stop offset="100%" style="stop-color:#9B59B6"/>
    </linearGradient>
    <linearGradient id="gB1" x1="0%" y1="100%" x2="0%" y2="0%">
      <stop offset="0%" style="stop-color:#E05C5C"/><stop offset="100%" style="stop-color:#FF8C42"/>
    </linearGradient>
    <linearGradient id="gB2" x1="0%" y1="100%" x2="0%" y2="0%">
      <stop offset="0%" style="stop-color:#F5C842"/><stop offset="100%" style="stop-color:#39D98A"/>
    </linearGradient>
    <linearGradient id="gB3" x1="0%" y1="100%" x2="0%" y2="0%">
      <stop offset="0%" style="stop-color:#39D98A"/><stop offset="100%" style="stop-color:#4DA6FF"/>
    </linearGradient>
  </defs>
  <!-- Rainbow ring -->
  <circle cx="90" cy="60" r="52" fill="none" stroke="url(#gRainbow)" stroke-width="7"/>
  <!-- Dark fill -->
  <circle cx="90" cy="60" r="44" fill="#060A10"/>
  <!-- Bars inside -->
  <rect x="60" y="52" width="9" height="22" rx="1.5" fill="url(#gB1)"/>
  <rect x="73" y="44" width="9" height="30" rx="1.5" fill="url(#gB2)"/>
  <rect x="86" y="36" width="9" height="38" rx="1.5" fill="url(#gB3)"/>
  <rect x="99" y="28" width="9" height="46" rx="1.5" fill="url(#gB3)"/>
  <!-- Arrow inside -->
  <path d="M62 60 Q90 30 122 20" fill="none" stroke="#4DA6FF" stroke-width="2.5" stroke-linecap="round"/>
  <polygon points="122,20 114,24 118,32" fill="#4DA6FF"/>
  <!-- MANSA text inside circle -->
  <text x="90" y="85" text-anchor="middle" font-family="sans-serif" font-size="14"
        font-weight="900" letter-spacing="3" fill="white">MANSA</text>
  <!-- Bottom text -->
  <text x="90" y="118" text-anchor="middle" font-family="sans-serif" font-size="8"
        letter-spacing="2" fill="#507060">GOLD INTELLIGENCE</text>
</svg>"""

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════
UNITS = {
    "Troy Ounce (oz t)":      {"factor": 1,            "symbol": "oz t"},
    "Gram (g)":               {"factor": 1/31.1035,    "symbol": "g"},
    "Kilogram (kg)":          {"factor": 1/0.0311035,  "symbol": "kg"},
    "Tola (11.66 g)":         {"factor": 1/2.6667,     "symbol": "tola"},
    "Baht Thailand (15.2 g)": {"factor": 1/2.04545,    "symbol": "baht"},
    "Pennyweight (dwt)":      {"factor": 20,            "symbol": "dwt"},
    "Grain":                  {"factor": 480,           "symbol": "gr"},
}

PURITIES = {
    "24K — 999.9":  {"mult":1.0000,"karat":24,"fine":"999.9","label":"24K"},
    "22K — 916":    {"mult":0.9167,"karat":22,"fine":"916",  "label":"22K"},
    "21K — 875":    {"mult":0.8750,"karat":21,"fine":"875",  "label":"21K"},
    "18K — 750":    {"mult":0.7500,"karat":18,"fine":"750",  "label":"18K"},
    "14K — 585":    {"mult":0.5833,"karat":14,"fine":"585",  "label":"14K"},
    "10K — 417":    {"mult":0.4167,"karat":10,"fine":"417",  "label":"10K"},
    "9K  — 375":    {"mult":0.3750,"karat": 9,"fine":"375",  "label":"9K"},
}

MARKETS = {
    "Jordan (JOD)":      {"flag":"🇯🇴","currency":"JOD","fx_ticker":"USDJOD=X","fx_approx":0.709,  "unit_label":"gram",   "unit_factor_from_oz":1/31.1035,"note":"سوق عمّان للذهب",        "note_en":"Amman Gold Market",         "fx_inverse":False},
    "Saudi Arabia (SAR)":{"flag":"🇸🇦","currency":"SAR","fx_ticker":"USDSAR=X","fx_approx":3.75,   "unit_label":"gram",   "unit_factor_from_oz":1/31.1035,"note":"سوق الذهب بالرياض",      "note_en":"Riyadh Gold Market",         "fx_inverse":False},
    "UAE (AED)":         {"flag":"🇦🇪","currency":"AED","fx_ticker":"USDAED=X","fx_approx":3.6725, "unit_label":"gram",   "unit_factor_from_oz":1/31.1035,"note":"سوق دبي للذهب",          "note_en":"Dubai Gold Souk",           "fx_inverse":False},
    "Egypt (EGP)":       {"flag":"🇪🇬","currency":"EGP","fx_ticker":"USDEGP=X","fx_approx":50.9,   "unit_label":"gram",   "unit_factor_from_oz":1/31.1035,"note":"سوق القاهرة",            "note_en":"Cairo Gold Market",          "fx_inverse":False},
    "Kuwait (KWD)":      {"flag":"🇰🇼","currency":"KWD","fx_ticker":"USDKWD=X","fx_approx":0.307,  "unit_label":"gram",   "unit_factor_from_oz":1/31.1035,"note":"سوق الكويت للذهب",       "note_en":"Kuwait Gold Market",         "fx_inverse":False},
    "Qatar (QAR)":       {"flag":"🇶🇦","currency":"QAR","fx_ticker":"USDQAR=X","fx_approx":3.64,   "unit_label":"gram",   "unit_factor_from_oz":1/31.1035,"note":"سوق الدوحة",             "note_en":"Doha Gold Market",           "fx_inverse":False},
    "Bahrain (BHD)":     {"flag":"🇧🇭","currency":"BHD","fx_ticker":"USDBHD=X","fx_approx":0.376,  "unit_label":"gram",   "unit_factor_from_oz":1/31.1035,"note":"سوق المنامة",            "note_en":"Manama Gold Market",         "fx_inverse":False},
    "Oman (OMR)":        {"flag":"🇴🇲","currency":"OMR","fx_ticker":"USDOMR=X","fx_approx":0.385,  "unit_label":"gram",   "unit_factor_from_oz":1/31.1035,"note":"سوق مسقط",               "note_en":"Muscat Gold Market",         "fx_inverse":False},
    "Iraq (IQD)":        {"flag":"🇮🇶","currency":"IQD","fx_ticker":"USDIQD=X","fx_approx":1310.0, "unit_label":"gram",   "unit_factor_from_oz":1/31.1035,"note":"سوق بغداد",              "note_en":"Baghdad Gold Market",        "fx_inverse":False},
    "Turkey (TRY)":      {"flag":"🇹🇷","currency":"TRY","fx_ticker":"USDTRY=X","fx_approx":38.0,   "unit_label":"gram",   "unit_factor_from_oz":1/31.1035,"note":"البازار الكبير - إسطنبول","note_en":"Grand Bazaar - Istanbul",    "fx_inverse":False},
    "USA (USD)":         {"flag":"🇺🇸","currency":"USD","fx_ticker":None,      "fx_approx":1.0,    "unit_label":"troy oz","unit_factor_from_oz":1.0,       "note":"COMEX Spot",             "note_en":"COMEX Spot",                "fx_inverse":False},
    "UK (GBP)":          {"flag":"🇬🇧","currency":"GBP","fx_ticker":"GBPUSD=X","fx_approx":1.27,   "unit_label":"troy oz","unit_factor_from_oz":1.0,       "note":"London Bullion Market",  "note_en":"London Bullion Market",     "fx_inverse":True},
    "EU (EUR)":          {"flag":"🇪🇺","currency":"EUR","fx_ticker":"EURUSD=X","fx_approx":1.08,   "unit_label":"troy oz","unit_factor_from_oz":1.0,       "note":"Frankfurt / Paris",      "note_en":"Frankfurt / Paris",         "fx_inverse":True},
    "India (INR)":       {"flag":"🇮🇳","currency":"INR","fx_ticker":"USDINR=X","fx_approx":84.5,   "unit_label":"10g",    "unit_factor_from_oz":10/31.1035,"note":"MCX Mumbai",             "note_en":"MCX Mumbai",                "fx_inverse":False},
    "China (CNY)":       {"flag":"🇨🇳","currency":"CNY","fx_ticker":"USDCNY=X","fx_approx":7.27,   "unit_label":"gram",   "unit_factor_from_oz":1/31.1035,"note":"SGE Shanghai",           "note_en":"SGE Shanghai",              "fx_inverse":False},
    "Lebanon (USD)":     {"flag":"🇱🇧","currency":"USD","fx_ticker":None,      "fx_approx":1.0,    "unit_label":"gram",   "unit_factor_from_oz":1/31.1035,"note":"بيروت (USD)",            "note_en":"Beirut (USD)",              "fx_inverse":False},
}

STOCK_OPTIONS = {
    "S&P 500":"^GSPC","NASDAQ":"^IXIC","Dow Jones":"^DJI",
    "Apple":"AAPL","Microsoft":"MSFT","Amazon":"AMZN","Google":"GOOGL",
    "Tesla":"TSLA","NVIDIA":"NVDA","Meta":"META",
    "Aramco":"2222.SR","Bitcoin":"BTC-USD","Ethereum":"ETH-USD",
}

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS  — all tuneable parameters in one place
# ═══════════════════════════════════════════════════════════════════════════════

#: Directory containing serialised ML model files (produced by train_models.py)
MODELS_DIR: str = "models"

#: LSTM input sequence length (number of timesteps)
TIME_STEP: int = 10

#: Live-price cache TTL — slightly under 60 s so cache rarely serves stale zeros
CACHE_TTL_PRICES: int = 55

#: General history/analytics cache TTL
CACHE_TTL_HISTORY: int = 300

#: Google Drive download cache TTL (10 min — infrequent, large payload)
CACHE_TTL_DRIVE: int = 600

#: Score/sentiment analytics cache TTL (15 min)
CACHE_TTL_SCORE: int = 900

#: Auto-refresh countdown (seconds)
REFRESH_INTERVAL: int = 60

MODEL_FILES = {
    "Linear Regression": ("model1_lr.pkl",     "sklearn"),
    "Random Forest":     ("model2_rf.pkl",      "sklearn"),
    "Gradient Boosting": ("model3_gb.pkl",      "sklearn"),
    "XGBoost":           ("model4_xgb.pkl",     "sklearn"),
    "LSTM":              ("model5_lstm.keras",  "keras"),
    "Prophet":           ("model6_prophet.pkl", "prophet"),
}
FEATURE_COLS = [
    "SPX_Close","CPI","EFFR","USD_Index","Oil_Price","Silver_Price",
    "Real_Interest_Rate","VIX","US10Y_Yield",
    "Gold_VIX_Ratio","Gold_US10Y_Ratio","Gold_Oil_Ratio","Gold_Silver_Ratio",
    "Gold_SPX_Ratio","Gold_DXY_Ratio","Gold_CPI_Ratio","Gold_EFFR_Ratio",
    "Gold_RealRate_Ratio",
]

# ═══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════════
DEFAULTS = {
    "theme":             list(THEMES.keys())[0],   # Islamic & Arab default
    "lang":              "العربية 🇸🇦",
    "unit":              "Gram (g)",
    "purity":            "21K — 875",
    "primary_mkt":       "Jordan (JOD)",
    "active_mkts":       ["Jordan (JOD)","Saudi Arabia (SAR)","UAE (AED)","Egypt (EGP)",
                          "Kuwait (KWD)","Qatar (QAR)","Bahrain (BHD)","Turkey (TRY)",
                          "USA (USD)","UK (GBP)","EU (EUR)","India (INR)"],
    "period":            "1y",
    "auto_refresh":      True,   # ON by default — JS handles refresh without sleep loop
    "show_purity_table": True,
    "active_stocks":     ["S&P 500","Bitcoin","NVIDIA","Aramco"],
    "nav":               None,
    "advisor_profile":   None,
    # Portfolio tracker
    "portfolio_entries": [],      # list of {qty, unit, buy_price, currency, date, label}
    # Calculator
    "calc_history":      [],
    # Price alerts
    "price_alerts":      [],   # [{id, price, direction, label, triggered}]
    # Trade journal
    "trade_journal":     [],   # [{id, date, direction, entry, exit, qty, unit, pnl, note, ai_rec}]
    # User profile
    "user_name":         "",
    "user_risk":         "Medium",
    "persona":           "trader",   # trader|investor|shop|factory|designer|buyer
    "shop_name":         "",
    "shop_spread":       2.0,
    "shop_currency":     "USD",
    "invoices":          [],
    "saved_designs":     [],
    # Demo trading account
    "demo_balance":      10000.0,    # virtual USD starting balance
    "demo_holdings_g":   0.0,        # virtual gold in grams
    "demo_trades":       [],         # [{id, type, qty_g, price_usd, ts, pnl}]
    "demo_total_bought": 0.0,        # cumulative cost basis
    # Gold savings plans
    "savings_plans":     [],         # [{id, monthly, currency, start, label}]
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ═══════════════════════════════════════════════════════════════════════════════
# SUPABASE PERSISTENCE LAYER
# ───────────────────────────────────────────────────────────────────────────────
# Set these in .streamlit/secrets.toml to enable cross-session persistence:
#
#   [secrets]
#   SUPABASE_URL  = "https://xxxx.supabase.co"
#   SUPABASE_KEY  = "eyJ..."   # anon/public key
#
# Required Supabase table (run once in the SQL Editor):
#
#   create table if not exists mansa_user_data (
#     user_id      text primary key,
#     portfolio    jsonb default '[]'::jsonb,
#     alerts       jsonb default '[]'::jsonb,
#     journal      jsonb default '[]'::jsonb,
#     profile      jsonb default '{}'::jsonb,
#     updated_at   timestamptz default now()
#   );
#   alter table mansa_user_data enable row level security;
#   create policy "Users can manage own data"
#     on mansa_user_data for all
#     using  (user_id = current_user)
#     with check (user_id = current_user);
#
# Without Supabase credentials everything works exactly as before — all data
# lives in session_state only (lost on browser close).
# ═══════════════════════════════════════════════════════════════════════════════

def _get_sb_config() -> tuple:
    """Return (supabase_url, supabase_key) from secrets or env vars, or (None, None)."""
    try:
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_KEY"]
        if url and key:
            return url, key
    except Exception:
        # st.secrets raises KeyError when the key is absent — not an error
        _log.debug("SUPABASE_URL/KEY not in secrets, trying env vars")
    url = os.environ.get("SUPABASE_URL", "")
    key = os.environ.get("SUPABASE_KEY", "")
    return (url, key) if (url and key) else (None, None)


def _sb_user_id() -> str:
    """Return a stable anonymous user ID stored in session_state.

    Uses the Streamlit session ID when available, otherwise generates a
    random UUID on first call and persists it for the session lifetime.
    """
    if "sb_user_id" not in st.session_state:
        import hashlib, random
        # Stable within a session; anonymous — no PII stored
        seed = str(random.getrandbits(128))
        st.session_state["sb_user_id"] = "mansa_" + hashlib.md5(seed.encode()).hexdigest()[:16]
    return st.session_state["sb_user_id"]


def _sb_load() -> None:
    """Load persisted user data from Supabase into session_state.

    Called once per session on first render.  Silently no-ops when
    Supabase credentials are not configured.
    """
    if st.session_state.get("_sb_loaded"):
        return
    st.session_state["_sb_loaded"] = True

    url, key = _get_sb_config()
    if not url:
        return  # No credentials — use session_state only

    user_id = _sb_user_id()
    try:
        import urllib.request as _sbr, json as _sbj
        endpoint = f"{url}/rest/v1/mansa_user_data?user_id=eq.{user_id}&select=*"
        req = _sbr.Request(
            endpoint,
            headers={
                "apikey":        key,
                "Authorization": f"Bearer {key}",
                "Content-Type":  "application/json",
            },
        )
        with _sbr.urlopen(req, timeout=8) as resp:
            rows = _sbj.loads(resp.read())

        if rows:
            row = rows[0]
            # Only overwrite if session_state is still at defaults (fresh session)
            if not st.session_state.get("portfolio_entries"):
                st.session_state["portfolio_entries"] = row.get("portfolio", [])
            if not st.session_state.get("price_alerts"):
                st.session_state["price_alerts"] = row.get("alerts", [])
            if not st.session_state.get("trade_journal"):
                st.session_state["trade_journal"] = row.get("journal", [])
            profile = row.get("profile", {})
            if profile.get("user_name") and not st.session_state.get("user_name"):
                st.session_state["user_name"] = profile["user_name"]
            if profile.get("user_risk"):
                st.session_state["user_risk"] = profile["user_risk"]
            _log.debug("Supabase: loaded user data for %s", user_id)
    except Exception:
        _log.warning("Supabase load failed — using session_state only", exc_info=True)


def _sb_save() -> None:
    """Upsert current session_state data to Supabase.

    Called automatically after any mutation to portfolio, alerts or journal.
    Silently no-ops when Supabase credentials are not configured.
    """
    url, key = _get_sb_config()
    if not url:
        return

    user_id = _sb_user_id()
    payload = {
        "user_id":    user_id,
        "portfolio":  st.session_state.get("portfolio_entries", []),
        "alerts":     st.session_state.get("price_alerts", []),
        "journal":    st.session_state.get("trade_journal", []),
        "profile": {
            "user_name": st.session_state.get("user_name", ""),
            "user_risk": st.session_state.get("user_risk", "Medium"),
        },
        "updated_at": datetime.datetime.utcnow().isoformat() + "Z",
    }
    try:
        import urllib.request as _sbw, json as _sbj2
        body = _sbj2.dumps(payload).encode()
        endpoint = f"{url}/rest/v1/mansa_user_data"
        req = _sbw.Request(
            endpoint,
            data=body,
            method="POST",
            headers={
                "apikey":        key,
                "Authorization": f"Bearer {key}",
                "Content-Type":  "application/json",
                "Prefer":        "resolution=merge-duplicates",  # upsert
            },
        )
        with _sbw.urlopen(req, timeout=8) as resp:
            _ = resp.read()
        _log.debug("Supabase: saved user data for %s", user_id)
    except Exception:
        _log.warning("Supabase save failed", exc_info=True)


# ── Load persisted data on first render ───────────────────────────────────────
_sb_load()

# ── Language shortcut ─────────────────────────────────────────────────────────
L  = LANGS.get(st.session_state["lang"], LANGS["العربية 🇸🇦"])

def is_rtl() -> bool:
    """Returns True when the active language uses right-to-left script (Arabic / Urdu)."""
    return st.session_state["lang"].startswith("ال") or st.session_state["lang"].startswith("اردو")

def t(key: str, fallback_ar: str = "", fallback_en: str = "") -> str:
    """Shortcut: fetch translation key from active LANGS dict, with optional fallback."""
    return L.get(key, fallback_ar if is_rtl() else fallback_en)
if st.session_state["nav"] is None:
    st.session_state["nav"] = L["nav_dashboard"]
if st.session_state["advisor_profile"] is None:
    st.session_state["advisor_profile"] = L["profile_opts"][0]

# ── Active theme colours ──────────────────────────────────────────────────────
C  = THEMES[st.session_state["theme"]]

# ═══════════════════════════════════════════════════════════════════════════════
# CSS  — rebuilt on every run so theme change is instant
# ═══════════════════════════════════════════════════════════════════════════════
def build_css(C):
    """Generate and inject the full CSS stylesheet for the active theme.
    
        Parameters
        ----------
        C : dict
            Active theme colour/font dictionary from ``THEMES``.
        """
    is_arabic = C["font_h"] == "Amiri"
    if is_arabic:
        fi = "@import url('https://fonts.googleapis.com/css2?family=Amiri:wght@400;700&family=Cairo:wght@400;600;700;900&family=JetBrains+Mono:wght@300;400;500&display=swap');"
    elif C["font_h"] == "Share Tech Mono":
        fi = "@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=IBM+Plex+Sans:wght@300;400;600&display=swap');"
    else:
        fi = "@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;600;700;900&family=Cormorant+Garamond:ital,wght@0,300;0,400;0,600;1,300;1,400&family=JetBrains+Mono:wght@300;400;500&display=swap');"

    return f"""
<style>
{fi}
*, *::before, *::after {{ box-sizing:border-box; margin:0; padding:0; }}
html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"], .main .block-container {{
    background:{C['bg']} !important; color:{C['text']};
    font-family:'{C['font_b']}','Cairo',Georgia,sans-serif;
    font-size:{'15px' if is_arabic else '14px'};
}}
.main .block-container {{ padding:1.5rem 2rem 3rem !important; max-width:1600px !important; }}
[data-testid="stSidebar"] {{ background:{C['bg2']} !important; border-right:1px solid {C['border2']}; }}
[data-testid="stSidebar"] * {{ color:{C['text']} !important; }}
.mansa-divider {{
    height:1px;
    background:linear-gradient(90deg,transparent,{C['gold']}66 30%,{C['gold']} 50%,{C['gold']}66 70%,transparent);
    margin:1.5rem 0;
}}
.section-label {{
    font-family:'{C['font_h']}','Cairo',serif; font-size:{'13px' if is_arabic else '9px'}; font-weight:700;
    letter-spacing:{'0' if is_arabic else '.3em'}; text-transform:{'none' if is_arabic else 'uppercase'}; color:{C['gold']};
    margin:1.4rem 0 1rem; padding-bottom:.5rem; border-bottom:1px solid {C['border2']};
}}
.hero-wrap {{
    background:linear-gradient(160deg,{C['card2']} 0%,{C['card']} 60%,{C['bg']} 100%);
    border:1px solid {C['border2']}; border-radius:6px; padding:2rem 2.5rem;
    position:relative; overflow:hidden;
}}
.hero-wrap::after {{
    content:""; position:absolute; top:0; left:0; right:0; height:2px;
    background:linear-gradient(90deg,transparent,{C['gold']},{C['gold_hi']},{C['gold']},transparent);
}}
/* hero / stat-card / ticker base styles defined in full below (with animations) */
/* ── Keyframe animations ──────────────────────────────────────────────────── */
@keyframes pulse        {{ 0%,100%{{opacity:1}} 50%{{opacity:.5}} }}
@keyframes fadeInUp     {{ from{{opacity:0;transform:translateY(18px)}} to{{opacity:1;transform:translateY(0)}} }}
@keyframes fadeInLeft   {{ from{{opacity:0;transform:translateX(-18px)}} to{{opacity:1;transform:translateX(0)}} }}
@keyframes fadeInRight  {{ from{{opacity:0;transform:translateX(18px)}} to{{opacity:1;transform:translateX(0)}} }}
@keyframes scaleIn      {{ from{{opacity:0;transform:scale(.94)}} to{{opacity:1;transform:scale(1)}} }}
@keyframes shimmer      {{
  0%   {{background-position:-200% 0}}
  100% {{background-position: 200% 0}}
}}
@keyframes goldPulse    {{
  0%,100% {{box-shadow:0 0 0 0 {C['gold']}44}}
  50%     {{box-shadow:0 0 12px 4px {C['gold']}22}}
}}
@keyframes slideDown    {{
  from {{opacity:0; max-height:0; transform:translateY(-8px)}}
  to   {{opacity:1; max-height:1000px; transform:translateY(0)}}
}}
@keyframes borderGlow   {{
  0%,100% {{border-color:{C['border2']}}}
  50%     {{border-color:{C['gold']}88}}
}}
@keyframes countUp      {{
  from {{opacity:0; transform:translateY(6px)}}
  to   {{opacity:1; transform:translateY(0)}}
}}
@keyframes spinGold     {{
  from {{transform:rotate(0deg)}}
  to   {{transform:rotate(360deg)}}
}}

/* ── Global transition defaults ──────────────────────────────────────────── */
*, *::before, *::after {{
  transition-property: background-color, border-color, color, opacity, transform, box-shadow;
  transition-duration: 0.22s;
  transition-timing-function: cubic-bezier(.4,0,.2,1);
}}

/* ── Page entry animation ─────────────────────────────────────────────────── */
.main .block-container {{
  animation: fadeInUp 0.35s cubic-bezier(.4,0,.2,1) both;
}}

/* ── Stat cards ───────────────────────────────────────────────────────────── */
.stat-card {{
    background:{C['card']}; border:1px solid {C['border']}; border-radius:4px;
    padding:14px 16px; position:relative; overflow:hidden; margin-bottom:6px;
    animation: scaleIn 0.3s cubic-bezier(.4,0,.2,1) both;
    transition: transform .22s cubic-bezier(.4,0,.2,1),
                box-shadow .22s cubic-bezier(.4,0,.2,1),
                border-color .22s cubic-bezier(.4,0,.2,1),
                background .22s !important;
}}
.stat-card:hover {{
    transform: translateY(-3px);
    box-shadow: 0 8px 24px {C['gold']}18;
    border-color: {C['gold']}55;
}}
.stat-card::after {{
    content:""; position:absolute; bottom:0; left:0; right:0; height:1px;
    background:linear-gradient(90deg,transparent,{C['gold']}33,transparent);
}}

/* ── Hero section ─────────────────────────────────────────────────────────── */
.hero-wrap {{
    background:linear-gradient(160deg,{C['card2']} 0%,{C['card']} 60%,{C['bg']} 100%);
    border:1px solid {C['border2']}; border-radius:6px; padding:2rem 2.5rem;
    position:relative; overflow:hidden;
    animation: fadeInLeft 0.4s cubic-bezier(.4,0,.2,1) both;
    transition: box-shadow .3s, border-color .3s !important;
}}
.hero-wrap:hover {{
    box-shadow: 0 0 32px {C['gold']}22;
    border-color: {C['gold']}44;
}}
.hero-wrap::after {{
    content:""; position:absolute; top:0; left:0; right:0; height:2px;
    background:linear-gradient(90deg,transparent,{C['gold']},{C['gold_hi']},{C['gold']},transparent);
    background-size:200% 100%;
    animation: shimmer 3s linear infinite;
}}
.hero-price {{
    font-family:'{C['font_h']}','Cairo',serif; font-size:52px; font-weight:900;
    color:{C['gold_pale']}; line-height:1.1;
    animation: countUp 0.5s cubic-bezier(.4,0,.2,1) both;
}}
.hero-unit  {{ font-family:'{C['font_b']}',serif; font-size:18px; font-style:italic; color:{C['muted']}; margin-left:8px; }}
.hero-change {{
    font-family:'{C['font_m']}',monospace; font-size:16px; font-weight:500; margin-top:12px;
    animation: fadeInUp 0.6s cubic-bezier(.4,0,.2,1) both;
    animation-delay: 0.1s;
}}
.hero-meta   {{ font-family:'{C['font_h']}',serif; font-size:8px; letter-spacing:.3em; color:{C['dim']}; margin-top:10px; text-transform:uppercase; }}

/* ── Section labels ───────────────────────────────────────────────────────── */
.section-label {{
    font-family:'{C['font_h']}','Cairo',serif; font-size:{'13px' if is_arabic else '9px'}; font-weight:700;
    letter-spacing:{'0' if is_arabic else '.3em'}; text-transform:{'none' if is_arabic else 'uppercase'};
    color:{C['gold']}; margin:1.4rem 0 1rem; padding-bottom:.5rem;
    border-bottom:1px solid {C['border2']};
    animation: fadeInLeft 0.3s cubic-bezier(.4,0,.2,1) both;
}}

/* ── Live badge ───────────────────────────────────────────────────────────── */
.live-badge {{
    display:inline-flex; align-items:center; gap:6px;
    background:{C['green']}18; border:1px solid {C['green']}44; border-radius:20px;
    padding:4px 12px;
    animation: scaleIn 0.3s cubic-bezier(.4,0,.2,1) both;
    transition: background .22s, border-color .22s !important;
}}
.live-badge:hover {{
    background:{C['green']}28;
    border-color:{C['green']}88;
}}
.live-dot   {{
    width:7px; height:7px; border-radius:50%; background:{C['green']};
    animation: pulse 2s infinite;
}}
.live-text  {{ font-family:'{C['font_h']}','Cairo',serif; font-size:{'12px' if is_arabic else '8px'}; letter-spacing:{'0' if is_arabic else '.2em'}; color:{C['green']}; text-transform:{'none' if is_arabic else 'uppercase'}; }}

/* ── Market cards ─────────────────────────────────────────────────────────── */
.mkt-card {{
    background:{C['card2']}; border:1px solid {C['border2']}; border-radius:5px;
    padding:16px 18px; margin-bottom:8px; position:relative; overflow:hidden;
    transition: transform .22s cubic-bezier(.4,0,.2,1),
                box-shadow .22s,
                border-color .22s !important;
    animation: fadeInUp 0.3s cubic-bezier(.4,0,.2,1) both;
}}
.mkt-card:hover {{
    transform: translateY(-2px) scale(1.005);
    box-shadow: 0 6px 20px {C['gold']}14;
    border-color: {C['gold']}44;
}}
.mkt-card::after {{
    content:""; position:absolute; top:0; left:0; right:0; height:1px;
    background:linear-gradient(90deg,transparent,{C['gold']}55,transparent);
    transition: opacity .22s !important;
}}
.mkt-card:hover::after {{ opacity:1.5; }}
.mkt-name  {{ font-family:'{C['font_h']}','Cairo',serif; font-size:{'13px' if is_arabic else '9px'}; letter-spacing:{'0' if is_arabic else '.2em'}; color:{C['gold']}; text-transform:{'none' if is_arabic else 'uppercase'}; margin-bottom:6px; transition:color .22s !important; }}
.mkt-price {{ font-family:'{C['font_m']}',monospace; font-size:20px; color:{C['gold_pale']}; transition:color .22s !important; }}
.mkt-unit  {{ font-family:'{C['font_b']}',serif; font-size:11px; font-style:italic; color:{C['muted']}; margin-left:4px; }}
.mkt-chg   {{ font-family:'{C['font_m']}',monospace; font-size:11px; margin-top:4px; transition:color .22s !important; }}

/* ── Purity badge ─────────────────────────────────────────────────────────── */
.purity-badge {{
    display:inline-block; background:{C['gold_dark']}44; border:1px solid {C['gold']}66;
    border-radius:3px; padding:2px 8px; font-family:'{C['font_h']}',serif;
    font-size:8px; letter-spacing:.15em; color:{C['gold']}; margin-right:4px;
    transition: background .22s, border-color .22s, transform .18s !important;
}}
.purity-badge:hover {{
    background:{C['gold_dark']}66;
    border-color:{C['gold']}aa;
    transform:scale(1.05);
}}

/* ── Prediction cards ─────────────────────────────────────────────────────── */
.pred-card {{
    background:{C['card2']}; border:1px solid {C['border2']}; border-radius:6px;
    padding:16px 20px; margin-bottom:10px; position:relative; overflow:hidden;
    animation: slideDown 0.35s cubic-bezier(.4,0,.2,1) both;
    transition: transform .22s, box-shadow .22s, border-color .22s !important;
}}
.pred-card:hover {{
    transform: translateX(4px);
    box-shadow: 4px 0 16px {C['gold']}14;
    border-color: {C['gold']}44;
}}
.pred-card::before {{
    content:""; position:absolute; left:0; top:0; bottom:0; width:3px;
    background:linear-gradient(180deg,{C['gold']},{C['gold_dark']});
    transition: width .22s !important;
}}
.pred-card:hover::before {{ width:5px; }}
.pred-best::before {{ background:linear-gradient(180deg,{C['gold_hi']},{C['gold']}) !important; }}
.pred-algo  {{ font-family:'{C['font_h']}','Cairo',serif; font-size:{'12px' if is_arabic else '8px'}; letter-spacing:{'0' if is_arabic else '.25em'}; color:{C['muted']}; text-transform:{'none' if is_arabic else 'uppercase'}; margin-bottom:4px; }}
.pred-price {{ font-family:'{C['font_m']}',monospace; font-size:26px; color:{C['gold_hi']}; font-weight:700; transition:color .22s !important; }}
.pred-diff  {{ font-family:'{C['font_m']}',monospace; font-size:12px; margin-top:4px; }}
.pred-r2    {{ font-family:'{C['font_h']}',serif; font-size:8px; letter-spacing:.2em; color:{C['dim']}; margin-top:6px; }}

/* ── Ticker cards ─────────────────────────────────────────────────────────── */
.ticker-card {{
    background:{C['card']}; border:1px solid {C['border']}; border-radius:4px;
    padding:10px 12px; text-align:center;
    transition: transform .22s, box-shadow .22s, border-color .22s !important;
    animation: fadeInUp 0.3s cubic-bezier(.4,0,.2,1) both;
}}
.ticker-card:hover {{
    transform:translateY(-2px);
    box-shadow:0 4px 14px {C['gold']}14;
    border-color:{C['gold']}44;
}}
.ticker-name  {{ font-family:'{C['font_h']}','Cairo',serif; font-size:{'12px' if is_arabic else '8px'}; letter-spacing:{'0' if is_arabic else '.2em'}; color:{C['muted']}; text-transform:{'none' if is_arabic else 'uppercase'}; transition:color .22s !important; }}
.ticker-price {{ font-family:'{C['font_m']}',monospace; font-size:13px; color:{C['gold_pale']}; margin:3px 0; transition:color .22s !important; }}
.ticker-chg   {{ font-family:'{C['font_m']}',monospace; font-size:11px; transition:color .22s !important; }}

/* ── Settings cards ───────────────────────────────────────────────────────── */
.settings-card {{
    background:{C['card2']}; border:1px solid {C['border2']}; border-radius:5px;
    padding:20px 22px; margin-bottom:12px;
    transition: border-color .22s, box-shadow .22s !important;
    animation: fadeInUp 0.35s cubic-bezier(.4,0,.2,1) both;
}}
.settings-card:hover {{
    border-color:{C['gold']}44;
    box-shadow:0 2px 12px {C['gold']}0f;
}}
.settings-title {{
    font-family:'{C['font_h']}',serif; font-size:10px; font-weight:700;
    letter-spacing:.25em; color:{C['gold']}; text-transform:uppercase;
    margin-bottom:14px; padding-bottom:8px; border-bottom:1px solid {C['border']};
}}

/* ── Buttons ──────────────────────────────────────────────────────────────── */
.stButton > button {{
    font-family:'{C['font_h']}',serif !important; font-size:10px !important;
    letter-spacing:.2em !important; text-transform:uppercase !important;
    background:linear-gradient(135deg,{C['gold_dark']},{C['gold']}) !important;
    color:#050400 !important; border:none !important; border-radius:3px !important;
    padding:10px 28px !important;
    transition: opacity .2s, transform .18s, box-shadow .2s !important;
    position:relative; overflow:hidden;
}}
.stButton > button::after {{
    content:''; position:absolute; top:0; left:-100%; width:100%; height:100%;
    background:linear-gradient(90deg,transparent,rgba(255,255,255,.15),transparent);
    transition:left .4s !important;
}}
.stButton > button:hover {{
    opacity:.9 !important;
    transform:translateY(-1px) !important;
    box-shadow:0 4px 14px {C['gold']}33 !important;
}}
.stButton > button:hover::after {{ left:100%; }}
.stButton > button:active {{ transform:translateY(0) !important; opacity:.8 !important; }}

/* ── Sidebar nav items ────────────────────────────────────────────────────── */
[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] button {{
    transition: color .18s, background .18s, padding-left .18s !important;
}}
[data-testid="stSidebar"] .stRadio label:hover {{
    color:{C['gold']} !important;
    padding-left:6px;
}}

/* ── Inputs ───────────────────────────────────────────────────────────────── */
.stTextInput input, .stNumberInput input, .stTextArea textarea {{
    transition: border-color .22s, box-shadow .22s, background .22s !important;
    border-radius:4px !important;
}}
.stTextInput input:focus, .stNumberInput input:focus, .stTextArea textarea:focus {{
    border-color:{C['gold']}88 !important;
    box-shadow:0 0 0 2px {C['gold']}22 !important;
}}

/* ── Select / Slider ──────────────────────────────────────────────────────── */
.stSelectbox > div, .stSlider > div {{
    transition:opacity .22s !important;
}}
.stSlider [data-baseweb="slider"] {{
    transition:filter .22s !important;
}}
[data-baseweb="slider"]:hover {{
    filter: brightness(1.1);
}}
/* Gold track color for sliders */
[data-testid="stSlider"] [role="slider"] {{
    background:{C['gold']} !important;
    transition:transform .18s, box-shadow .18s !important;
}}
[data-testid="stSlider"] [role="slider"]:hover {{
    transform:scale(1.25) !important;
    box-shadow:0 0 0 4px {C['gold']}33 !important;
}}

/* ── Expanders ────────────────────────────────────────────────────────────── */
[data-testid="stExpander"] {{
    transition:border-color .22s, box-shadow .22s !important;
    border-radius:6px !important;
    overflow:hidden;
}}
[data-testid="stExpander"]:hover {{
    border-color:{C['gold']}44 !important;
    box-shadow:0 2px 10px {C['gold']}0f !important;
}}
[data-testid="stExpander"] > div:first-child {{
    transition:background .22s !important;
}}
[data-testid="stExpander"] > div:first-child:hover {{
    background:{C['gold']}0a !important;
}}

/* ── Tabs ─────────────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab"] {{
    transition:color .2s, border-color .2s, background .2s !important;
}}
.stTabs [data-baseweb="tab"]:hover {{
    color:{C['gold']} !important;
    background:{C['gold']}0a !important;
}}
.stTabs [aria-selected="true"] {{
    color:{C['gold_hi']} !important;
    border-bottom:2px solid {C['gold']} !important;
}}
.stTabs [data-baseweb="tab-list"] {{
    border-bottom:1px solid {C['border2']} !important;
    gap:4px !important;
}}

/* ── Dataframes ───────────────────────────────────────────────────────────── */
[data-testid="stDataFrame"] {{
    animation: fadeInUp 0.35s cubic-bezier(.4,0,.2,1) both;
    border-radius:6px !important; overflow:hidden;
}}
[data-testid="stDataFrame"] tbody tr {{
    transition:background .18s !important;
}}
[data-testid="stDataFrame"] tbody tr:hover {{
    background:{C['gold']}0a !important;
}}

/* ── Alerts / messages ────────────────────────────────────────────────────── */
[data-testid="stAlert"] {{
    animation: slideDown 0.3s cubic-bezier(.4,0,.2,1) both;
    transition: opacity .22s !important;
}}

/* ── Charts ───────────────────────────────────────────────────────────────── */
[data-testid="stVegaLiteChart"],
[data-testid="stPlotlyChart"] {{
    animation: scaleIn 0.4s cubic-bezier(.4,0,.2,1) both;
}}

/* ── Plotly chart background ──────────────────────────────────────────────── */
.js-plotly-plot .plotly {{
    border-radius:6px;
}}

/* ── Spinner ──────────────────────────────────────────────────────────────── */
[data-testid="stSpinner"] > div {{
    border-top-color:{C['gold']} !important;
    animation:spinGold 0.7s linear infinite;
}}

/* ── Sidebar ──────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {{
    background:{C['bg2']} !important;
    border-right:1px solid {C['border2']};
    transition:width .3s cubic-bezier(.4,0,.2,1) !important;
}}
[data-testid="stSidebar"] * {{ color:{C['text']} !important; }}

/* ── Page nav buttons in sidebar ─────────────────────────────────────────── */
[data-testid="stSidebar"] [data-testid="stRadio"] label {{
    padding:4px 8px; border-radius:4px;
    transition:background .18s, color .18s, transform .18s !important;
    display:block;
}}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {{
    background:{C['gold']}14;
    color:{C['gold']} !important;
    transform:translateX(4px);
}}

/* ── Toggle switches ──────────────────────────────────────────────────────── */
[data-testid="stToggle"] [role="switch"] {{
    transition:background .22s !important;
}}
[data-testid="stToggle"] [role="switch"][aria-checked="true"] {{
    background:{C['gold']} !important;
}}

/* ── Columns fade-in stagger ──────────────────────────────────────────────── */
[data-testid="column"]:nth-child(1) {{ animation:fadeInUp .3s ease both; }}
[data-testid="column"]:nth-child(2) {{ animation:fadeInUp .3s ease .05s both; }}
[data-testid="column"]:nth-child(3) {{ animation:fadeInUp .3s ease .10s both; }}
[data-testid="column"]:nth-child(4) {{ animation:fadeInUp .3s ease .15s both; }}
[data-testid="column"]:nth-child(5) {{ animation:fadeInUp .3s ease .20s both; }}
[data-testid="column"]:nth-child(6) {{ animation:fadeInUp .3s ease .25s both; }}

/* ── Divider ──────────────────────────────────────────────────────────────── */
.mansa-divider {{
    height:1px;
    background:linear-gradient(90deg,transparent,{C['gold']}66 30%,{C['gold']} 50%,{C['gold']}66 70%,transparent);
    margin:1.5rem 0;
    animation:shimmer 4s linear infinite;
    background-size:200% 100%;
}}

/* ── Stat label / value ──────────────────────────────────────────────────── */
.stat-label {{
    font-family:'{C['font_h']}','Cairo',serif; font-size:{'12px' if is_arabic else '8px'};
    letter-spacing:{'0' if is_arabic else '.2em'};
    color:{C['muted']}; text-transform:{'none' if is_arabic else 'uppercase'};
    margin-bottom:6px; transition:color .22s !important;
}}
.stat-value {{
    font-family:'{C['font_m']}',monospace; font-size:18px; color:{C['gold_pale']};
    line-height:1.1; transition:color .22s !important;
}}

/* ── Radio buttons ────────────────────────────────────────────────────────── */
.stRadio > div {{ gap:6px; }}
.stRadio label {{
    transition:color .18s, background .18s !important;
    border-radius:3px; padding:2px 6px;
}}
.stRadio label:hover {{ color:{C['gold']} !important; }}

/* ── Number input arrows ──────────────────────────────────────────────────── */
.stNumberInput button {{
    transition:background .18s, color .18s !important;
}}
.stNumberInput button:hover {{
    background:{C['gold']}22 !important;
    color:{C['gold']} !important;
}}

/* ── Respect prefers-reduced-motion ──────────────────────────────────────── */
@media (prefers-reduced-motion: reduce) {{
  *, *::before, *::after {{
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important;
  }}
}}
</style>"""

st.markdown(build_css(C), unsafe_allow_html=True)

# ── PWA: Progressive Web App support ─────────────────────────────────────────
# Allows "Add to Home Screen" on mobile — installs like a native app
_pwa_theme = C["bg"]
_pwa_gold  = C["gold"]
st.markdown(f"""
<link rel="manifest" href="data:application/json,{{
  %22name%22:%22MANSA+Gold+Intelligence%22,
  %22short_name%22:%22MANSA%22,
  %22description%22:%22Professional+Gold+Trading+Intelligence+Platform%22,
  %22start_url%22:%22/%22,
  %22display%22:%22standalone%22,
  %22background_color%22:%22{_pwa_theme}%22,
  %22theme_color%22:%22{_pwa_gold}%22,
  %22orientation%22:%22any%22,
  %22icons%22:[
    {{%22src%22:%22data:image/svg+xml,%3Csvg+xmlns%3D'http%3A//www.w3.org/2000/svg'+viewBox%3D'0+0+192+192'%3E%3Crect+width%3D'192'+height%3D'192'+fill%3D'%23{_pwa_theme.replace("#","")}'/%3E%3Ctext+x%3D'96'+y%3D'130'+text-anchor%3D'middle'+font-size%3D'120'+fill%3D'%23{_pwa_gold.replace("#","")}'%3E%E2%98%BD%3C/text%3E%3C/svg%3E%22,
      %22sizes%22:%22192x192%22,%22type%22:%22image/svg+xml%22}}
  ]
}}">
<meta name="mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="apple-mobile-web-app-title" content="MANSA Gold">
<meta name="theme-color" content="{_pwa_gold}">
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
<style>
/* ── Mobile optimizations ────────────────────────────────────────────────── */
@media (max-width: 768px) {{
  .main .block-container {{ padding:0.8rem 0.8rem 2rem !important; }}
  .hero-price {{ font-size:36px !important; }}
  .hero-unit  {{ font-size:13px !important; }}
  .stat-card  {{ padding:10px 12px !important; }}
  .mkt-card   {{ padding:12px 14px !important; }}
  /* Larger tap targets */
  .stButton > button {{ padding:12px 20px !important; min-height:44px !important; }}
  .stRadio label      {{ min-height:40px !important; display:flex !important; align-items:center !important; }}
  /* Stack columns on small screens */
  [data-testid="column"] {{ min-width:140px !important; }}
  /* Readable font sizes */
  .stat-value  {{ font-size:15px !important; }}
  .section-label {{ font-size:12px !important; }}
}}
@media (max-width: 480px) {{
  .hero-price {{ font-size:28px !important; }}
  [data-testid="column"] {{ min-width:100px !important; }}
}}
/* ── Install prompt button ─────────────────────────────────────────────────── */
#pwa-install-btn {{
  display:none; position:fixed; bottom:20px; right:20px;
  background:linear-gradient(135deg,{C['gold_dark']},{C['gold']});
  color:#050400; border:none; border-radius:24px;
  padding:12px 22px; font-size:13px; font-weight:700;
  cursor:pointer; z-index:9999;
  box-shadow:0 4px 20px {C['gold']}55;
  animation:fadeInUp 0.4s cubic-bezier(.4,0,.2,1) both;
  letter-spacing:.1em;
}}
#pwa-install-btn:hover {{ transform:translateY(-2px); box-shadow:0 6px 24px {C['gold']}77; }}
</style>
<button id="pwa-install-btn" onclick="installPWA()">
  📱 {'تثبيت التطبيق' if st.session_state.get('lang','').startswith('ال') else 'Install App'}
</button>
<script>
let deferredPrompt;
window.addEventListener('beforeinstallprompt', (e) => {{
  e.preventDefault();
  deferredPrompt = e;
  document.getElementById('pwa-install-btn').style.display = 'block';
}});
function installPWA() {{
  if (deferredPrompt) {{
    deferredPrompt.prompt();
    deferredPrompt.userChoice.then((r) => {{
      deferredPrompt = null;
      document.getElementById('pwa-install-btn').style.display = 'none';
    }});
  }}
}}
window.addEventListener('appinstalled', () => {{
  document.getElementById('pwa-install-btn').style.display = 'none';
}});
// Service Worker registration for offline support
if ('serviceWorker' in navigator) {{
  navigator.serviceWorker.register('data:application/javascript,' +
    encodeURIComponent(`
      const CACHE='mansa-v1';
      const ASSETS=['/'];
      self.addEventListener('install', e=>{{
        e.waitUntil(caches.open(CACHE).then(c=>c.addAll(ASSETS)));
      }});
      self.addEventListener('fetch', e=>{{
        e.respondWith(
          caches.match(e.request).then(r=>r||fetch(e.request))
        );
      }});
    `)
  ).catch(()=>{{}});
}}
</script>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# DATA HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
# ── Twelve Data symbol map (real-time primary source) ─────────────────────────
# Keys match the internal asset names used in fetch_live().
# Set TWELVE_DATA_API_KEY in .streamlit/secrets.toml or as an env var to
# activate real-time prices (<1 min latency) instead of Yahoo (~15 min).
_TD_SYMBOLS: dict = {
    "gold":  "XAU/USD",
    "silver":"XAG/USD",
    "oil":   "WTI/USD",
    "spx":   "SPX",
    "dxy":   "DXY",
    "vix":   "VIX",
    "us10y": "US10Y",
    "btc":   "BTC/USD",
    "plat":  "XPT/USD",
}

_TD_BASE_URL: str = "https://api.twelvedata.com"


def _get_td_key() -> str:
    """Resolve Twelve Data API key from secrets → env var."""
    try:
        return st.secrets["TWELVE_DATA_API_KEY"]
    except Exception:
        # st.secrets raises KeyError when the key is absent — expected, not an error
        _log.debug("TWELVE_DATA_API_KEY not in secrets, trying env var")
    return os.environ.get("TWELVE_DATA_API_KEY", "")


def _td_price(asset_key: str) -> tuple:
    """Fetch real-time price from Twelve Data for a single asset.

    Parameters
    ----------
    asset_key : str
        Internal asset key, e.g. ``"gold"``.  Must exist in ``_TD_SYMBOLS``.

    Returns
    -------
    tuple[float, float]
        *(price, prev_close)* or *(0.0, 0.0)* on failure.
    """
    api_key = _get_td_key()
    if not api_key:
        return 0.0, 0.0
    symbol = _TD_SYMBOLS.get(asset_key)
    if not symbol:
        return 0.0, 0.0
    try:
        import urllib.request as _ur2, json as _j2
        url = (f"{_TD_BASE_URL}/quote"
               f"?symbol={symbol}&apikey={api_key}")
        req = _ur2.Request(url, headers={"User-Agent": "MANSA/4.0"})
        with _ur2.urlopen(req, timeout=6) as resp:
            data = _j2.loads(resp.read())
        if data.get("status") == "error":
            _log.debug("Twelve Data error for %s: %s", asset_key, data.get("message"))
            return 0.0, 0.0
        price = float(data.get("close", 0) or 0)
        prev  = float(data.get("previous_close", price) or price)
        if price > 0:
            return price, prev
    except Exception:
        _log.debug("Twelve Data fetch failed for %s", asset_key, exc_info=True)
    return 0.0, 0.0


@st.cache_data(ttl=CACHE_TTL_PRICES)
def _td_batch_prices() -> dict:
    """Fetch all assets in a single Twelve Data batch request.

    Returns
    -------
    dict
        Mapping of asset_key → *(price, prev_close)* tuples.
        Empty dict when no API key is set or the request fails.
    """
    api_key = _get_td_key()
    if not api_key:
        return {}
    symbols_param = ",".join(_TD_SYMBOLS.values())
    try:
        import urllib.request as _ur3, json as _j3
        url = (f"{_TD_BASE_URL}/batch"
               f"?symbols={symbols_param}&apikey={api_key}")
        req = _ur3.Request(url, headers={"User-Agent": "MANSA/4.0"})
        with _ur3.urlopen(req, timeout=10) as resp:
            raw = _j3.loads(resp.read())
        result = {}
        # raw is a list of quote objects when using batch endpoint
        items = raw if isinstance(raw, list) else raw.get("data", [])
        sym_to_key = {v: k for k, v in _TD_SYMBOLS.items()}
        for item in items:
            sym  = item.get("symbol", "")
            key  = sym_to_key.get(sym)
            if not key:
                continue
            price = float(item.get("close", 0) or 0)
            prev  = float(item.get("previous_close", price) or price)
            if price > 0:
                result[key] = (price, prev)
        if result:
            _log.debug("Twelve Data batch: received %d/%d symbols",
                       len(result), len(_TD_SYMBOLS))
        return result
    except Exception:
        _log.warning("Twelve Data batch fetch failed", exc_info=True)
        return {}


def _yf_price(sym: str) -> tuple:
    """Fetch price via Yahoo Finance — used as fallback when Twelve Data is unavailable.

    Robust price fetch — 4 methods in order.
    Returns (price, prev_close).
    Never returns (0, 0) if any method succeeds.
    """
    # Method 1: yf.Ticker.history — cleanest, works on all yfinance versions
    try:
        hist = yf.Ticker(sym).history(period="5d", interval="1d")
        if not hist.empty and "Close" in hist.columns:
            closes = hist["Close"].dropna()
            if len(closes) >= 2:
                return float(closes.iloc[-1]), float(closes.iloc[-2])
            elif len(closes) == 1:
                p = float(closes.iloc[0])
                return p, p
    except Exception:
        _log.debug("Suppressed %s", exc_info=True)
    # Method 2: yf.download
    try:
        df = yf.download(sym, period="5d", auto_adjust=True,
                         progress=False, show_errors=False)
        if not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = ['_'.join(str(c) for c in col2).strip('_')
                              for col2 in df.columns]
            cc = next((c for c in df.columns if c.lower().startswith("close")), None)
            if cc:
                closes = df[cc].dropna()
                if len(closes) >= 2:
                    return float(closes.iloc[-1]), float(closes.iloc[-2])
                elif len(closes) == 1:
                    p = float(closes.iloc[0])
                    return p, p
    except Exception:
        _log.debug("Suppressed %s", exc_info=True)
    # Method 3: fast_info
    try:
        fi = yf.Ticker(sym).fast_info
        p  = float(fi["last_price"])
        pv = float(fi.get("previous_close", p) or p)
        if p > 0:
            return p, pv
    except Exception:
        _log.debug("Suppressed %s", exc_info=True)
    # Method 4: Free open metals price API (gold/silver only)
    try:
        import urllib.request as _ur, json as _j
        _FREE_METALS = {
            "GC=F": "https://query1.finance.yahoo.com/v8/finance/chart/GC=F?interval=1d&range=5d",
            "SI=F": "https://query1.finance.yahoo.com/v8/finance/chart/SI=F?interval=1d&range=5d",
        }
        if sym in _FREE_METALS:
            req = _ur.Request(_FREE_METALS[sym],
                              headers={"User-Agent":"Mozilla/5.0"})
            with _ur.urlopen(req, timeout=8) as r:
                data = _j.loads(r.read())
            closes = data["chart"]["result"][0]["indicators"]["quote"][0]["close"]
            closes = [x for x in closes if x is not None]
            if len(closes) >= 2:
                return float(closes[-1]), float(closes[-2])
    except Exception:
        _log.debug("Suppressed %s", exc_info=True)
    return 0.0, 0.0

# Cache that auto-clears if all prices are zero (avoids caching failures)
_live_cache = {"ts": 0, "data": None}

def _live_cache_valid():
    """Return True when the in-process price cache holds fresh, non-zero data."""
    import time as _t
    if _live_cache["data"] is None: return False
    if _t.time() - _live_cache["ts"] > 60: return False
    # Invalidate if gold came back 0 (means fetch failed)
    if _live_cache["data"].get("gold",{}).get("price",0) == 0: return False
    return True

# Reasonable fallback prices (updated quarterly) — shown when ALL fetches fail
_FALLBACK = {
    # Updated March 2026 — gold ~$5000/oz, silver ~$33/oz
    "gold":  (5000.0, 4980.0), "silver": (33.0,  32.8),
    "oil":   (70.0,   69.5),   "spx":    (5400.0, 5380.0),
    "dxy":   (104.0,  103.8),  "vix":    (16.0,   15.8),
    "us10y": (4.3,    4.28),   "btc":    (85000.0,84000.0),
    "plat":  (980.0,  975.0),
}

#: Yahoo Finance ticker map (fallback data source)
_YF_TICKERS: dict = {
    "gold":  "GC=F",   "silver": "SI=F",    "oil":   "CL=F",
    "spx":   "^GSPC",  "dxy":   "DX-Y.NYB", "vix":   "^VIX",
    "us10y": "^TNX",   "btc":   "BTC-USD",  "plat":  "PL=F",
}


@st.cache_data(ttl=CACHE_TTL_PRICES)
def fetch_live() -> dict:
    """Fetch live prices for gold, silver, oil, indices and crypto.

    Data source priority
    --------------------
    1. **Twelve Data** (real-time, <1 min latency) — activated when
       ``TWELVE_DATA_API_KEY`` is set in secrets or environment.
    2. **Yahoo Finance** (~15 min delay) — always-available fallback.
    3. **_FALLBACK** static values — used only when both live sources fail
       (exchange closed, network unavailable).

    Returns
    -------
    dict
        Mapping of asset key → ``{price, prev, change, pct, live, source}``.
    """
    # ── Try Twelve Data batch first (real-time) ───────────────────────────────
    td_prices = _td_batch_prices() if _get_td_key() else {}
    using_td  = bool(td_prices)

    out      = {}
    all_zero = True

    for k, yf_sym in _YF_TICKERS.items():
        # 1. Twelve Data
        if using_td and k in td_prices:
            p, pv = td_prices[k]
            source = "twelvedata"
        else:
            # 2. Yahoo Finance fallback
            p, pv = _yf_price(yf_sym)
            source = "yahoo"

        if p > 0:
            all_zero = False
        else:
            # 3. Static fallback — never show zeros to the user
            p, pv   = _FALLBACK.get(k, (0.0, 0.0))
            source  = "fallback"

        ch  = p - pv
        pct = (ch / pv * 100) if pv else 0.0
        out[k] = {
            "price":  p,
            "prev":   pv,
            "change": ch,
            "pct":    pct,
            "live":   source != "fallback",
            "source": source,
        }

    if all_zero:
        # Nothing came back — clear cache so next rerun retries immediately
        fetch_live.clear()

    return out

@st.cache_data(ttl=CACHE_TTL_PRICES)
def fetch_stock(sym: str) -> dict:
    """Fetch latest price and daily change for a single Yahoo Finance ticker.
    
        Parameters
        ----------
        sym : str
            Yahoo Finance ticker symbol, e.g. ``"BTC-USD"``.
        """
    p, pv = _yf_price(sym)
    ch  = p - pv
    pct = (ch / pv * 100) if pv else 0.0
    return {"price": p, "prev": pv, "change": ch, "pct": pct}

@st.cache_data(ttl=CACHE_TTL_HISTORY // 2)
def fetch_fx(ticker: Optional[str], fx_inverse: bool, fx_approx: float) -> float:
    """Return the current FX rate for a currency pair.
    
        Parameters
        ----------
        ticker : str or None
            Yahoo Finance FX ticker (``None`` → return *fx_approx* directly).
        fx_inverse : bool
            When ``True`` the ticker is quoted as CCY/USD, so the rate is inverted.
        fx_approx : float
            Fallback rate used when the live fetch fails.
        """
    if ticker is None:
        return fx_approx
    p, _ = _yf_price(ticker)
    return p if p > 0 else fx_approx

@st.cache_data(ttl=CACHE_TTL_HISTORY)
def fetch_history(period: str = "1y", ticker: str = "GC=F") -> pd.DataFrame:
    """Download OHLCV history for *ticker* over *period*.
    
        Parameters
        ----------
        period : str
            yfinance period string, e.g. ``"1y"``, ``"6mo"``, ``"5d"``.
        ticker : str
            Yahoo Finance symbol.
    
        Returns
        -------
        pd.DataFrame
            Reset-index DataFrame with a ``Date`` column and OHLCV columns.
            Empty DataFrame on failure.
        """
    try:
        df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
        df = df.reset_index()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ['_'.join(str(c) for c in col).strip('_') for col in df.columns]
        return df
    except Exception:
        return pd.DataFrame()

def find_col(df: pd.DataFrame, candidates: list) -> Optional[str]:
    """Return the first column name from *candidates* that exists in *df*, or ``None``."""
    for c in candidates:
        if c in df.columns: return c
    return None

def mkt_price(spot_usd_oz, mkt_cfg, purity_key):
    """Convert a USD/oz spot price to a local-market price.
    
        Parameters
        ----------
        spot_usd_oz : float
            Gold spot price in USD per troy ounce.
        mkt_cfg : dict
            Market configuration entry from ``MARKETS``.
        purity_key : str
            Active purity key, e.g. ``"21K — 875"``.
    
        Returns
        -------
        tuple[float, float]
            *(local_price, fx_rate)* where *local_price* is in the market's
            currency and unit, and *fx_rate* is USD → local.
        """
    pm   = PURITIES[purity_key]["mult"]
    uf   = mkt_cfg["unit_factor_from_oz"]
    rate = fetch_fx(mkt_cfg["fx_ticker"], mkt_cfg.get("fx_inverse",False), mkt_cfg["fx_approx"])
    fx   = 1.0/rate if mkt_cfg.get("fx_inverse",False) else rate
    return spot_usd_oz * pm * uf * fx, fx

# ── Writable CSV path (always tries to write here for updates) ────────────────
_CSV_WRITE_PATH = "updated_financial_data.csv"

def _get_csv_read_path():
    """Return first readable CSV path."""
    for p in [_CSV_WRITE_PATH,
              "merged_financial_data.csv",
              "/mnt/user-data/uploads/merged_financial_data.csv",
              "/mnt/user-data/uploads/1986_updated.csv"]:
        if os.path.exists(p):
            return p
    return None

# ── Google Drive fallback (Streamlit Cloud) ──────────────────────────────────
_GDRIVE_ID = "1eLCj83FImzFDsm1iAhvXSGuESzSn7P_H"
_GDRIVE_DL = ("https://drive.usercontent.google.com/download"
              "?id=1eLCj83FImzFDsm1iAhvXSGuESzSn7P_H&export=download&confirm=t")

@st.cache_data(ttl=CACHE_TTL_DRIVE)
def _load_from_drive():
    """Download training CSV from Google Drive when no local file exists."""
    try:
        import urllib.request as _ur3, io as _io3
        req = _ur3.Request(_GDRIVE_DL, headers={"User-Agent":"Mozilla/5.0"})
        with _ur3.urlopen(req, timeout=30) as _r3:
            raw = _r3.read()
        if raw[:20].strip().lower().startswith(b"<!doctype"):
            url2 = f"https://drive.google.com/uc?export=download&id={_GDRIVE_ID}&confirm=t"
            with _ur3.urlopen(_ur3.Request(url2,headers={"User-Agent":"Mozilla/5.0"}),timeout=30) as _r3b:
                raw = _r3b.read()
        df = pd.read_csv(_io3.BytesIO(raw), parse_dates=["Date"])
        df = df.sort_values("Date").reset_index(drop=True)
        try: df.to_csv(_CSV_WRITE_PATH, index=False)
        except Exception:
            _log.debug("Suppressed %s", exc_info=True)
        return df
    except Exception:
        _log.warning("Google Drive CSV fallback failed", exc_info=True)
        return pd.DataFrame()

@st.cache_data(ttl=CACHE_TTL_HISTORY)
def load_csv():
    """Load the training CSV, falling back to Google Drive if no local file exists.
    
        Returns
        -------
        pd.DataFrame
            Date-sorted DataFrame of historical gold and macro features.
            Empty DataFrame when all sources fail.
        """
    p = _get_csv_read_path()
    if p is not None:
        df = pd.read_csv(p, parse_dates=["Date"])
        return df.sort_values("Date").reset_index(drop=True)
    return _load_from_drive()  # fallback: download from Google Drive

def load_csv_fresh() -> pd.DataFrame:
    """Load CSV bypassing cache — used after an update."""
    p = _get_csv_read_path()
    if p is None:
        return pd.DataFrame()
    df = pd.read_csv(p, parse_dates=["Date"])
    return df.sort_values("Date").reset_index(drop=True)

def _compute_ratios(row, gold_col="Gold_Price"):
    """Recompute all ratio columns for a row dict."""
    gp = row.get(gold_col, 0) or 0
    def sr(a, b): return round(a / b, 6) if b and b != 0 else 0.0
    row["Gold_VIX_Ratio"]      = sr(gp, row.get("VIX", 1))
    row["Gold_US10Y_Ratio"]    = sr(gp, row.get("US10Y_Yield", 1))
    row["Gold_Oil_Ratio"]      = sr(gp, row.get("Oil_Price", 1))
    row["Gold_Silver_Ratio"]   = sr(gp, row.get("Silver_Price", 1))
    row["Gold_SPX_Ratio"]      = sr(gp, row.get("SPX_Close", 1))
    row["Gold_DXY_Ratio"]      = sr(gp, row.get("USD_Index", 1))
    row["Gold_CPI_Ratio"]      = sr(gp, row.get("CPI", 1))
    row["Gold_EFFR_Ratio"]     = sr(gp, row.get("EFFR", 1))
    row["Gold_RealRate_Ratio"] = sr(gp, row.get("Real_Interest_Rate", 1))
    return row

def fetch_new_rows(last_date):
    """
    Download new daily rows from Yahoo Finance since last_date.
    Returns a DataFrame of new rows (may be empty if already up-to-date).
    """
    import datetime as _dt
    today  = _dt.date.today()
    start  = (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    if pd.Timestamp(start).date() > today:
        return pd.DataFrame()   # already up-to-date

    tickers = {
        "Gold_Price":   "GC=F",
        "Silver_Price": "SI=F",
        "Oil_Price":    "CL=F",
        "SPX_Close":    "^GSPC",
        "USD_Index":    "DX-Y.NYB",
        "VIX":          "^VIX",
        "US10Y_Yield":  "^TNX",
    }

    dfs = {}
    for col, sym in tickers.items():
        try:
            df_t = yf.download(sym, start=start, auto_adjust=True, progress=False)
            if df_t.empty:
                continue
            df_t = df_t.reset_index()
            if isinstance(df_t.columns, pd.MultiIndex):
                df_t.columns = ["_".join(str(c) for c in col2).strip("_")
                                 for col2 in df_t.columns]
            cc = next((c for c in df_t.columns if c.lower().startswith("close")), None)
            dt = next((c for c in df_t.columns if c.lower() in ("date","datetime")), None)
            if cc and dt:
                dfs[col] = df_t[[dt, cc]].rename(columns={dt: "Date", cc: col})
                dfs[col]["Date"] = pd.to_datetime(dfs[col]["Date"])
        except Exception:
            _log.debug("Suppressed %s", exc_info=True)
    if not dfs:
        return pd.DataFrame()

    # Merge all on Date
    merged = None
    for col, df_c in dfs.items():
        if merged is None:
            merged = df_c
        else:
            merged = pd.merge(merged, df_c, on="Date", how="outer")

    if merged is None or merged.empty:
        return pd.DataFrame()

    merged = merged.sort_values("Date").reset_index(drop=True)

    # Fill FRED-sourced columns (CPI, EFFR, Real_Interest_Rate) with last known values
    # These update monthly so forward-fill is acceptable for daily rows
    for col in ["CPI", "EFFR", "Real_Interest_Rate"]:
        merged[col] = float("nan")

    # Compute ratio columns
    merged = merged.apply(lambda r: pd.Series(_compute_ratios(r.to_dict())), axis=1)

    # Ensure all original columns exist (fill missing with NaN)
    base_df = load_csv_fresh()
    if not base_df.empty:
        for c in base_df.columns:
            if c not in merged.columns:
                merged[c] = float("nan")
        merged = merged[base_df.columns]

    return merged

def update_csv_with_new_rows(new_rows_df, base_df):
    """
    Append new_rows_df to base_df, deduplicate on Date, save to CSV.
    Returns (updated_df, n_new_rows_added).
    """
    combined = pd.concat([base_df, new_rows_df], ignore_index=True)
    combined["Date"] = pd.to_datetime(combined["Date"])
    combined = combined.drop_duplicates(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    n_new = len(combined) - len(base_df)

    # For FRED columns, forward-fill from the existing data
    for col in ["CPI", "EFFR", "Real_Interest_Rate"]:
        if col in combined.columns:
            combined[col] = combined[col].ffill()

    try:
        combined.to_csv(_CSV_WRITE_PATH, index=False)
    except Exception:
        _log.debug("Suppressed", exc_info=True)   # read-only filesystem — still return updated df in memory

    return combined, n_new

@st.cache_resource
def load_models():
    """Load all serialised ML models from ``MODELS_DIR``.
    
        Returns
        -------
        tuple[dict, dict]
            *(models, r2_scores)* where *models* maps model name → *(obj, type)*
            and *r2_scores* maps model name → R² float.
        """
    models, r2s = {}, {}
    for name,(fname,mtype) in MODEL_FILES.items():
        path = os.path.join(MODELS_DIR, fname)
        if os.path.exists(path):
            try:
                if mtype == "keras":
                    from tensorflow.keras.models import load_model as lm
                    models[name] = (lm(path), mtype)
                else:
                    import joblib
                    models[name] = (joblib.load(path), mtype)
            except Exception:
                    _log.debug("Suppressed", exc_info=True)
    r2p = os.path.join(MODELS_DIR,"r2_scores.csv")
    if os.path.exists(r2p):
        try:
            r2df = pd.read_csv(r2p, index_col=0, names=["Model","R2"], header=0)
            r2s  = r2df["R2"].to_dict()
        except Exception:
                    _log.debug("Suppressed", exc_info=True)
    return models, r2s

def run_prediction(model_obj, mtype, feat_row):
    """
    feat_row: dict with all FEATURE_COLS keys.
    Returns float USD/oz or None.
    Prophet predicts the NEXT day properly.
    """
    try:
        if mtype == "prophet":
            # Prophet next-day: make 1 future date beyond last training date
            train_df = load_csv()
            if not train_df.empty and "Date" in train_df.columns:
                last_ds = train_df["Date"].max()
            else:
                last_ds = pd.Timestamp.today()
            future = model_obj.make_future_dataframe(periods=1)
            fc = model_obj.predict(future)
            return float(fc[fc["ds"] > last_ds]["yhat"].iloc[0])
        elif mtype == "keras":
            row = pd.DataFrame([feat_row])[FEATURE_COLS].values    # (1, 18)
            seq = np.tile(row, (TIME_STEP, 1))[np.newaxis]          # (1,10,18)
            return float(model_obj.predict(seq, verbose=0).flatten()[0])
        else:
            row = pd.DataFrame([feat_row])[FEATURE_COLS]
            return float(model_obj.predict(row)[0])
    except Exception:
        _log.debug("run_prediction failed", exc_info=True)
        return None

# ── Build feature row from last CSV row + live prices ─────────────────────────
def rgba(hex_color: str, alpha: float = 0.15) -> str:
    """Convert #RRGGBB to rgba(r,g,b,alpha) for plotly compatibility."""
    h = hex_color.lstrip('#')
    if len(h) == 6:
        r,g,b = int(h[0:2],16),int(h[2:4],16),int(h[4:6],16)
        return f'rgba({r},{g},{b},{alpha})'
    return hex_color

def _fmt_val(val: float) -> str:
    """Format a numeric value with adaptive decimal places for currency display."""
    if val < 100:
        return f"{val:,.4f}"
    if val < 1_000_000:
        return f"{val:,.2f}"
    return f"{val:,.0f}"

def build_features(g_ref: float, live: dict) -> dict:
    """Use last clean row from CSV as base, overwrite with today's live prices."""
    train_df = load_csv()
    base = {}
    if not train_df.empty:
        valid = train_df.dropna(subset=FEATURE_COLS)
        if not valid.empty:
            base = valid.iloc[-1][FEATURE_COLS].to_dict()

    sp  = live["silver"]["price"] or base.get("Silver_Price", 33.0)
    op  = live["oil"]["price"]    or base.get("Oil_Price",    72.0)
    sx  = live["spx"]["price"]    or base.get("SPX_Close",  5500.0)
    dx  = live["dxy"]["price"]    or base.get("USD_Index",   104.0)
    vx  = live["vix"]["price"]    or base.get("VIX",          17.0)
    t   = live["us10y"]["price"]  or base.get("US10Y_Yield",   4.3)
    cpi = base.get("CPI",  314.0)
    eff = base.get("EFFR",   4.33)
    rir = base.get("Real_Interest_Rate", 2.0)

    def s(a,b): return round(a/b,6) if b and b!=0 else 0.0
    return {
        "SPX_Close":           sx,  "CPI":          cpi, "EFFR":         eff,
        "USD_Index":           dx,  "Oil_Price":     op,  "Silver_Price":  sp,
        "Real_Interest_Rate":  rir, "VIX":           vx,  "US10Y_Yield":   t,
        "Gold_VIX_Ratio":      s(g_ref,vx),  "Gold_US10Y_Ratio":  s(g_ref,t),
        "Gold_Oil_Ratio":      s(g_ref,op),  "Gold_Silver_Ratio": s(g_ref,sp),
        "Gold_SPX_Ratio":      s(g_ref,sx),  "Gold_DXY_Ratio":    s(g_ref,dx),
        "Gold_CPI_Ratio":      s(g_ref,cpi), "Gold_EFFR_Ratio":   s(g_ref,eff),
        "Gold_RealRate_Ratio": s(g_ref,rir),
    }


def demo_predictions(g_ref: float) -> tuple:
    """
    Generate plausible demo predictions when model files are not found.
    Uses simple linear extrapolation and statistical offsets from the CSV.
    Returns dict of {model_name: predicted_price}.
    """
    tdf = load_csv()
    results = {}
    offsets = {
        "Linear Regression": 0.003,
        "Random Forest":     0.006,
        "Gradient Boosting": 0.008,
        "XGBoost":           0.009,
        "LSTM":              0.005,
        "Prophet":           0.004,
    }
    # Use recent 30-day momentum as base signal
    momentum = 0.0
    if not tdf.empty and "Gold_Price" in tdf.columns:
        recent = tdf["Gold_Price"].dropna().tail(30)
        if len(recent) >= 2:
            momentum = (float(recent.iloc[-1]) - float(recent.iloc[0])) / float(recent.iloc[0])
    for name, base_offset in offsets.items():
        noise = (hash(name) % 100 - 50) / 10000.0
        pred = g_ref * (1 + base_offset + momentum * 0.3 + noise)
        results[name] = round(pred, 2)
    # Demo R2 scores
    demo_r2 = {
        "Linear Regression": 0.8234,
        "Random Forest":     0.9512,
        "Gradient Boosting": 0.9678,
        "XGBoost":           0.9721,
        "LSTM":              0.9344,
        "Prophet":           0.8891,
    }
    return results, demo_r2

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    C2 = THEMES[st.session_state["theme"]]
    logo_svg = get_logo_svg(st.session_state["theme"], width=148)
    # Use components.html so SVG gradients/defs are NOT stripped by Streamlit
    st.components.v1.html(f"""
    <div style="text-align:center;padding:8px 0 4px;background:transparent;">
      {logo_svg}
      <div style="font-family:Georgia,serif;font-size:11px;font-style:italic;
                  color:{C2['muted']};margin-top:4px;">{L['tagline']}</div>
    </div>""", height=int(148*0.85)+40, scrolling=False)
    st.markdown("<div class='mansa-divider'></div>", unsafe_allow_html=True)

    # ── User Profile (quick name + risk) ─────────────────────────────────────
    with st.expander("👤 " + ("ملفي الشخصي" if st.session_state["lang"].startswith("ال") else "My Profile"), expanded=False):
        uname = st.text_input("الاسم / Name", value=st.session_state.get("user_name",""), key="sb_uname",
                               placeholder="عون الانصاري / Own Al Ansari")
        urisk = st.select_slider("المخاطرة / Risk", options=["Low","Medium","High","Aggressive"],
                                  value=st.session_state.get("user_risk","Medium"), key="sb_risk")
        if uname != st.session_state.get("user_name",""):
            st.session_state["user_name"] = uname
            _sb_save()
        if urisk != st.session_state.get("user_risk","Medium"):
            st.session_state["user_risk"] = urisk
            _sb_save()
        if st.session_state.get("user_name"):
            st.markdown(f"""
            <div style='background:{C2["card2"]};border:1px solid {C2["gold"]}44;border-radius:6px;
                        padding:8px 12px;margin-top:6px;font-size:11px;color:{C2["gold"]};'>
              👤 {st.session_state["user_name"]} · ⚖️ {urisk}
            </div>""", unsafe_allow_html=True)
        # Quick P&L if portfolio exists
        if st.session_state.get("portfolio_entries"):
            # live is fetched after the sidebar block; use fallback price for preview
            _g_preview = _FALLBACK["gold"][0]
            pnl_q = sum(
                (e["qty"]/({"gram":31.1035,"oz":1,"kg":31103.5,"tola":26.717}.get(e["unit"],31.1035)))
                * _g_preview - e["buy_price"] * e["qty"] /
                ({"gram":31.1035,"oz":1,"kg":31103.5,"tola":26.717}.get(e["unit"],31.1035))
                * {"USD":1,"JOD":1/0.709,"SAR":1/3.75,"AED":1/3.6725,"EGP":1/50.9,
                   "KWD":1/0.307,"QAR":1/3.64,"BHD":1/0.377,"GBP":1.27,"EUR":1.08,"TRY":1/32.0
                   }.get(e.get("currency","USD"),1)
                for e in st.session_state["portfolio_entries"]
            )
            pnl_col = C2["green"] if pnl_q >= 0 else C2["red"]
            st.markdown(f"""
            <div style='background:{C2["card2"]};border:1px solid {pnl_col}44;border-radius:6px;
                        padding:8px 12px;margin-top:6px;font-size:11px;'>
              <span style='color:{C2["muted"]};'>P&L: </span>
              <span style='color:{pnl_col};font-weight:700;'>${pnl_q:+,.2f}</span>
            </div>""", unsafe_allow_html=True)
        # Alert badge
        active_alerts = [a for a in st.session_state.get("price_alerts",[]) if not a.get("triggered")]
        if active_alerts:
            gold_col2 = C2["gold"]
            st.markdown(f"<div style='font-size:10px;color:{gold_col2};margin-top:4px;'>🔔 {len(active_alerts)} active alert(s)</div>", unsafe_allow_html=True)

    # Language selector  ── first so L is correct before nav
    lang_opts = list(LANGS.keys())
    new_lang  = st.selectbox("🌐 Language / اللغة", lang_opts,
                              index=lang_opts.index(st.session_state["lang"]),
                              key="lang_sel")
    if new_lang != st.session_state["lang"]:
        st.session_state["lang"] = new_lang
        st.session_state["nav"]  = LANGS[new_lang]["nav_dashboard"]
        st.rerun()

    L = LANGS[st.session_state["lang"]]

    # ── API key (for Claude chatbot) ─────────────────────────────────────────
    if not get_api_key():
        with st.expander("🔑 " + ("مفتاح API للمستشار الذكي" if st.session_state["lang"].startswith("ال") or st.session_state["lang"].startswith("اردو") else "Anthropic API Key"), expanded=False):
            _ki = st.text_input(
                "Key",
                value=st.session_state.get("_api_key",""),
                type="password",
                placeholder="sk-ant-...",
                label_visibility="collapsed",
                help="Get free key: console.anthropic.com · Stored in session only"
            )
            if _ki:
                st.session_state["_api_key"] = _ki
                st.success("✓")
    else:
        st.markdown(f"<div style='font-size:10px;color:{C2['green']};padding:2px 4px;'>🟢 Claude API active · Unlimited chat</div>", unsafe_allow_html=True)

    # Navigation
    nav_opts = [L["nav_dashboard"],L["nav_markets"],L["nav_charts"],L["nav_simulator"],
                L["nav_predictions"],L["nav_data"],L["nav_advisor"],
                L["nav_portfolio"],L["nav_demo"],L["nav_savings"],
                L["nav_calculator"],L["nav_calendar"],
                L["nav_sentiment"],L["nav_sessions"],
                L["nav_alerts"],L["nav_heatmap"],L["nav_mansa_score"],
                L["nav_zakat"],L["nav_journal"],
                L["nav_goldmap"],L["nav_drawdown"],
                L["nav_supply"],L["nav_currency"],
                L["nav_signals"],L["nav_geo"],L["nav_oilgold"],L["nav_report"],L["nav_cb"],
                L["nav_shopboard"],L["nav_invoice"],L["nav_production"],
                L["nav_fairprice"],L["nav_piecepricing"],
                L["nav_game"],L["nav_about"],L["nav_settings"]]
    # Ensure stored nav is in current language's options
    if st.session_state["nav"] not in nav_opts:
        st.session_state["nav"] = nav_opts[0]

    nav = st.radio("nav", nav_opts,
                   index=nav_opts.index(st.session_state["nav"]),
                   label_visibility="collapsed", key="nav_radio")
    st.session_state["nav"] = nav

    # ── Supabase sync status indicator ─────────────────────────────────────────
    sb_url, _ = _get_sb_config()
    if sb_url:
        sb_col = C2["green"]
        sb_lbl = "☁️ " + ("مزامن · Supabase" if st.session_state["lang"].startswith("ال") or st.session_state["lang"].startswith("اردو") else "Synced · Supabase")
    else:
        sb_col = C2["dim"]
        sb_lbl = "💾 " + ("تخزين محلي فقط" if st.session_state["lang"].startswith("ال") or st.session_state["lang"].startswith("اردو") else "Session only — no sync")
    st.markdown(
        f"<div style='font-size:9px;color:{sb_col};padding:2px 4px;letter-spacing:.05em;'>"
        f"{sb_lbl}</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<div class='mansa-divider'></div>", unsafe_allow_html=True)

    # Quick settings
    unit_list = list(UNITS.keys())
    st.session_state["unit"] = st.selectbox(
        L["weight_unit"], unit_list, index=unit_list.index(st.session_state["unit"]))

    pur_list = list(PURITIES.keys())
    idx_p = pur_list.index(st.session_state["purity"]) if st.session_state["purity"] in pur_list else 2
    st.session_state["purity"] = st.selectbox(L["purity"], pur_list, index=idx_p)

    theme_list = list(THEMES.keys())
    new_theme  = st.selectbox(L["theme"], theme_list,
                              index=theme_list.index(st.session_state["theme"]))
    if new_theme != st.session_state["theme"]:
        st.session_state["theme"] = new_theme
        st.rerun()

    # ── Persona selector ────────────────────────────────────────────────────────
    st.sidebar.markdown(f"<div style='font-size:9px;letter-spacing:.2em;color:{C['muted']};padding:6px 4px 2px;'>"
                        f"{L.get('persona_lbl','Persona').upper()}</div>", unsafe_allow_html=True)
    _persona_opts = {
        "trader":   L.get("persona_trader",   "📈 Trader"),
        "investor": L.get("persona_investor",  "💰 Investor"),
        "shop":     L.get("persona_shop",      "🏪 Shop Owner"),
        "factory":  L.get("persona_factory",   "🏭 Factory"),
        "designer": L.get("persona_designer",  "💍 Designer"),
        "buyer":    L.get("persona_buyer",     "🛍️ Buyer"),
    }
    _cur_persona = st.session_state.get("persona", "trader")
    _persona_labels = list(_persona_opts.values())
    _persona_keys   = list(_persona_opts.keys())
    _persona_idx    = _persona_keys.index(_cur_persona) if _cur_persona in _persona_keys else 0
    _selected_label = st.sidebar.selectbox(
        "", _persona_labels, index=_persona_idx,
        key="persona_selector", label_visibility="collapsed"
    )
    st.session_state["persona"] = _persona_keys[_persona_labels.index(_selected_label)]

    # Auto-refresh is on by default; user can pause it
    _ar_label = ("⏸ إيقاف التحديث التلقائي" if is_rtl() else "⏸ Pause auto-refresh")                 if st.session_state["auto_refresh"] else                 ("▶ تفعيل التحديث التلقائي" if is_rtl() else "▶ Enable auto-refresh")
    st.session_state["auto_refresh"] = st.checkbox(
        _ar_label, st.session_state["auto_refresh"])

    st.markdown("<div class='mansa-divider'></div>", unsafe_allow_html=True)
    ts = datetime.datetime.now().strftime("%H:%M:%S")

    # Mini gold sparkline in sidebar
    try:
        _sb_df = fetch_history("1mo","GC=F")
        if not _sb_df.empty:
            _sb_cl = find_col(_sb_df, ["Close","Close_GC=F"])
            _sb_dt = find_col(_sb_df, ["Date","Datetime"])
            if _sb_cl and _sb_dt:
                import plotly.graph_objects as _pgo
                _sb_fig = _pgo.Figure(_pgo.Scatter(
                    x=_sb_df[_sb_dt], y=_sb_df[_sb_cl],
                    line=dict(color=C2["gold"], width=2),
                    fill="tozeroy", fillcolor=rgba(C2["gold_dark"], 0.18),
                ))
                _sb_fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    height=70, margin=dict(l=0,r=0,t=0,b=0),
                    xaxis=dict(visible=False), yaxis=dict(visible=False),
                    showlegend=False,
                )
                st.plotly_chart(_sb_fig, use_container_width=True, config={"displayModeBar":False})
    except Exception:
        _log.debug("Suppressed %s", exc_info=True)
    st.markdown(f"""
    <div style='text-align:center;'>
      <span class='live-badge'><span class='live-dot'></span>
        <span class='live-text'>{L['live']} · {ts}</span></span>
    </div>
    <p style='text-align:center;font-family:{C['font_h']},serif;font-size:8px;
              color:{C2['dim']};margin-top:.8rem;letter-spacing:.1em;'>
      Yahoo Finance · ~15 min<br>{L['not_financial']}
    </p>""", unsafe_allow_html=True)

# ── Auto-refresh with visible countdown timer ────────────────────────────────
if "refresh_counter" not in st.session_state:
    st.session_state["refresh_counter"] = 60


# ── Auto-refresh via JavaScript (no CPU polling, no sleep loop) ──────────────
# When enabled: a silent JS timer fires after REFRESH_INTERVAL seconds,
# clears the price cache server-side, then reloads the page.
# The user sees a live badge update — no manual button needed anywhere.
_refresh_interval_ms = REFRESH_INTERVAL * 1000

if st.session_state["auto_refresh"]:
    # Inject a JS countdown that reloads the page after REFRESH_INTERVAL seconds
    # st.cache_data uses TTL so by the time the page reloads, fresh data is served.
    st.sidebar.markdown(f"""
    <div id="ar-ring" style='display:flex;align-items:center;gap:10px;
                              padding:6px 4px;'>
      <svg width="40" height="40" viewBox="0 0 40 40" id="ar-svg">
        <circle cx="20" cy="20" r="16" fill="none"
                stroke="{C['border2']}" stroke-width="3"/>
        <circle cx="20" cy="20" r="16" fill="none"
                stroke="{C['green']}" stroke-width="3"
                stroke-dasharray="100.5" stroke-dashoffset="0"
                stroke-linecap="round" transform="rotate(-90 20 20)"
                id="ar-arc"/>
        <text x="20" y="24" text-anchor="middle" font-family="monospace"
              font-size="9" fill="{C['green']}" font-weight="700"
              id="ar-txt">{REFRESH_INTERVAL}s</text>
      </svg>
      <div style='font-size:10px;color:{C["muted"]};'>
        <div style='color:{C["green"]};font-weight:700;font-size:9px;
                    letter-spacing:.1em;'>{"تحديث تلقائي" if L.get("dir")=="rtl" else "AUTO-REFRESH"}</div>
        <div id="ar-lbl">{"يُحدَّث خلال {REFRESH_INTERVAL}s" if L.get("dir")=="rtl"
                          else f"Refreshes in {REFRESH_INTERVAL}s"}</div>
      </div>
    </div>
    <script>
    (function() {{
      var total = {REFRESH_INTERVAL};
      var circ  = 2 * Math.PI * 16;   // 100.53
      var arc   = document.getElementById('ar-arc');
      var txt   = document.getElementById('ar-txt');
      var lbl   = document.getElementById('ar-lbl');
      var rem   = total;
      function tick() {{
        rem--;
        if (rem <= 0) {{
          window.location.reload();
          return;
        }}
        var pct  = rem / total;
        var dash = circ * (1 - pct);
        if (arc) arc.style.strokeDashoffset = dash;
        var col  = rem > 20 ? '{C["green"]}' : (rem > 10 ? '{C["gold"]}' : '{C["red"]}');
        if (arc) arc.style.stroke = col;
        if (txt) {{ txt.textContent = rem + 's'; txt.style.fill = col; }}
        if (lbl) lbl.textContent = '{("يُحدَّث خلال" if L.get("dir")=="rtl" else "Refreshes in")} ' + rem + 's';
      }}
      setInterval(tick, 1000);
    }})();
    </script>
    """, unsafe_allow_html=True)
    # Also pre-clear the cache so when JS reloads the page, fresh data is ready
    # (cache TTL handles this naturally, but explicit clear ensures it)
    if st.session_state.get("_refresh_pre_cleared") != int(datetime.datetime.now().minute):
        if st.session_state["refresh_counter"] <= 5:
            fetch_live.clear()
            st.session_state["_refresh_pre_cleared"] = int(datetime.datetime.now().minute)
    # Decrement counter for cache pre-clear trigger only (no sleep/rerun loop)
    st.session_state["refresh_counter"] = max(0, st.session_state["refresh_counter"] - 1)
    if st.session_state["refresh_counter"] <= 0:
        st.session_state["refresh_counter"] = REFRESH_INTERVAL
        fetch_live.clear()

# ── Shared live values ────────────────────────────────────────────────────────
live      = fetch_live()
g_ref     = live["gold"]["price"] if live["gold"]["price"] > 0 else _FALLBACK["gold"][0]
unit_cfg  = UNITS[st.session_state["unit"]]
pur_cfg   = PURITIES[st.session_state["purity"]]
U_FACTOR  = unit_cfg["factor"]
P_MULT    = pur_cfg["mult"]
U_SYM     = unit_cfg["symbol"]
P_LABEL   = pur_cfg["label"]
PM        = MARKETS[st.session_state["primary_mkt"]]
pm_price, pm_fx = mkt_price(g_ref, PM, st.session_state["purity"])
pm_change = live["gold"]["change"] * P_MULT * PM["unit_factor_from_oz"] * pm_fx
pm_col    = C["green"] if pm_change >= 0 else C["red"]
pm_arr    = "▲" if pm_change >= 0 else "▼"
tomorrow  = (datetime.date.today() + datetime.timedelta(days=1)).strftime("%d %b %Y")

# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL LIVE TICKER STRIP — shown on every page
# ══════════════════════════════════════════════════════════════════════════════
def render_ticker_strip():
    """Render the animated live-price ticker bar at the top of every page."""
    """Scrolling live price ticker rendered above every page."""
    ticker_items = [
        ("XAU/USD",  live["gold"],   "$",  "/oz"),
        ("XAG/USD",  live["silver"], "$",  "/oz"),
        ("WTI Oil",  live["oil"],    "$",  "/bbl"),
        ("S&P 500",  live["spx"],    "",   ""),
        ("DXY",      live["dxy"],    "",   ""),
        ("VIX",      live["vix"],    "",   ""),
        ("US 10Y",   live["us10y"],  "",   "%"),
        ("BTC/USD",  live["btc"],    "$",  ""),
        ("Platinum", live["plat"],   "$",  "/oz"),
    ]
    # Build ticker HTML items (doubled for seamless loop)
    items_html = ""
    for name, d, pfx, sfx in ticker_items:
        p   = d["price"]; pct = d["pct"]
        col = "#52D98A" if pct >= 0 else "#FF5555"
        arr = "▲" if pct >= 0 else "▼"
        is_gold = name == "XAU/USD"
        items_html += f"""
        <span class='tk-item {"tk-gold" if is_gold else ""}'>
          <span class='tk-name'>{name}</span>
          <span class='tk-price'>{pfx}{p:,.2f}{sfx}</span>
          <span class='tk-chg' style='color:{col};'>{arr}{abs(pct):.2f}%</span>
        </span>
        <span class='tk-sep'>◆</span>"""
    # Double for seamless infinite scroll
    full = items_html + items_html

    now_str = datetime.datetime.now().strftime("%H:%M:%S UTC")
    live_col = "#52D98A" if live["gold"].get("live", True) else "#F5C830"

    st.components.v1.html(f"""
    <style>
      @keyframes ticker-scroll {{
        0%   {{ transform:translateX(0); }}
        100% {{ transform:translateX(-50%); }}
      }}
      .tk-wrap {{
        width:100%; overflow:hidden; background:{C['bg2']};
        border-bottom:1px solid {C['gold']}33;
        display:flex; align-items:center;
        padding:0; position:relative;
        font-family:'JetBrains Mono','Share Tech Mono',monospace;
      }}
      .tk-label {{
        background:linear-gradient(135deg,{C['gold_dark']},{C['gold']});
        color:#050400; font-size:9px; font-weight:900;
        letter-spacing:.2em; padding:6px 12px;
        white-space:nowrap; flex-shrink:0; z-index:2;
        text-transform:uppercase;
      }}
      .tk-scroll-outer {{
        flex:1; overflow:hidden; position:relative;
      }}
      .tk-scroll-outer::before, .tk-scroll-outer::after {{
        content:''; position:absolute; top:0; bottom:0; width:40px;
        z-index:1; pointer-events:none;
      }}
      .tk-scroll-outer::before {{ left:0; background:linear-gradient(90deg,{C['bg2']},transparent); }}
      .tk-scroll-outer::after  {{ right:0; background:linear-gradient(-90deg,{C['bg2']},transparent); }}
      .tk-scroll {{
        display:flex; align-items:center;
        white-space:nowrap;
        animation:ticker-scroll 38s linear infinite;
      }}
      .tk-scroll:hover {{ animation-play-state:paused; }}
      .tk-item {{
        display:inline-flex; align-items:center; gap:6px;
        padding:6px 14px; font-size:11px;
      }}
      .tk-gold {{
        background:{C['gold']}14;
        border-radius:3px;
      }}
      .tk-name  {{ color:{C['muted']}; font-size:9px; letter-spacing:.1em; }}
      .tk-price {{ color:{C['gold_pale']}; font-weight:700; }}
      .tk-chg   {{ font-size:10px; }}
      .tk-sep   {{ color:{C['gold']}33; font-size:8px; margin:0 2px; }}
      .tk-ts    {{
        background:{C['bg2']}; color:{live_col};
        font-size:9px; padding:0 10px; flex-shrink:0;
        letter-spacing:.1em; white-space:nowrap;
        border-left:1px solid {C['border2']};
      }}
    </style>
    <div class='tk-wrap'>
      <div class='tk-label'>{'● مباشر' if L.get('dir')=='rtl' else '● LIVE'}</div>
      <div class='tk-scroll-outer'>
        <div class='tk-scroll'>{full}</div>
      </div>
      <div class='tk-ts'>⏱ {now_str}</div>
    </div>
    """, height=34, scrolling=False)

# Render ticker at top of every page
render_ticker_strip()

# ── Page header helper ────────────────────────────────────────────────────────
def ph(title: str, sub: str = "") -> None:
    """Render a consistent page header with title, optional subtitle, and a gold divider.
    
        Parameters
        ----------
        title : str
            Primary heading text (supports HTML entities).
        sub : str, optional
            Secondary subtitle rendered in italic muted text.
        """
    rtl = "direction:rtl;text-align:right;" if L["dir"]=="rtl" else ""
    st.markdown(f"""
    <h1 style='font-family:{C['font_h']},serif;font-size:26px;font-weight:900;
               color:{C['gold_pale']};letter-spacing:.06em;margin-bottom:2px;{rtl}'>{title}</h1>
    <p style='font-family:{C['font_b']},serif;font-size:14px;font-style:italic;
              color:{C['muted']};{rtl}'>{sub}</p>""", unsafe_allow_html=True)
    st.markdown("<div class='mansa-divider'></div>", unsafe_allow_html=True)

def stat_card(label: str, value: str, sub: str = "", col_override: Optional[str] = None) -> str:
    """Return the HTML string for a single stat card widget.
    
        Parameters
        ----------
        label : str
            Small uppercase label shown above the value.
        value : str
            Primary metric value (formatted string).
        sub : str, optional
            Small secondary annotation shown below the value.
        col_override : str, optional
            CSS colour override for the value text.
        """
    vc = col_override or C["gold_pale"]
    return f"""
    <div class='stat-card'>
      <div class='stat-label'>{label}</div>
      <div class='stat-value' style='color:{vc};'>{value}</div>
      {"<div style='font-size:11px;color:"+C['dim']+"';>"+sub+"</div>" if sub else ""}
    </div>"""

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
if nav == L["nav_dashboard"]:
    ph(f"{C['brand']}  {L['app_name']} · {L['gold_intelligence']}", L["real_time"])

    # ── Live status + refresh ─────────────────────────────────────────────────
    is_live_data = live["gold"].get("live", True) and live["gold"]["price"] > 0
    use_ar = is_rtl()
    db_c1, db_c2, db_c3 = st.columns([4, 2, 1])
    with db_c1:
        if not is_live_data:
            st.warning("⚠️ تعذّر جلب الأسعار الحية — تُعرض أسعار تقريبية. اضغط 🔄 للمحاولة." if use_ar else
                       "⚠️ Live prices unavailable — showing estimated values. Press 🔄 to retry.")
        else:
            # Show which data source is active
            src = live["gold"].get("source", "yahoo")
            src_badge = ("🟢 Twelve Data · Real-time"
                         if src == "twelvedata"
                         else "🟡 Yahoo Finance · ~15 min delay")
            src_col  = C["green"] if src == "twelvedata" else C["gold"]
            st.markdown(
                f"<div style='font-size:11px;color:{src_col};padding-top:4px;'>"
                f"{src_badge} · {datetime.datetime.now().strftime('%H:%M:%S')} UTC"
                f"</div>",
                unsafe_allow_html=True,
            )
    with db_c2:
        st.markdown(f"<div style='font-size:10px;color:{C['dim']};padding-top:6px;'>{(L['yf_delay'])}</div>", unsafe_allow_html=True)
    with db_c3:
        # Manual refresh still available but auto-refresh handles it silently
        if st.button("🔄", key="dash_refresh_btn",
                     help=("تحديث يدوي" if use_ar else "Manual refresh"),
                     use_container_width=True):
            fetch_live.clear()
            fetch_stock.clear()
            st.session_state["refresh_counter"] = REFRESH_INTERVAL
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Hero ──────────────────────────────────────────────────────────────────
    h1, h2 = st.columns([3, 2], gap="medium")
    with h1:
        gold_chg_col = C["green"] if live["gold"]["change"] >= 0 else C["red"]
        gold_arr     = "▲" if live["gold"]["change"] >= 0 else "▼"
        st.markdown(f"""
        <div class='hero-wrap'>
          <div class='stat-label'>{PM['flag']} {st.session_state['primary_mkt']} · {L['primary_market']}</div>
          <div class='hero-price'>{pm_price:,.3f}<span class='hero-unit'>{PM['currency']} / {PM['unit_label']}</span></div>
          <div class='hero-change' style='color:{pm_col};'>
            {pm_arr} {abs(pm_change):,.4f} ({live['gold']['pct']:+.2f}%)
          </div>
          <div style='margin-top:10px;'>
            <span class='purity-badge'>{P_LABEL} · {pur_cfg['fine']}‰</span>
            <span class='purity-badge'>{U_SYM}</span>
          </div>
          <div class='hero-meta'>{datetime.datetime.now().strftime('%d %b %Y · %H:%M')} UTC</div>
        </div>""", unsafe_allow_html=True)
    with h2:
        spot_unit = g_ref * U_FACTOR * P_MULT
        st.markdown(f"""
        <div class='hero-wrap' style='height:100%;'>
          <div class='stat-label'>🌐 {L['spot_price']} · XAU/USD</div>
          <div class='hero-price' style='font-size:40px;'>${g_ref:,.2f}<span class='hero-unit'>USD/oz</span></div>
          <div class='hero-change' style='color:{gold_chg_col};font-size:13px;'>
            {gold_arr} ${abs(live['gold']['change']):,.2f} ({live['gold']['pct']:+.2f}%)
          </div>
          <div style='margin-top:12px;'>
            <div class='stat-label' style='margin-bottom:4px;'>{P_LABEL} · {U_SYM}</div>
            <div style='font-family:{C['font_m']},monospace;font-size:22px;color:{C['gold_hi']};'>
              ${spot_unit:,.5f}
            </div>
          </div>
          <div class='hero-meta'>24K ref: ${g_ref:,.2f} / oz t</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Live market strip ─────────────────────────────────────────────────────
    st.markdown(f"<div class='section-label'>{L['market_overview']}</div>", unsafe_allow_html=True)
    strip_data = [
        ("Silver",   live["silver"], "$", "/oz"),
        ("Crude Oil",live["oil"],    "$", "/bbl"),
        ("DXY",      live["dxy"],    "",  ""),
        ("VIX",      live["vix"],    "",  ""),
        ("US 10Y",   live["us10y"],  "",  "%"),
        ("Platinum", live["plat"],   "$", "/oz"),
    ]
    scols = st.columns(len(strip_data))
    for col, (name, d, pfx, sfx) in zip(scols, strip_data):
        p   = d["price"]; pct = d["pct"]; ch = d["change"]
        pct_col = C["green"] if pct >= 0 else C["red"]
        arr_s   = "▲" if pct >= 0 else "▼"
        with col:
            st.markdown(f"""
            <div class='ticker-card'>
              <div class='ticker-name'>{name}</div>
              <div class='ticker-price'>{pfx}{p:,.2f}{sfx}</div>
              <div class='ticker-chg' style='color:{pct_col};'>{arr_s} {pct:+.2f}%</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Live market snapshot table ────────────────────────────────────────────
    st.markdown(f"<div class='section-label'>{L['live_snapshot']}</div>", unsafe_allow_html=True)
    mkt_table_rows = [
        ("XAU/USD — Gold",   live["gold"],   "$", "/oz",  "🥇"),
        ("XAG/USD — Silver", live["silver"], "$", "/oz",  "🪙"),
        ("WTI Crude Oil",    live["oil"],    "$", "/bbl", "🛢️"),
        ("S&P 500",          live["spx"],    "",  "pts",  "📈"),
        ("USD Index (DXY)",  live["dxy"],    "",  "",     "💵"),
        ("VIX Fear Index",   live["vix"],    "",  "",     "😨"),
        ("US 10Y Yield",     live["us10y"],  "",  "%",    "🏦"),
        ("Bitcoin",          live["btc"],    "$", "",     "₿"),
        ("Platinum",         live["plat"],   "$", "/oz",  "⬡"),
    ]
    tbl_html = f"""<table style='width:100%;border-collapse:collapse;font-family:{C['font_m']},monospace;font-size:13px;'>
      <thead><tr style='border-bottom:1px solid {C['border2']};'>
        <th style='text-align:left;padding:8px 10px;font-size:9px;letter-spacing:.2em;color:{C['muted']};'>{L['col_asset']}</th>
        <th style='text-align:right;padding:8px 10px;font-size:9px;letter-spacing:.2em;color:{C['muted']};'>{L['col_price']}</th>
        <th style='text-align:right;padding:8px 10px;font-size:9px;letter-spacing:.2em;color:{C['muted']};'>{L['col_change']}</th>
        <th style='text-align:right;padding:8px 10px;font-size:9px;letter-spacing:.2em;color:{C['muted']};'>%</th>
        <th style='text-align:center;padding:8px 10px;font-size:9px;color:{C['muted']};'>{L['col_status']}</th>
      </tr></thead><tbody>"""
    for name, d, pfx, sfx, icon in mkt_table_rows:
        p = d["price"]; ch = d["change"]; pct = d["pct"]; is_lv = d.get("live", True)
        ch_col = C["green"] if ch >= 0 else C["red"]
        arr    = "▲" if ch >= 0 else "▼"
        bar_w  = min(abs(pct)*8, 100)
        badge  = f"<span style='color:{C['green']};font-size:9px;'>🟢 Live</span>" if is_lv else f"<span style='color:{C['gold']};font-size:9px;'>🟡 Est.</span>"
        tbl_html += f"""<tr style='border-bottom:1px solid {C['border']}22;'>
          <td style='padding:10px;color:{C['text']};'><span style='font-size:16px;margin-right:8px;'>{icon}</span>{name}</td>
          <td style='text-align:right;padding:10px;color:{C['gold_pale']};font-weight:700;font-size:15px;'>{pfx}{p:,.3f}{sfx}</td>
          <td style='text-align:right;padding:10px;color:{ch_col};'>{arr} {pfx}{abs(ch):,.3f}</td>
          <td style='text-align:right;padding:10px;'>
            <div style='display:flex;align-items:center;justify-content:flex-end;gap:6px;'>
              <div style='width:50px;height:4px;background:{C['border']};border-radius:2px;'>
                <div style='width:{bar_w:.0f}%;height:100%;background:{ch_col};border-radius:2px;'></div>
              </div>
              <span style='color:{ch_col};min-width:52px;text-align:right;'>{pct:+.2f}%</span>
            </div>
          </td>
          <td style='text-align:center;padding:10px;'>{badge}</td>
        </tr>"""
    tbl_html += "</tbody></table>"
    st.markdown(f"<div style='background:{C['card2']};border:1px solid {C['border2']};border-radius:6px;overflow:hidden;'>{tbl_html}</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Key stats ─────────────────────────────────────────────────────────────
    st.markdown(f"<div class='section-label'>{L['key_stats']}</div>", unsafe_allow_html=True)
    @st.cache_data(ttl=CACHE_TTL_HISTORY * 12)
    def get_52w():
        """Return the 52-week high and low for gold (XAU/USD).
        
            Returns
            -------
            tuple[float, float]
                *(high, low)* in USD per troy ounce.
            """
        df = fetch_history("1y","GC=F")
        if df.empty: return None, None, None
        cl = find_col(df,["Close","Close_GC=F"])
        if not cl: return None, None, None
        c = df[cl].dropna()
        return float(c.max()), float(c.min()), float(c.mean())
    hi52, lo52, avg52 = get_52w()
    ytd_ret = (g_ref / avg52 - 1) * 100 if avg52 else 0
    gsr     = g_ref / live["silver"]["price"] if live["silver"]["price"] > 0 else 0
    gor     = g_ref / live["oil"]["price"]    if live["oil"]["price"]    > 0 else 0
    k1,k2,k3,k4,k5,k6 = st.columns(6)
    for col, lbl, val in [
        (k1, L["wk52_high"],   f"${hi52:,.0f}"   if hi52  else "—"),
        (k2, L["wk52_low"],    f"${lo52:,.0f}"   if lo52  else "—"),
        (k3, L["wk52_avg"],    f"${avg52:,.0f}"  if avg52 else "—"),
        (k4, L["ytd_return"],  f"{ytd_ret:+.1f}%"),
        (k5, L["gold_silver"], f"{gsr:.1f}x"),
        (k6, L["gold_oil"],    f"{gor:.1f}x"),
    ]:
        with col:
            st.markdown(f"""<div class='stat-card' style='text-align:center;'>
              <div class='stat-label'>{lbl}</div>
              <div class='stat-value' style='font-size:15px;'>{val}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── All purities table ────────────────────────────────────────────────────
    if st.session_state.get("show_purity_table", True):
        st.markdown(f"<div class='section-label'>{L['all_purities']} · {PM['flag']} {st.session_state['primary_mkt']}</div>", unsafe_allow_html=True)
        pur_rows = []
        for pn, pc2 in PURITIES.items():
            pv, _ = mkt_price(g_ref, PM, pn)
            pur_rows.append({
                ("العيار" if use_ar else "Karat"): pc2["label"], "Fineness": f"{pc2['fine']}‰",
                f"{PM['currency']}/{PM['unit_label']}": f"{pv:,.4f}",
                "USD/oz": f"${g_ref * pc2['mult']:,.2f}",
            })
        st.dataframe(pd.DataFrame(pur_rows), use_container_width=True, hide_index=True)

    # ── Stocks strip ──────────────────────────────────────────────────────────
    active_stks = st.session_state.get("active_stocks", ["GLD","SLV","NEM","GOLD","WPM"])
    if active_stks:
        st.markdown(f"<div class='section-label'>{L['stocks_indices']}</div>", unsafe_allow_html=True)
        stk_cols = st.columns(len(active_stks))
        for col, stk in zip(stk_cols, active_stks):
            sd = fetch_stock(stk)
            sc2 = C["green"] if sd["pct"] >= 0 else C["red"]
            sa  = "▲" if sd["pct"] >= 0 else "▼"
            with col:
                st.markdown(f"""
                <div class='ticker-card'>
                  <div class='ticker-name'>{stk}</div>
                  <div class='ticker-price'>${sd['price']:,.2f}</div>
                  <div class='ticker-chg' style='color:{sc2};'>{sa} {sd['pct']:+.2f}%</div>
                </div>""", unsafe_allow_html=True)

elif nav == L["nav_markets"]:
    use_ar = is_rtl()
    ph(L["nav_markets"], f"{P_LABEL} · {PM['unit_label']}")

    arab_flags  = {"🇯🇴","🇸🇦","🇦🇪","🇪🇬","🇰🇼","🇶🇦","🇧🇭","🇴🇲","🇱🇧","🇮🇶","🇹🇷"}
    arab_mkts   = {k:v for k,v in MARKETS.items() if v["flag"] in arab_flags}
    global_mkts = {k:v for k,v in MARKETS.items() if k not in arab_mkts}

    def render_grid(mkt_dict, ncols=3):
        """Render a responsive grid of market price cards.
        
            Parameters
            ----------
            mkt_dict : dict
                Subset of ``MARKETS`` to display.
            ncols : int
                Number of columns in the grid.
            """
        active = st.session_state["active_mkts"]
        keys   = [k for k in mkt_dict if k in active]
        if not keys:
            st.info("—"); return
        for row_keys in [keys[i:i+ncols] for i in range(0,len(keys),ncols)]:
            cols_r = st.columns(ncols)
            for col_r, mk in zip(cols_r, row_keys):
                m   = MARKETS[mk]
                mp, mfx = mkt_price(g_ref, m, st.session_state["purity"])
                mp_chg  = live["gold"]["change"] * P_MULT * m["unit_factor_from_oz"] * mfx
                mc  = C["green"] if mp_chg>=0 else C["red"]
                ma  = "▲" if mp_chg>=0 else "▼"
                r24,_ = mkt_price(g_ref, m, "24K — 999.9")
                is_pm = mk == st.session_state["primary_mkt"]
                bs    = f"border-color:{C['gold']};" if is_pm else ""
                pb    = "<span class='purity-badge'>✦</span>" if is_pm else ""
                with col_r:
                    st.markdown(f"""
                    <div class='mkt-card' style='{bs}'>
                      <div style='display:flex;justify-content:space-between;'>
                        <div><span style='font-size:20px;'>{m['flag']}</span>
                          <div class='mkt-name'>{mk}</div></div>
                        <div>{pb}</div>
                      </div>
                      <div class='mkt-price'>{mp:,.3f}<span class='mkt-unit'>{m['currency']}/{m['unit_label']}</span></div>
                      <div class='mkt-chg' style='color:{mc};'>{ma} {abs(mp_chg):,.4f} ({live['gold']['pct']:+.2f}%)</div>
                      <div style='margin-top:6px;'>
                        <span class='purity-badge'>{P_LABEL}</span>
                        <span style='font-size:9px;color:{C['dim']};'>24K: {r24:,.3f} {m['currency']}</span>
                      </div>
                      <div style='font-size:11px;font-style:italic;color:{C['dim']};margin-top:3px;'>{m['note'] if use_ar else m.get('note_en', m['note'])}</div>
                    </div>""", unsafe_allow_html=True)

    st.markdown(f"<div class='section-label'>{L['arab_markets']}</div>", unsafe_allow_html=True)
    render_grid(arab_mkts, 3)
    st.markdown(f"<div class='section-label'>{L['intl_markets']}</div>", unsafe_allow_html=True)
    render_grid(global_mkts, 3)

    st.markdown(f"<div class='section-label'>{L['purity_matrix']}</div>", unsafe_allow_html=True)
    act_m = [k for k in st.session_state["active_mkts"] if k in MARKETS]
    rows  = []
    for pn,pc in PURITIES.items():
        row = {"Purity": f"{pc['label']} ({pc['fine']}‰)"}
        for mk in act_m:
            v,_ = mkt_price(g_ref, MARKETS[mk], pn)
            row[f"{MARKETS[mk]['flag']} {mk.split('(')[0].strip()}"] = f"{v:,.2f} {MARKETS[mk]['currency']}"
        rows.append(row)
    if rows: st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: CHARTS
# ═══════════════════════════════════════════════════════════════════════════════
elif nav == L["nav_charts"]:
    use_ar = is_rtl()
    ph(L["nav_charts"])

    # ── Controls row ──────────────────────────────────────────────────────────
    ca1,ca2,ca3,ca4 = st.columns([2,2,2,2])
    with ca1:
        chart_asset = st.selectbox(
            ("الأصل" if use_ar else "Asset"),
            ["Gold","Silver","Crude Oil","S&P 500","Platinum","Bitcoin","USD Index"],
            key="ch_asset"
        )
    with ca2:
        chart_period = st.select_slider(
            ("الفترة" if use_ar else "Period"),
            options=["5d","1mo","3mo","6mo","1y","2y","5y"], value="1y", key="ch_period"
        )
    with ca3:
        chart_type = st.selectbox(
            ("نوع الرسم" if use_ar else "Chart Type"),
            ["Candlestick","Line","Area"], key="ch_type"
        )
    with ca4:
        indicators = st.multiselect(
            ("المؤشرات" if use_ar else "Indicators"),
            ["MA20","MA50","MA200","Bollinger Bands","MACD","RSI","Stochastic","Fibonacci"],
            default=["MA50","MA200"],
            key="ch_inds"
        )

    tm = {
        "Gold":"GC=F","Silver":"SI=F","Crude Oil":"CL=F","S&P 500":"^GSPC",
        "Platinum":"PL=F","Bitcoin":"BTC-USD","USD Index":"DX-Y.NYB"
    }
    sym   = tm[chart_asset]
    df_ch = fetch_history(chart_period, sym)
    fch   = U_FACTOR*P_MULT if chart_asset=="Gold" else 1.0

    if not df_ch.empty:
        cl = find_col(df_ch,["Close",f"Close_{sym}"])
        op = find_col(df_ch,["Open",f"Open_{sym}"])
        hi = find_col(df_ch,["High",f"High_{sym}"])
        lo = find_col(df_ch,["Low",f"Low_{sym}"])
        vo = find_col(df_ch,["Volume",f"Volume_{sym}"])
        dt = find_col(df_ch,["Date","Datetime"])

        # How many indicator rows needed
        has_macd  = "MACD"       in indicators
        has_rsi   = "RSI"        in indicators
        has_stoch = "Stochastic" in indicators
        n_sub = 1 + sum([has_macd, has_rsi or has_stoch, bool(vo)])
        row_h = [0.60]
        if has_macd:             row_h.append(0.15)
        if has_rsi or has_stoch: row_h.append(0.13)
        if vo:                   row_h.append(0.12)
        # Normalize
        s = sum(row_h)
        row_h = [r/s for r in row_h]

        fig = make_subplots(
            rows=n_sub, cols=1, shared_xaxes=True,
            vertical_spacing=0.03, row_heights=row_h
        )

        # ── Main price trace ──────────────────────────────────────────────────
        if chart_type == "Candlestick" and all([op,hi,lo,cl,dt]):
            fig.add_trace(go.Candlestick(
                x=df_ch[dt],
                open=df_ch[op]*fch, high=df_ch[hi]*fch,
                low=df_ch[lo]*fch,  close=df_ch[cl]*fch,
                increasing=dict(line_color=C["green"], fillcolor=rgba(C["green"], 0.4)),
                decreasing=dict(line_color="#FF5555",  fillcolor="rgba(255,85,85,0.4)"),
                name=chart_asset
            ), row=1, col=1)
        elif cl and dt:
            fig.add_trace(go.Scatter(
                x=df_ch[dt], y=df_ch[cl]*fch, name=chart_asset,
                line=dict(color=C["gold"], width=2),
                fill="tozeroy" if chart_type=="Area" else None,
                fillcolor=rgba(C["gold_dark"],0.15)
            ), row=1, col=1)

        if cl and dt:
            c_ser = df_ch[cl] * fch

            # ── Moving Averages ───────────────────────────────────────────────
            ma_cfg = {"MA20":(20,"#7B9FD4",1.0),"MA50":(50,C["gold_hi"],1.2),"MA200":(200,C["red"],1.2)}
            for ma_key,(w,col,lw) in ma_cfg.items():
                if ma_key in indicators and len(df_ch)>=w:
                    ma_s = c_ser.rolling(w).mean()
                    fig.add_trace(go.Scatter(
                        x=df_ch[dt], y=ma_s, name=ma_key,
                        line=dict(color=col, width=lw, dash="dot"), opacity=0.8
                    ), row=1, col=1)

            # ── Bollinger Bands ────────────────────────────────────────────────
            if "Bollinger Bands" in indicators and len(df_ch)>=20:
                bb_m = c_ser.rolling(20).mean()
                bb_s = c_ser.rolling(20).std()
                bb_u = bb_m + 2*bb_s
                bb_l = bb_m - 2*bb_s
                fig.add_trace(go.Scatter(
                    x=df_ch[dt], y=bb_u, name="BB Upper",
                    line=dict(color=C["accent"],width=1,dash="dot"), opacity=0.7
                ), row=1, col=1)
                fig.add_trace(go.Scatter(
                    x=df_ch[dt], y=bb_l, name="BB Lower",
                    line=dict(color=C["accent"],width=1,dash="dot"), opacity=0.7,
                    fill="tonexty", fillcolor=rgba(C["accent"],0.06)
                ), row=1, col=1)
                fig.add_trace(go.Scatter(
                    x=df_ch[dt], y=bb_m, name="BB Middle",
                    line=dict(color=C["accent"],width=0.8,dash="longdash"), opacity=0.5
                ), row=1, col=1)

            # ── Fibonacci Retracement ─────────────────────────────────────────
            if "Fibonacci" in indicators:
                fib_hi = float(c_ser.max())
                fib_lo = float(c_ser.min())
                fib_range = fib_hi - fib_lo
                fib_levels = {
                    "Fib 0.0%":   fib_lo,
                    "Fib 23.6%":  fib_lo + 0.236*fib_range,
                    "Fib 38.2%":  fib_lo + 0.382*fib_range,
                    "Fib 50.0%":  fib_lo + 0.500*fib_range,
                    "Fib 61.8%":  fib_lo + 0.618*fib_range,
                    "Fib 78.6%":  fib_lo + 0.786*fib_range,
                    "Fib 100%":   fib_hi,
                }
                fib_colors = ["#FF5555","#FF9955","#FFCC55","#55CC55","#5599FF","#9955FF","#FF55CC"]
                for (lbl,lvl),fcol in zip(fib_levels.items(), fib_colors):
                    fig.add_hline(
                        y=lvl, line_dash="dot", line_color=fcol,
                        line_width=0.8, opacity=0.6,
                        annotation_text=f" {lbl} ${lvl:,.0f}",
                        annotation_position="right",
                        annotation_font=dict(size=8, color=fcol),
                        row=1, col=1
                    )

            # ── Sub-row tracker
            sub_row = 2

            # ── MACD ─────────────────────────────────────────────────────────
            if has_macd and len(df_ch)>=26:
                ema12 = c_ser.ewm(span=12).mean()
                ema26 = c_ser.ewm(span=26).mean()
                macd  = ema12 - ema26
                signal= macd.ewm(span=9).mean()
                hist  = macd - signal
                fig.add_trace(go.Bar(
                    x=df_ch[dt], y=hist,
                    marker_color=[C["green"] if v>=0 else "#FF5555" for v in hist],
                    name="MACD Hist", opacity=0.7
                ), row=sub_row, col=1)
                fig.add_trace(go.Scatter(
                    x=df_ch[dt], y=macd, name="MACD",
                    line=dict(color=C["gold"],width=1.2)
                ), row=sub_row, col=1)
                fig.add_trace(go.Scatter(
                    x=df_ch[dt], y=signal, name="Signal",
                    line=dict(color=C["red"],width=1.2,dash="dot")
                ), row=sub_row, col=1)
                fig.add_annotation(
                    text="MACD", x=0.01, y=0.5, xref="paper",
                    yref=f"y{sub_row} domain", showarrow=False,
                    font=dict(size=9, color=C["muted"])
                )
                sub_row += 1

            # ── RSI ───────────────────────────────────────────────────────────
            if has_rsi and len(df_ch)>=15:
                d_rsi = c_ser.diff()
                ag_rsi = d_rsi.clip(lower=0).rolling(14).mean()
                al_rsi = (-d_rsi.clip(upper=0)).rolling(14).mean()
                rsi_s  = 100-(100/(1+ag_rsi/al_rsi.replace(0,1e-9)))
                fig.add_trace(go.Scatter(
                    x=df_ch[dt], y=rsi_s, name="RSI(14)",
                    line=dict(color="#9B59B6",width=1.5)
                ), row=sub_row, col=1)
                fig.add_hline(y=70, line_dash="dot", line_color=C["red"],
                              line_width=0.8, opacity=0.6, row=sub_row, col=1)
                fig.add_hline(y=30, line_dash="dot", line_color=C["green"],
                              line_width=0.8, opacity=0.6, row=sub_row, col=1)
                fig.add_hline(y=50, line_dash="dot", line_color=C["muted"],
                              line_width=0.6, opacity=0.4, row=sub_row, col=1)
                fig.update_yaxes(range=[0,100], row=sub_row, col=1)

            # ── Stochastic ─────────────────────────────────────────────────────
            if has_stoch and len(df_ch)>=14 and hi and lo:
                low14 = df_ch[lo].rolling(14).min() * fch
                hi14  = df_ch[hi].rolling(14).max() * fch
                k_line= 100*(c_ser - low14)/(hi14 - low14 + 1e-9)
                d_line= k_line.rolling(3).mean()
                fig.add_trace(go.Scatter(
                    x=df_ch[dt], y=k_line, name="%K",
                    line=dict(color="#00C9A7",width=1.2)
                ), row=sub_row, col=1)
                fig.add_trace(go.Scatter(
                    x=df_ch[dt], y=d_line, name="%D",
                    line=dict(color="#FF8C42",width=1.2,dash="dot")
                ), row=sub_row, col=1)
                fig.add_hline(y=80, line_dash="dot", line_color=C["red"],
                              line_width=0.7, opacity=0.5, row=sub_row, col=1)
                fig.add_hline(y=20, line_dash="dot", line_color=C["green"],
                              line_width=0.7, opacity=0.5, row=sub_row, col=1)
                fig.update_yaxes(range=[0,100], row=sub_row, col=1)
            if has_rsi or has_stoch:
                sub_row += 1

            # ── Volume ────────────────────────────────────────────────────────
            if vo and dt:
                vc2 = [C["green"] if (op and df_ch[cl].iloc[i]>=df_ch[op].iloc[i])
                       else "#FF5555" for i in range(len(df_ch))]
                fig.add_trace(go.Bar(
                    x=df_ch[dt], y=df_ch[vo],
                    marker_color=vc2, opacity=0.5, name="Volume"
                ), row=sub_row, col=1)
                fig.add_annotation(
                    text=("حجم" if use_ar else "VOL"), x=0.01, y=0.5, xref="paper",
                    yref=f"y{sub_row} domain", showarrow=False,
                    font=dict(size=9, color=C["muted"])
                )

        # ── Layout ────────────────────────────────────────────────────────────
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=C["card"],
            font=dict(color=C["text"], family=C["font_m"], size=10),
            legend=dict(bgcolor=C["card2"], bordercolor=C["border2"],
                        font=dict(size=9), orientation="h",
                        yanchor="bottom", y=1.01, xanchor="left", x=0),
            margin=dict(l=0,r=60,t=40,b=0),
            height=620 + 80*sum([has_macd,has_rsi or has_stoch]),
            xaxis_rangeslider_visible=False,
            hovermode="x unified",
            hoverlabel=dict(bgcolor=C["card2"], bordercolor=C["gold"],
                            font=dict(color=C["text"], size=10)),
            dragmode="zoom",
        )
        for i in range(1, n_sub+1):
            fig.update_xaxes(gridcolor=C["border2"], color=C["muted"], row=i, col=1)
            fig.update_yaxes(gridcolor=C["border2"], color=C["muted"], row=i, col=1)

        st.plotly_chart(fig, use_container_width=True)

        # ── Stats bar ─────────────────────────────────────────────────────────
        if cl and dt:
            v  = df_ch[cl]*fch
            # Current signal summary
            sig_rsi = "—"; sig_ma = "—"; sig_bb = "—"
            if len(df_ch)>=15:
                d2=v.diff(); ag2=d2.clip(lower=0).rolling(14).mean()
                al2=(-d2.clip(upper=0)).rolling(14).mean()
                rsi_now = float((100-(100/(1+ag2/al2.replace(0,1e-9)))).iloc[-1])
                sig_rsi = f"RSI {rsi_now:.0f} {('🟢 ذروة بيع' if rsi_now<30 else '🔴 ذروة شراء' if rsi_now>70 else '🟡 محايد') if use_ar else ('🟢 Oversold' if rsi_now<30 else '🔴 Overbought' if rsi_now>70 else '🟡 Neutral')}"
            if len(df_ch)>=50:
                ma50_now  = float(v.rolling(50).mean().iloc[-1])
                sig_ma    = f"MA50 {('🟢 فوق' if float(v.iloc[-1])>ma50_now else '🔴 تحت') if use_ar else ('🟢 Above' if float(v.iloc[-1])>ma50_now else '🔴 Below')}"
            if len(df_ch)>=20:
                bm2=v.rolling(20).mean(); bs2=v.rolling(20).std()
                bb_pos = (float(v.iloc[-1])-float((bm2-2*bs2).iloc[-1]))/(float(4*bs2.iloc[-1])+1e-9)*100
                sig_bb = f"BB {('🟢 ذروة بيع' if bb_pos<15 else '🔴 ذروة شراء' if bb_pos>85 else '🟡 وسط') if use_ar else ('🟢 Oversold' if bb_pos<15 else '🔴 Overbought' if bb_pos>85 else '🟡 Mid')}"

            st_cols = st.columns(8)
            for sc,lbl,val in [
                (st_cols[0],("أعلى" if use_ar else "High"),    f"${v.max():,.2f}"),
                (st_cols[1],("أدنى" if use_ar else "Low"),     f"${v.min():,.2f}"),
                (st_cols[2],("متوسط" if use_ar else "Avg"),    f"${v.mean():,.2f}"),
                (st_cols[3],("انحراف" if use_ar else "Std Dev"),f"${v.std():,.2f}"),
                (st_cols[4],("عائد" if use_ar else "Return"),  f"{((v.iloc[-1]/v.iloc[0])-1)*100:+.2f}%"),
                (st_cols[6],sig_ma,      ""),
                (st_cols[7],sig_bb,      ""),
            ]:
                with sc:
                    st.markdown(f"""<div class='stat-card' style='text-align:center;padding:8px;'>
                      <div class='stat-label' style='font-size:8px;'>{lbl}</div>
                      <div style='font-size:11px;color:{C["gold_pale"]};font-weight:600;'>{val}</div>
                    </div>""", unsafe_allow_html=True)

        # ── Fullscreen tip ────────────────────────────────────────────────────
        st.markdown(f"""
        <div style='font-size:10px;color:{C["dim"]};margin-top:4px;text-align:right;'>
          💡 {"انقر مرتين على الرسم لإعادة الضبط · اسحب لتكبير · حوّم للتفاصيل · عجلة الماوس للتكبير" if use_ar else
               "Double-click to reset · Drag to zoom · Hover for details · Scroll to zoom · Click legend to toggle"}
        </div>""", unsafe_allow_html=True)
    else:
        st.warning("⏳ " + ("جاري تحميل البيانات..." if use_ar else "Loading chart data..."))

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: SIMULATOR
# ═══════════════════════════════════════════════════════════════════════════════
elif nav == L["nav_simulator"]:
    use_ar = is_rtl()
    ph(L["nav_simulator"])
    def safe(d): return round(d["price"],2) if d["price"]>0 else 0.0
    sc1,sc2 = st.columns(2,gap="large")
    with sc1:
        sim_spx=st.number_input(("مؤشر S&P 500" if use_ar else "S&P 500"),value=safe(live["spx"]),step=10.0,format="%.2f")
        sim_dxy=st.number_input(("مؤشر الدولار DXY" if use_ar else "DXY"),value=safe(live["dxy"]),step=0.1,format="%.2f")
        sim_oil=st.number_input(("النفط (دولار/برميل)" if use_ar else "Oil (USD/bbl)"),value=safe(live["oil"]),step=1.0,format="%.2f")
        sim_silver=st.number_input(("الفضة (دولار/أوقية)" if use_ar else "Silver (USD/oz)"),value=safe(live["silver"]),step=0.1,format="%.2f")
        sim_vix=st.number_input(("مؤشر VIX" if use_ar else "VIX"),value=safe(live["vix"]),step=0.5,format="%.2f")
    with sc2:
        sim_cpi=st.number_input(("مؤشر CPI التضخم" if use_ar else "CPI Inflation"),value=314.0,step=1.0,format="%.1f")
        sim_effr=st.number_input(("الفائدة الفعلية EFFR (%)" if use_ar else "EFFR (%)"),value=4.33,step=0.25,format="%.2f")
        sim_real=st.number_input(("معدل الفائدة الحقيقي (%)" if use_ar else "Real Rate (%)"),value=2.0,step=0.1,format="%.2f")
        sim_us10y=st.number_input(("عائد 10Y (%)" if use_ar else "US 10Y (%)"),value=safe(live["us10y"]),step=0.05,format="%.2f")
        sim_btc=st.number_input(("بيتكوين (دولار)" if use_ar else "Bitcoin (USD)"),value=safe(live["btc"]),step=100.0,format="%.0f")
    BL=dict(spx=5500,dxy=104,oil=72,silver=33,vix=17,cpi=310,effr=4.33,real=2.0,us10y=4.3,btc=85000)
    SN=dict(spx=0.08,dxy=-12.5,oil=3.2,silver=55.0,vix=8.5,cpi=4.2,effr=-35.0,real=-28.0,us10y=-18.0,btc=0.003)
    delta=sum([SN["spx"]*(sim_spx-BL["spx"]),SN["dxy"]*(sim_dxy-BL["dxy"]),SN["oil"]*(sim_oil-BL["oil"]),
        SN["silver"]*(sim_silver-BL["silver"]),SN["vix"]*(sim_vix-BL["vix"]),SN["cpi"]*(sim_cpi-BL["cpi"]),
        SN["effr"]*(sim_effr-BL["effr"]),SN["real"]*(sim_real-BL["real"]),SN["us10y"]*(sim_us10y-BL["us10y"]),
        SN["btc"]*(sim_btc-BL["btc"])])
    sim_spot=g_ref+delta; sim_pct=(delta/g_ref)*100
    smc=C["green"] if delta>=0 else C["red"]; sma="▲" if delta>=0 else "▼"
    sim_pm,_=mkt_price(sim_spot,PM,st.session_state["purity"])
    cur_pm,_=mkt_price(g_ref,PM,st.session_state["purity"])
    st.markdown("<br>",unsafe_allow_html=True)
    st.markdown(f"""
    <div class='hero-wrap' style='text-align:center;'>
      <div class='stat-label'>{PM['flag']} Simulated · {st.session_state['primary_mkt']}</div>
      <div class='hero-price'>{sim_pm:,.3f}<span class='hero-unit'>{PM['currency']}/{PM['unit_label']}</span></div>
      <div class='hero-change' style='color:{smc};'>{sma} {abs(sim_pm-cur_pm):,.4f} ({sim_pct:+.2f}%)</div>
      <div class='hero-meta'>Simulated: ${sim_spot:,.2f}/oz · Current: ${g_ref:,.2f}/oz</div>
    </div>""",unsafe_allow_html=True)
    impacts={"S&P 500":SN["spx"]*(sim_spx-BL["spx"]),"DXY":SN["dxy"]*(sim_dxy-BL["dxy"]),
        "Oil":SN["oil"]*(sim_oil-BL["oil"]),"Silver":SN["silver"]*(sim_silver-BL["silver"]),
        "VIX":SN["vix"]*(sim_vix-BL["vix"]),"CPI":SN["cpi"]*(sim_cpi-BL["cpi"]),
        "EFFR":SN["effr"]*(sim_effr-BL["effr"]),"Real Rate":SN["real"]*(sim_real-BL["real"]),
        "US10Y":SN["us10y"]*(sim_us10y-BL["us10y"]),"BTC":SN["btc"]*(sim_btc-BL["btc"])}
    si=sorted(impacts.items(),key=lambda x:abs(x[1]))
    fig_t=go.Figure(go.Bar(y=[i[0] for i in si],x=[i[1] for i in si],orientation="h",
        marker_color=[C["green"] if v>=0 else C["red"] for _,v in si],
        text=[f"${v:+,.1f}" for _,v in si],textposition="outside",
        textfont=dict(family=C["font_m"],size=10,color=C["text"])))
    fig_t.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor=C["card"],
        font=dict(color=C["text"]),height=340,margin=dict(l=0,r=80,t=20,b=0),
        xaxis=dict(gridcolor=C["border2"],color=C["muted"],zeroline=True,zerolinecolor=C["gold_dark"]),
        yaxis=dict(gridcolor=C["border2"],color=C["muted"]))
    st.plotly_chart(fig_t,use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: AI PREDICTIONS  — fully rebuilt
# ═══════════════════════════════════════════════════════════════════════════════
elif nav == L["nav_predictions"]:
    ph(L["nav_predictions"], f"{L['prediction_date']}: {tomorrow}")

    models_dict, r2_scores = load_models()

    # Prediction date banner
    st.markdown(f"""
    <div style='background:{C['card2']};border:1px solid {C['gold']}55;border-radius:5px;
                padding:12px 20px;margin-bottom:1rem;display:flex;justify-content:space-between;align-items:center;'>
      <div>
        <div class='stat-label'>{L['prediction_date']} · {L['tomorrow']}</div>
        <div style='font-family:{C['font_h']},serif;font-size:22px;font-weight:700;
                    color:{C['gold_hi']};'>{tomorrow}</div>
      </div>
      <div style='text-align:right;'>
        <div class='stat-label'>Live Spot Reference</div>
        <div style='font-family:{C['font_m']},monospace;font-size:20px;color:{C['gold_pale']};'>
          ${g_ref:,.2f} <span style='font-size:11px;color:{C['muted']};'>USD / oz t</span>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

    is_demo = not models_dict
    if is_demo:
        demo_results, demo_r2 = demo_predictions(g_ref)
        results   = demo_results
        r2_scores = demo_r2
        valid     = results.copy()
        st.info(L.get("models_demo_note","⚠️ Showing demo predictions (run train_models.py for real AI predictions)"))
    else:
        # Build features & run all models
        feats  = build_features(g_ref, live)
        results = {}
        for name,(model_obj,mtype) in models_dict.items():
            results[name] = run_prediction(model_obj, mtype, feats)
        valid = {k:v for k,v in results.items() if v is not None}

    # Identify best model by R²
    best_model = None
    if r2_scores:
        r2_valid = {k:r2_scores[k] for k in valid if k in r2_scores}
        if r2_valid:
            best_model = max(r2_valid, key=r2_valid.get)

    if True:  # always show predictions

        # ── Model selector ────────────────────────────────────────────────────
        st.markdown(f"<div class='section-label'>{L['select_model']}</div>", unsafe_allow_html=True)
        sel_col1, sel_col2 = st.columns([2,3])
        with sel_col1:
            model_names_avail = list(valid.keys()) if valid else list(MODEL_FILES.keys())
            default_idx = model_names_avail.index(best_model) if best_model and best_model in model_names_avail else 0
            selected_model = st.selectbox(L["select_model"], model_names_avail,
                                          index=default_idx, label_visibility="collapsed")
        with sel_col2:
            if best_model:
                st.markdown(f"""
                <div style='padding:8px 14px;background:{C['gold']}22;border:1px solid {C['gold']}88;
                            border-radius:4px;display:inline-block;'>
                  <span style='font-family:{C['font_h']},serif;font-size:9px;letter-spacing:.2em;
                               color:{C['gold_hi']};'>⭐ {L['best_model']}: {best_model}
                    {"  (R² = "+f"{r2_scores[best_model]:.4f})" if best_model in r2_scores else ""}
                  </span>
                </div>""", unsafe_allow_html=True)

        # ── Selected model big display ────────────────────────────────────────
        if selected_model in valid:
            sel_pred = valid[selected_model]
            sel_diff = sel_pred - g_ref
            sel_col  = C["green"] if sel_diff>=0 else C["red"]
            sel_arr  = "▲" if sel_diff>=0 else "▼"
            sel_r2   = r2_scores.get(selected_model)

            st.markdown(f"""
            <div class='hero-wrap' style='margin:1rem 0;'>
              <div style='display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:1rem;'>
                <div>
                  <div class='stat-label'>{selected_model} · {tomorrow}</div>
                  <div class='hero-price'>${sel_pred:,.2f}<span class='hero-unit'>USD / oz t</span></div>
                  <div class='hero-change' style='color:{sel_col};'>
                    {sel_arr} ${abs(sel_diff):,.2f} ({sel_diff/g_ref*100:+.2f}% {L['vs_spot']})
                  </div>
                  {"<div class='hero-meta'>R² = "+f"{sel_r2:.4f}</div>" if sel_r2 else ""}
                </div>
                <div style='text-align:right;'>
                  <div class='stat-label' style='margin-bottom:6px;'>{"⭐ "+L['best_model'] if selected_model==best_model else ""}</div>
                  <span class='purity-badge'>{P_LABEL}</span>
                  <span class='purity-badge'>{U_SYM}</span>
                </div>
              </div>
            </div>""", unsafe_allow_html=True)

            # ── Multi-unit multi-currency output ──────────────────────────────
            st.markdown(f"<div class='section-label'>{L['pred_multi_unit']}</div>", unsafe_allow_html=True)

            # Units
            unit_cols = st.columns(len(UNITS))
            for col, (uname, ucfg) in zip(unit_cols, UNITS.items()):
                val = sel_pred * ucfg["factor"] * P_MULT
                with col:
                    st.markdown(f"""
                    <div class='stat-card' style='text-align:center;'>
                      <div class='stat-label'>{ucfg['symbol']}</div>
                      <div class='stat-value' style='font-size:14px;'>${val:,.4f}</div>
                      <div style='font-size:10px;color:{C['dim']};'>{P_LABEL}</div>
                    </div>""", unsafe_allow_html=True)

            # Key currencies (gram 21K)
            st.markdown("<br>", unsafe_allow_html=True)
            key_markets = ["Jordan (JOD)","Saudi Arabia (SAR)","UAE (AED)","Egypt (EGP)",
                           "Kuwait (KWD)","USA (USD)","UK (GBP)","EU (EUR)"]
            mkt_cols = st.columns(4)
            row_idx = 0
            for mk in key_markets:
                if mk not in MARKETS: continue
                m   = MARKETS[mk]
                pp,_ = mkt_price(sel_pred, m, st.session_state["purity"])
                cur_p,_ = mkt_price(g_ref, m, st.session_state["purity"])
                dd   = pp - cur_p
                dc   = C["green"] if dd>=0 else C["red"]
                da   = "▲" if dd>=0 else "▼"
                with mkt_cols[row_idx % 4]:
                    st.markdown(f"""
                    <div class='stat-card'>
                      <div class='stat-label'>{m['flag']} {mk.split('(')[0].strip()}</div>
                      <div class='stat-value' style='font-size:15px;'>{pp:,.3f}
                        <span style='font-size:10px;color:{C['muted']};'>{m['currency']}/{m['unit_label']}</span>
                      </div>
                      <div style='font-family:{C['font_m']},monospace;font-size:10px;color:{dc};margin-top:3px;'>
                        {da} {abs(dd):,.3f} {m['currency']}
                      </div>
                    </div>""", unsafe_allow_html=True)
                row_idx += 1

        # ── All algorithms grid ───────────────────────────────────────────────
        st.markdown(f"<div class='section-label'>{L['algo_predictions']} · USD/oz · {tomorrow}</div>", unsafe_allow_html=True)

        ALGO_INFO = {
            "Linear Regression": {"icon":"📐","desc_ar":"نموذج خطي أساسي · سريع وقابل للتفسير","desc_en":"Linear baseline model · fast & interpretable"},
            "Random Forest":     {"icon":"🌲","desc_ar":"غابة من أشجار القرار · يلتقط الأنماط غير الخطية","desc_en":"Ensemble of decision trees · captures nonlinear patterns"},
            "Gradient Boosting": {"icon":"⚡","desc_ar":"تعزيز متدرج · دقة عالية على البيانات المالية","desc_en":"Sequential boosting · high accuracy on financial data"},
            "XGBoost":           {"icon":"🚀","desc_ar":"تعزيز متدرج محسّن · الأفضل للبيانات الجدولية","desc_en":"Optimized gradient boosting · best for tabular data"},
            "LSTM":              {"icon":"🧠","desc_ar":"شبكة ذاكرة LSTM · تتعلم الاعتماديات الزمنية","desc_en":"LSTM neural net · learns temporal dependencies"},
            "Prophet":           {"icon":"🔮","desc_ar":"نموذج السلاسل الزمنية من Meta · أفضل للتنبؤ المستقبلي","desc_en":"Meta time-series model · best for forward forecasts"},
        }
        use_ar = is_rtl()

        all_names = list(ALGO_INFO.keys())
        for row_start in range(0, len(all_names), 3):
            cols3 = st.columns(3)
            for col, algo_name in zip(cols3, all_names[row_start:row_start+3]):
                pred = results.get(algo_name)
                r2   = r2_scores.get(algo_name)
                info = ALGO_INFO[algo_name]
                desc = info["desc_ar"] if use_ar else info["desc_en"]
                is_best = algo_name == best_model
                is_sel  = algo_name == selected_model

                if pred is not None:
                    diff    = pred - g_ref
                    dc      = C["green"] if diff>=0 else C["red"]
                    da      = "▲" if diff>=0 else "▼"
                    pstr    = f"${pred:,.2f}"
                    dstr    = f"{da} ${abs(diff):,.2f} ({diff/g_ref*100:+.2f}%)"
                    r2str   = f"R² = {r2:.4f}" if r2 else "—"
                    opacity = ""
                else:
                    dc,da   = C["dim"],"—"
                    pstr    = "—"; dstr = "Model not loaded"; r2str = "—"
                    opacity = "opacity:0.45;"

                gold_border = f"border-color:{C['gold_hi']};box-shadow:0 0 10px {C['gold']}44;" if is_best else ""
                sel_border  = f"border-color:{C['gold']};" if is_sel and not is_best else ""
                best_tag    = f"<span class='purity-badge' style='background:{C['gold']}33;'>⭐ Best</span>" if is_best else ""

                with col:
                    st.markdown(f"""
                    <div class='pred-card {"pred-best" if is_best else ""}' style='{opacity}{gold_border or sel_border}'>
                      <div style='display:flex;align-items:center;justify-content:space-between;margin-bottom:4px;'>
                        <div style='display:flex;align-items:center;gap:8px;'>
                          <span style='font-size:18px;'>{info['icon']}</span>
                          <div class='pred-algo'>{algo_name}</div>
                        </div>
                        {best_tag}
                      </div>
                      <div class='pred-price'>{pstr}</div>
                      <div class='pred-diff' style='color:{dc};'>{dstr}</div>
                      <div class='pred-r2'>{r2str}</div>
                      <div style='font-family:{C['font_b']},serif;font-size:11px;font-style:italic;
                                  color:{C['dim']};margin-top:8px;border-top:1px solid {C['border']};padding-top:6px;'>
                        {desc}
                      </div>
                    </div>""", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

        # ── Consensus summary ─────────────────────────────────────────────────
        if valid:
            vals     = list(valid.values())
            cons     = float(np.mean(vals))
            hi_p     = max(vals); lo_p = min(vals)
            spread   = hi_p - lo_p
            cdiff    = cons - g_ref
            cc       = C["green"] if cdiff>=0 else C["red"]
            ca       = "▲" if cdiff>=0 else "▼"

            st.markdown(f"<div class='section-label'>{L['consensus']}</div>", unsafe_allow_html=True)
            c1,c2,c3,c4 = st.columns(4)
            for col,lbl,val,vc in [
                (c1,L['consensus'],    f"${cons:,.2f}", cc),
                (c2,L['model_high'],   f"${hi_p:,.2f}", C["green"]),
                (c3,L['model_low'],    f"${lo_p:,.2f}", C["red"]),
                (c4,L['spread'],       f"${spread:,.2f}", C["muted"]),
            ]:
                with col:
                    st.markdown(f"""<div class='stat-card' style='text-align:center;'>
                      <div class='stat-label'>{lbl}</div>
                      <div class='stat-value' style='font-size:20px;color:{vc};'>{val}</div>
                    </div>""", unsafe_allow_html=True)

            st.markdown(f"""
            <div style='background:{C['card2']};border:1px solid {C['border2']};border-radius:5px;
                        padding:14px 20px;margin-top:10px;'>
              <div class='stat-label' style='margin-bottom:8px;'>{L['consensus']} {L['vs_spot']}</div>
              <div style='font-family:{C['font_m']},monospace;font-size:20px;color:{cc};'>
                {ca} ${abs(cdiff):,.2f} &nbsp;
                <span style='font-size:13px;'>({cdiff/g_ref*100:+.2f}% · spot ${g_ref:,.2f})</span>
              </div>
              <div style='font-size:11px;font-style:italic;color:{C['dim']};margin-top:6px;'>
                {len(vals)} / {len(MODEL_FILES)} models · {L['disclaimer']}
              </div>
            </div>""", unsafe_allow_html=True)

        # ── R² chart ──────────────────────────────────────────────────────────
        if r2_scores:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"<div class='section-label'>{L['r2_scores']}</div>", unsafe_allow_html=True)
            rn = list(r2_scores.keys()); rv = [r2_scores[n] for n in rn]
            bc_list = [C["gold_hi"] if n==best_model else C["gold_dark"] for n in rn]
            fig_r = go.Figure(go.Bar(x=rn,y=rv,marker_color=bc_list,
                text=[f"{v:.4f}" for v in rv],textposition="outside",
                textfont=dict(family=C["font_m"],size=10,color=C["text"])))
            fig_r.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor=C["card"],
                font=dict(color=C["text"],family=C["font_m"],size=10),
                yaxis=dict(range=[0,1.05],gridcolor=C["border2"],color=C["muted"]),
                xaxis=dict(gridcolor=C["border2"],color=C["muted"]),
                height=300,margin=dict(l=0,r=0,t=20,b=0))
            st.plotly_chart(fig_r, use_container_width=True)

        # ── Prophet chart ─────────────────────────────────────────────────────
        if "Prophet" in models_dict:
            st.markdown(f"<div class='section-label'>{L['prophet_forecast']}</div>", unsafe_allow_html=True)
            tdf = load_csv()
            if not tdf.empty and "Gold_Price" in tdf.columns:
                pm_obj = models_dict["Prophet"][0]
                future = pm_obj.make_future_dataframe(periods=30)
                fc     = pm_obj.predict(future)
                hist   = tdf[["Date","Gold_Price"]].dropna().tail(500)
                fig_f  = go.Figure()
                fig_f.add_trace(go.Scatter(x=hist["Date"],y=hist["Gold_Price"],
                    name="Historical",line=dict(color=C["gold"],width=2)))
                fig_f.add_trace(go.Scatter(x=fc["ds"],y=fc["yhat"],
                    name="Forecast",line=dict(color=C["blue"],width=2,dash="dash")))
                fig_f.add_traces([
                    go.Scatter(x=fc["ds"],y=fc["yhat_upper"],fill=None,mode="lines",
                        line=dict(color=C["blue"],width=0),showlegend=False),
                    go.Scatter(x=fc["ds"],y=fc["yhat_lower"],fill="tonexty",mode="lines",
                        line=dict(color=C["blue"],width=0),name="Confidence",
                        fillcolor=rgba(C["blue"],0.2)),
                ])
                fig_f.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor=C["card"],
                    font=dict(color=C["text"],family=C["font_m"],size=10),
                    yaxis_title="Gold Price (USD)",hovermode="x unified",height=420,
                    margin=dict(l=0,r=0,t=0,b=0),
                    legend=dict(bgcolor=C["card2"],bordercolor=C["border2"],font=dict(size=9)),
                    xaxis=dict(gridcolor=C["border2"],color=C["muted"]),
                    yaxis=dict(gridcolor=C["border2"],color=C["muted"]))
                st.plotly_chart(fig_f, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: DATA EXPLORER
# ═══════════════════════════════════════════════════════════════════════════════
elif nav == L["nav_data"]:
    use_ar = is_rtl()

    ph(L["data_explorer"], L["training_data"])

    # ── Session state for live data ───────────────────────────────────────────
    if "data_tdf"         not in st.session_state: st.session_state["data_tdf"]         = None
    if "data_last_update" not in st.session_state: st.session_state["data_last_update"]  = None
    if "data_n_new"       not in st.session_state: st.session_state["data_n_new"]        = 0
    if "data_updating"    not in st.session_state: st.session_state["data_updating"]     = False
    if "data_auto_update" not in st.session_state: st.session_state["data_auto_update"]  = True

    # ── Load base data (cached) ───────────────────────────────────────────────
    base_tdf = load_csv()

    # Use session-cached updated version if available, otherwise use base
    tdf = st.session_state["data_tdf"] if st.session_state["data_tdf"] is not None else base_tdf

    if tdf.empty:
        st.warning(L["not_found_data"])
    else:
        # ── Update status bar ─────────────────────────────────────────────────
        last_data_date  = tdf["Date"].max() if "Date" in tdf.columns else None
        today_dt        = pd.Timestamp(datetime.date.today())
        days_behind     = (today_dt - last_data_date).days if last_data_date is not None else 999
        is_stale        = days_behind > 1

        status_col   = C["red"] if days_behind > 7 else (C["gold"] if is_stale else C["green"])
        status_icon  = "🔴" if days_behind > 7 else ("🟡" if is_stale else "🟢")
        # Top control row
        ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([3,2,2,1])
        with ctrl1:
            st.markdown(f"""
            <div style='background:{C['card2']};border:1px solid {status_col}55;border-radius:5px;
                        padding:10px 16px;display:flex;align-items:center;gap:10px;'>
              <span style='font-size:20px;'>{status_icon}</span>
              <div>
                <div style='font-family:{C['font_m']},monospace;font-size:11px;color:{status_col};'>
                  {L['data_up_to']}
                  <b>{last_data_date.strftime('%d %b %Y') if last_data_date else '—'}</b>
                  {'  ·  متأخر ' + str(days_behind) + ' يوم' if is_stale else '  ·  محدّث ✓' if not is_stale else ''}
                  {'  ·  ' + str(days_behind) + ' days behind' if is_stale and not use_ar else '  ·  Up to date ✓' if not is_stale and not use_ar else ''}
                </div>

              </div>
            </div>""", unsafe_allow_html=True)

        with ctrl2:
            if st.button(
                ("🔄 تحديث البيانات الآن" if use_ar else "🔄 Update Now"),
                use_container_width=True, key="data_update_btn"
            ):
                st.session_state["data_updating"] = True
                st.rerun()

        with ctrl3:
            auto_on = st.toggle(
                ("تحديث تلقائي عند الفتح" if use_ar else "Auto-update on load"),
                value=st.session_state["data_auto_update"],
                key="data_auto_toggle"
            )
            st.session_state["data_auto_update"] = auto_on

        with ctrl4:
            if st.button(("↩ إعادة" if use_ar else "↩ Reset"),
                          use_container_width=True, key="data_reset_btn"):
                st.session_state["data_tdf"]          = None
                st.session_state["data_last_update"]   = None
                st.session_state["data_n_new"]         = 0
                load_csv.clear()   # clear cache
                st.rerun()

        # ── Auto-update on first load if stale and toggle is on ───────────────
        if (st.session_state["data_auto_update"] and is_stale
                and st.session_state["data_last_update"] is None
                and not st.session_state["data_updating"]):
            st.session_state["data_updating"] = True
            st.rerun()

        # ── Run update if flagged ─────────────────────────────────────────────
        if st.session_state["data_updating"]:
            with st.spinner(
                "⏳ جاري جلب البيانات الجديدة من Yahoo Finance..." if use_ar
                else "⏳ Fetching new rows from Yahoo Finance..."
            ):
                try:
                    new_rows = fetch_new_rows(last_data_date)
                    if not new_rows.empty:
                        updated_df, n_new = update_csv_with_new_rows(new_rows, tdf)
                        st.session_state["data_tdf"]        = updated_df
                        st.session_state["data_n_new"]      = n_new
                        st.session_state["data_last_update"]= datetime.datetime.now()
                        load_csv.clear()  # clear cache so rest of app also sees new data
                        tdf = updated_df
                        if n_new > 0:
                            st.success(
                                f"✅ {L['added']} {n_new} "
                                f"{L['new_rows_to']} "
                                f"{tdf['Date'].max().strftime('%d %b %Y')}"
                            )
                        else:
                            st.info(
                                "✅ البيانات محدّثة بالفعل ✓" if use_ar
                                else "✅ Data is already up to date ✓"
                            )
                    else:
                        st.session_state["data_last_update"] = datetime.datetime.now()
                        st.info(
                            "✅ لا توجد بيانات جديدة. البيانات محدّثة." if use_ar
                            else "✅ No new data available. Already up to date."
                        )
                except Exception as ex:
                    st.error(
                        f"{L['update_error']} {str(ex)[:200]}"
                    )
                finally:
                    st.session_state["data_updating"] = False

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Summary stat cards ────────────────────────────────────────────────
        s1,s2,s3,s4 = st.columns(4)
        for col,lbl,val in [
            (s1, L["total_rows"],  f"{len(tdf):,}"),
            (s2, L["features"],    str(len(tdf.columns))),
            (s3, L["from_date"],   tdf["Date"].min().strftime("%d %b %Y") if "Date" in tdf.columns else "—"),
            (s4, L["to_date"],     tdf["Date"].max().strftime("%d %b %Y") if "Date" in tdf.columns else "—"),
        ]:
            with col:
                st.markdown(f"""<div class='stat-card' style='text-align:center;'>
                  <div class='stat-label'>{lbl}</div>
                  <div class='stat-value'>{val}</div></div>""", unsafe_allow_html=True)

        # ── New rows highlight ────────────────────────────────────────────────
        if st.session_state["data_n_new"] > 0:
            recent_n = min(st.session_state["data_n_new"], 10)
            new_slice = tdf.tail(recent_n)
            with st.expander(
                f"🆕 {L['latest']} {recent_n} "
                f"{L['rows_added']}",
                expanded=True
            ):
                show_cols_new = ["Date","Gold_Price","SPX_Close","CPI","EFFR",
                                 "USD_Index","Oil_Price","Silver_Price","VIX","US10Y_Yield"]
                show_cols_new = [c for c in show_cols_new if c in new_slice.columns]
                st.dataframe(
                    new_slice[show_cols_new].style.format({
                        c: "{:,.4f}" for c in show_cols_new if c != "Date"
                    }, na_rep="—"),
                    use_container_width=True, hide_index=True
                )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Filters ───────────────────────────────────────────────────────────
        # Quick range buttons
        qr_cols = st.columns(7)
        _qr_labels = ["1M","3M","6M","1Y","3Y","5Y",
                      ("الكل" if use_ar else "All")]
        _qr_sel = st.session_state.get("data_qr", "1Y")
        for qi, (qcol, qlbl) in enumerate(zip(qr_cols, _qr_labels)):
            with qcol:
                _active = qlbl == _qr_sel
                if st.button(
                    qlbl, key=f"qr_{qi}",
                    type="primary" if _active else "secondary",
                    use_container_width=True
                ):
                    st.session_state["data_qr"] = qlbl
                    st.rerun()

        _qr = st.session_state.get("data_qr", "1Y")
        _today   = pd.Timestamp(datetime.date.today())
        _min_dt  = tdf["Date"].min() if "Date" in tdf.columns else pd.Timestamp("1990-01-01")
        _max_dt  = tdf["Date"].max() if "Date" in tdf.columns else _today
        _qr_map  = {
            "1M": _today - pd.DateOffset(months=1),
            "3M": _today - pd.DateOffset(months=3),
            "6M": _today - pd.DateOffset(months=6),
            "1Y": _today - pd.DateOffset(years=1),
            "3Y": _today - pd.DateOffset(years=3),
            "5Y": _today - pd.DateOffset(years=5),
        }
        _default_from = _qr_map.get(_qr, _min_dt)
        if isinstance(_default_from, pd.Timestamp) and _default_from < _min_dt:
            _default_from = _min_dt

        st.markdown("<br>", unsafe_allow_html=True)
        dc0, dc1, dc2, dc3, dc4 = st.columns([1.5,1.5,2,2,1])
        with dc0:
            date_from = st.date_input(
                L.get("date_from","From"),
                value=_default_from.date() if hasattr(_default_from,"date") else _default_from,
                min_value=_min_dt.date() if hasattr(_min_dt,"date") else _min_dt,
                max_value=_max_dt.date() if hasattr(_max_dt,"date") else _max_dt,
                key="data_date_from"
            )
        with dc1:
            date_to = st.date_input(
                L.get("date_to","To"),
                value=_max_dt.date() if hasattr(_max_dt,"date") else _max_dt,
                min_value=_min_dt.date() if hasattr(_min_dt,"date") else _min_dt,
                max_value=_max_dt.date() if hasattr(_max_dt,"date") else _max_dt,
                key="data_date_to"
            )
        with dc2:
            default_cols = ["Date","Gold_Price","SPX_Close","CPI","EFFR",
                            "USD_Index","Oil_Price","Silver_Price","VIX","US10Y_Yield"]
            default_cols = [c for c in default_cols if c in tdf.columns]
            cs = st.multiselect(L["columns_display"], list(tdf.columns), default=default_cols)
        with dc3:
            nr = st.number_input(L["rows_show"], min_value=10, max_value=5000, value=100, step=50)
        with dc4:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🔄", key="data_reset_filter", help=("إعادة تعيين" if use_ar else "Reset"),
                         use_container_width=True):
                st.session_state["data_qr"] = "1Y"
                st.rerun()

        filt = tdf.copy()
        if "Date" in tdf.columns:
            filt = filt[
                (filt["Date"] >= pd.Timestamp(date_from)) &
                (filt["Date"] <= pd.Timestamp(date_to))
            ]
        if cs:
            filt = filt[[c for c in cs if c in filt.columns]]

        st.markdown(
            f"<div class='section-label'>{len(filt):,} "
            f"{L['rows_filtered']} / {len(tdf):,} "
            f"{L['total_lc']}</div>",
            unsafe_allow_html=True
        )

        # Style: highlight NaN and newest rows
        display_df = filt.tail(int(nr))
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        # ── Download button ───────────────────────────────────────────────────
        csv_bytes = tdf.to_csv(index=False).encode("utf-8")
        st.download_button(
            label=("⬇️ تنزيل البيانات الكاملة CSV" if use_ar else "⬇️ Download Full Dataset CSV"),
            data=csv_bytes,
            file_name=f"mansa_gold_data_{datetime.date.today()}.csv",
            mime="text/csv",
            use_container_width=False,
        )

        # ── Gold price history chart with new data highlighted ────────────────
        if "Date" in tdf.columns and "Gold_Price" in tdf.columns:
            st.markdown(f"<div class='section-label'>{L['gold_history']}</div>", unsafe_allow_html=True)

            fig_h = go.Figure()

            # Historical line (up to the CSV start date before updates)
            hist_end = base_tdf["Date"].max() if not base_tdf.empty else tdf["Date"].max()
            hist_part = tdf[tdf["Date"] <= hist_end]
            new_part  = tdf[tdf["Date"] > hist_end]

            fig_h.add_trace(go.Scatter(
                x=hist_part["Date"], y=hist_part["Gold_Price"],
                name=("بيانات تاريخية" if use_ar else "Historical"),
                line=dict(color=C["gold"], width=1.5),
                fill="tozeroy", fillcolor=rgba(C["gold_dark"], 0.13)
            ))

            # Newly fetched rows in a brighter highlighted colour
            if not new_part.empty:
                fig_h.add_trace(go.Scatter(
                    x=new_part["Date"], y=new_part["Gold_Price"],
                    name=("بيانات جديدة ✨" if use_ar else "New data ✨"),
                    line=dict(color=C["gold_hi"], width=2.5),
                    mode="lines+markers",
                    marker=dict(size=6, color=C["gold_hi"], symbol="circle"),
                ))

            fig_h.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=C["card"],
                font=dict(color=C["text"], family=C["font_m"], size=10),
                yaxis_title="USD/oz", height=360,
                margin=dict(l=0,r=0,t=0,b=0), hovermode="x unified",
                legend=dict(bgcolor=C["card2"], bordercolor=C["border2"], font=dict(size=9)),
                xaxis=dict(gridcolor=C["border2"], color=C["muted"]),
                yaxis=dict(gridcolor=C["border2"], color=C["muted"]),
            )
            st.plotly_chart(fig_h, use_container_width=True)

        # ── Column sparklines ─────────────────────────────────────────────────
        spark_cols = ["Gold_Price","SPX_Close","Oil_Price","Silver_Price","VIX","USD_Index"]
        spark_cols = [c for c in spark_cols if c in tdf.columns]
        if spark_cols:
            st.markdown(
                f"<div class='section-label'>{L['sparklines']}</div>",
                unsafe_allow_html=True
            )
            recent_yr = tdf[tdf["Date"] >= (tdf["Date"].max() - pd.Timedelta(days=365))]
            sp_cols_ui = st.columns(len(spark_cols))
            for spc, scol in zip(spark_cols, sp_cols_ui):
                if recent_yr[spc].dropna().empty:
                    continue
                fig_sp = go.Figure(go.Scatter(
                    x=recent_yr["Date"], y=recent_yr[spc],
                    line=dict(color=C["gold"], width=1.5),
                    fill="tozeroy", fillcolor=rgba(C["gold_dark"], 0.15),
                ))
                last_v = recent_yr[spc].dropna().iloc[-1]
                first_v = recent_yr[spc].dropna().iloc[0]
                chg = (last_v - first_v) / first_v * 100 if first_v else 0
                chg_col = C["green"] if chg >= 0 else C["red"]
                fig_sp.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=C["card"],
                    height=90, margin=dict(l=0,r=0,t=0,b=0),
                    xaxis=dict(visible=False), yaxis=dict(visible=False),
                    showlegend=False,
                )
                with scol:
                    st.markdown(f"""
                    <div style='text-align:center;margin-bottom:2px;'>
                      <div class='stat-label'>{spc.replace("_"," ")}</div>
                      <div style='font-family:{C['font_m']},monospace;font-size:13px;color:{C['gold_pale']};'>
                        {last_v:,.2f}
                      </div>
                      <div style='font-family:{C['font_m']},monospace;font-size:10px;color:{chg_col};'>
                        {'+' if chg>=0 else ''}{chg:.1f}% YoY
                      </div>
                    </div>""", unsafe_allow_html=True)
                    st.plotly_chart(fig_sp, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: GOLD ADVISOR  (fully rule-based, zero API needed)
# ═══════════════════════════════════════════════════════════════════════════════
elif nav == L["nav_advisor"]:
    use_ar = is_rtl()

    ph(L.get("chatbot_title","💬 مستشار الذهب الذكي"),
       ("مستشار ذكي يعمل بدون إنترنت أو مفاتيح · يجيب بناءً على بيانات السوق الحية"
        if use_ar else
        "Smart advisor — works offline, no API key needed · answers based on live market data"))

    # ── Session state ─────────────────────────────────────────────────────────
    if "chat_history"    not in st.session_state: st.session_state["chat_history"]    = []
    if "chat_init_done"  not in st.session_state: st.session_state["chat_init_done"]  = False

    # ══════════════════════════════════════════════════════════════════════════
    # RULE ENGINE — generates rich answers from live data, no LLM needed
    # ══════════════════════════════════════════════════════════════════════════
    @st.cache_data(ttl=CACHE_TTL_HISTORY)
    def get_ta_advisor():
        """Compute a full suite of technical indicators for the advisor page.
        
            Returns
            -------
            dict
                Keys: rsi, ma50, ma200, macd, macd_sig, macd_hist, macd_cross_bull,
                macd_cross_bear, golden_cross, death_cross, recent_golden,
                recent_death, bb_pos, stoch_k, stoch_d.
            """
        df = fetch_history("1y","GC=F")
        if df.empty: return {}
        cl = find_col(df, ["Close","Close_GC=F"])
        if not cl: return {}
        c = df[cl].dropna()
        def safe(fn):
            try: return fn()
            except Exception:
                _log.debug("Suppressed", exc_info=True)
                return None
        ma20  = safe(lambda: float(c.rolling(20).mean().iloc[-1]))
        ma50  = safe(lambda: float(c.rolling(50).mean().iloc[-1]))
        ma200 = safe(lambda: float(c.rolling(200).mean().iloc[-1]))
        d = c.diff()
        g = d.clip(lower=0).rolling(14).mean()
        l = (-d.clip(upper=0)).rolling(14).mean()
        rsi   = safe(lambda: float(100-(100/(1+g/l.replace(0,1e-9))).iloc[-1]))
        ema12 = c.ewm(span=12,adjust=False).mean()
        ema26 = c.ewm(span=26,adjust=False).mean()
        macd  = safe(lambda: float((ema12-ema26).iloc[-1]))
        sig   = safe(lambda: float((ema12-ema26).ewm(span=9,adjust=False).mean().iloc[-1]))
        bm    = c.rolling(20).mean()
        bs    = c.rolling(20).std()
        bb_u  = safe(lambda: float((bm+2*bs).iloc[-1]))
        bb_l  = safe(lambda: float((bm-2*bs).iloc[-1]))
        hi52  = safe(lambda: float(c.tail(252).max()))
        lo52  = safe(lambda: float(c.tail(252).min()))
        atr   = safe(lambda: float(c.diff().abs().rolling(14).mean().iloc[-1]))
        roc10 = safe(lambda: float(c.pct_change(10).iloc[-1]*100))
        return dict(ma20=ma20,ma50=ma50,ma200=ma200,rsi=rsi,macd=macd,sig=sig,
                    bb_u=bb_u,bb_l=bb_l,hi52=hi52,lo52=lo52,atr=atr,roc10=roc10)

    def advisor_answer(question: str, ta: dict) -> str:
        """
        Pure Python rule engine — no LLM.
        Maps question keywords to rich, data-driven answers in Arabic or English.
        """
        q = question.lower().strip()
        gold   = g_ref
        ch_pct = live["gold"]["pct"]
        dxy    = live["dxy"]["price"]
        dxy_ch = live["dxy"]["pct"]
        vix    = live["vix"]["price"]
        spx_ch = live["spx"]["pct"]
        slv    = live["silver"]["price"]
        us10y  = live["us10y"]["price"]
        rsi    = ta.get("rsi") or 50
        ma50   = ta.get("ma50") or gold
        ma200  = ta.get("ma200") or gold
        macd   = ta.get("macd") or 0
        sig    = ta.get("sig") or 0
        bb_u   = ta.get("bb_u") or gold*1.02
        bb_l   = ta.get("bb_l") or gold*0.98
        hi52   = ta.get("hi52") or gold
        lo52   = ta.get("lo52") or gold
        atr    = ta.get("atr") or gold*0.01

        # Determine overall signal
        bull_pts = 0; bear_pts = 0
        if gold > ma50:  bull_pts += 1
        else:            bear_pts += 1
        if gold > ma200: bull_pts += 1
        else:            bear_pts += 1
        if rsi < 40:     bull_pts += 1
        elif rsi > 65:   bear_pts += 1
        if macd > sig:   bull_pts += 1
        else:            bear_pts += 1
        if dxy_ch < -0.3:bull_pts += 1
        elif dxy_ch > 0.3:bear_pts += 1
        if vix > 22:     bull_pts += 1
        if spx_ch < -1:  bull_pts += 1
        elif spx_ch > 1: bear_pts += 1

        total_pts  = bull_pts + bear_pts or 1
        bull_pct_v = bull_pts / total_pts * 100
        if bull_pct_v >= 65:  overall = ("STRONG BUY",  "شراء قوي")
        elif bull_pct_v >= 50:overall = ("BUY",         "شراء")
        elif (bear_pts/total_pts*100) >= 65: overall = ("STRONG SELL","بيع قوي")
        elif (bear_pts/total_pts*100) >= 50: overall = ("SELL",       "بيع")
        else:                 overall = ("HOLD",        "احتفاظ")

        sl  = round(gold - 1.5*atr, 2)
        tp1 = round(gold + 1.5*atr, 2)
        tp2 = round(gold + 3.0*atr, 2)
        pm_p, _ = mkt_price(gold, PM, st.session_state["purity"])
        gsr = round(gold/slv, 1) if slv else 0

        # ── keywords → answer ───────────────────────────────────────────────
        # BUY / SELL / SHOULD I
        if any(w in q for w in ["buy","شراء","اشتري","purchase","should i","هل يجب","هل أشتري","متى أشتري","when to buy"]):
            if use_ar:
                return (
                    f"🔍 **تحليل قرار الشراء الآن**\n\n"
                    f"السعر الحالي: **${gold:,.2f}** | الإشارة الكلية: **{overall[1]}**\n\n"
                    f"📊 **المؤشرات الفنية:**\n"
                    f"• RSI(14) = {rsi:.1f} → {'منطقة شراء 🟢' if rsi<30 else 'منطقة ذروة بيع 🔴' if rsi>65 else 'محايد 🟡'}\n"
                    f"• السعر {'فوق' if gold>ma50 else 'تحت'} MA50 (${ma50:,.0f}) → {'اتجاه صاعد 🟢' if gold>ma50 else 'اتجاه هابط 🔴'}\n"
                    f"• السعر {'فوق' if gold>ma200 else 'تحت'} MA200 (${ma200:,.0f}) → {'هيكل صاعد طويل الأمد 🟢' if gold>ma200 else 'هيكل هابط 🔴'}\n"
                    f"• MACD {'فوق' if macd>sig else 'تحت'} خط الإشارة → {'زخم صاعد 🟢' if macd>sig else 'زخم هابط 🔴'}\n"
                    f"• DXY تغير {dxy_ch:+.2f}% → {'يدعم الذهب 🟢' if dxy_ch<-0.3 else 'يضغط على الذهب 🔴' if dxy_ch>0.3 else 'محايد'}\n"
                    f"• VIX = {vix:.1f} → {'خوف مرتفع = طلب على الذهب 🟢' if vix>22 else 'هدوء السوق 🟡'}\n\n"
                    f"🎯 **مستويات الصفقة:**\n"
                    f"• نقطة الدخول: ${gold:,.2f}\n"
                    f"• وقف الخسارة: ${sl:,.2f} (1.5× ATR أسفل)\n"
                    f"• جني أرباح 1: ${tp1:,.2f} | جني أرباح 2: ${tp2:,.2f}\n"
                    f"• نسبة المخاطرة/العائد: 1:1 إلى 1:2\n\n"
                    f"{'✅ الإشارة تشير إلى فرصة شراء — تأكد من إدارة المخاطر.' if 'BUY' in overall[0] else '⚠️ الإشارة محايدة أو هابطة — انتظر تأكيداً أقوى قبل الشراء.'}\n\n"
                    f"⚠️ هذا تحليل تعليمي وليس نصيحة مالية."
                )
            else:
                return (
                    f"🔍 **Buy Decision Analysis**\n\n"
                    f"Current price: **${gold:,.2f}** | Overall signal: **{overall[0]}**\n\n"
                    f"📊 **Technical Indicators:**\n"
                    f"• RSI(14) = {rsi:.1f} → {'Oversold — buy zone 🟢' if rsi<30 else 'Overbought 🔴' if rsi>65 else 'Neutral 🟡'}\n"
                    f"• Price {'above' if gold>ma50 else 'below'} MA50 (${ma50:,.0f}) → {'Uptrend 🟢' if gold>ma50 else 'Downtrend 🔴'}\n"
                    f"• Price {'above' if gold>ma200 else 'below'} MA200 (${ma200:,.0f}) → {'Bullish long-term structure 🟢' if gold>ma200 else 'Bearish structure 🔴'}\n"
                    f"• MACD {'above' if macd>sig else 'below'} signal → {'Bullish momentum 🟢' if macd>sig else 'Bearish momentum 🔴'}\n"
                    f"• DXY change {dxy_ch:+.2f}% → {'Supports gold 🟢' if dxy_ch<-0.3 else 'Headwind for gold 🔴' if dxy_ch>0.3 else 'Neutral'}\n"
                    f"• VIX = {vix:.1f} → {'High fear = safe haven demand 🟢' if vix>22 else 'Calm markets 🟡'}\n\n"
                    f"🎯 **Trade Levels:**\n"
                    f"• Entry: ${gold:,.2f}\n"
                    f"• Stop Loss: ${sl:,.2f} (1.5× ATR below)\n"
                    f"• Take Profit 1: ${tp1:,.2f} | TP2: ${tp2:,.2f}\n"
                    f"• Risk/Reward: 1:1 to 1:2\n\n"
                    f"{'✅ Signal favors a buy — manage your risk carefully.' if 'BUY' in overall[0] else '⚠️ Signal is neutral/bearish — wait for a stronger confirmation before buying.'}\n\n"
                    f"⚠️ Educational analysis only — not financial advice."
                )

        # SELL / SHOULD I SELL
        elif any(w in q for w in ["sell","بيع","أبيع","هل أبيع","when to sell","متى أبيع"]):
            if use_ar:
                return (
                    f"📉 **تحليل قرار البيع**\n\n"
                    f"السعر الحالي: **${gold:,.2f}** | الإشارة: **{overall[1]}**\n\n"
                    f"• RSI = {rsi:.1f} → {'ذروة شراء — مناسب للبيع 🔴' if rsi>70 else 'لم يصل لذروة الشراء بعد 🟢'}\n"
                    f"• السعر من أعلى 52 أسبوع (${hi52:,.0f}): {(hi52-gold)/hi52*100:.1f}% أسفل القمة\n"
                    f"• نطاق بولينجر العلوي: ${bb_u:,.2f} → {'السعر قريب من المقاومة 🔴' if gold > bb_u*0.98 else 'مساحة للصعود 🟢'}\n"
                    f"• DXY تغير {dxy_ch:+.2f}% → {'دولار قوي = ضغط على الذهب 🔴' if dxy_ch>0.4 else 'دولار ضعيف = دعم للذهب 🟢'}\n\n"
                    f"🎯 **مستويات البيع:**\n"
                    f"• مقاومة قريبة: ${bb_u:,.2f} (بولينجر العلوي)\n"
                    f"• أعلى 52 أسبوع: ${hi52:,.0f}\n"
                    f"• نقطة وقف الربح: ${tp1:,.2f}\n\n"
                    f"{'⚠️ مؤشرات تدل على ضغط بيعي — فكر في الخروج أو تقليل المركز.' if 'SELL' in overall[0] else '🟢 لا تزال الإشارة صاعدة — قد يكون من السابق للبيع الآن.'}\n\n"
                    f"⚠️ هذا تحليل تعليمي فقط."
                )
            else:
                return (
                    f"📉 **Sell Decision Analysis**\n\n"
                    f"Current price: **${gold:,.2f}** | Signal: **{overall[0]}**\n\n"
                    f"• RSI = {rsi:.1f} → {'Overbought — consider selling 🔴' if rsi>70 else 'Not overbought yet 🟢'}\n"
                    f"• Distance from 52W high (${hi52:,.0f}): {(hi52-gold)/hi52*100:.1f}% below peak\n"
                    f"• Bollinger Upper Band: ${bb_u:,.2f} → {'Near resistance 🔴' if gold > bb_u*0.98 else 'Room to run 🟢'}\n"
                    f"• DXY change {dxy_ch:+.2f}% → {'Strong dollar = headwind 🔴' if dxy_ch>0.4 else 'Weak dollar = gold support 🟢'}\n\n"
                    f"🎯 **Key Sell Levels:**\n"
                    f"• Near resistance: ${bb_u:,.2f} (Bollinger Upper)\n"
                    f"• 52-week high: ${hi52:,.0f}\n"
                    f"• Profit target: ${tp1:,.2f}\n\n"
                    f"{'⚠️ Bearish signals present — consider reducing position.' if 'SELL' in overall[0] else '🟢 Signal still bullish — may be too early to sell.'}\n\n"
                    f"⚠️ Educational only — not financial advice."
                )

        # PRICE / CURRENT
        elif any(w in q for w in ["price","سعر","كم سعر","كم الذهب","gold price","current","الآن","اليوم","today","now"]):
            if use_ar:
                return (
                    f"💰 **أسعار الذهب الآن — {datetime.datetime.now().strftime('%d %b %Y %H:%M')}**\n\n"
                    f"**السعر الفوري العالمي (24 قيراط):**\n"
                    f"• ${gold:,.2f} دولار / أونصة تروي\n"
                    f"• التغير اليومي: {live['gold']['change']:+.2f}$ ({ch_pct:+.2f}%)\n\n"
                    f"**السعر في سوقك الرئيسي ({st.session_state['primary_mkt']}):**\n"
                    f"• {pm_p:,.3f} {PM['currency']} / {PM['unit_label']}\n"
                    f"• العيار المحدد: {st.session_state['purity'].split('—')[0].strip()}\n\n"
                    f"**أسعار مرجعية (غرام 24 قيراط):**\n"
                    f"• غرام: ${gold/31.1035:,.4f}\n"
                    f"• مثقال (تولة): ${gold/2.6667:,.3f}\n"
                    f"• كيلوغرام: ${gold*32.1507:,.2f}\n\n"
                    f"📊 **السياق:** أعلى 52 أسبوع ${hi52:,.0f} · أدنى 52 أسبوع ${lo52:,.0f}"
                )
            else:
                return (
                    f"💰 **Gold Prices Now — {datetime.datetime.now().strftime('%d %b %Y %H:%M')}**\n\n"
                    f"**Global Spot (24K):**\n"
                    f"• ${gold:,.2f} USD / troy oz\n"
                    f"• Daily change: {live['gold']['change']:+.2f}$ ({ch_pct:+.2f}%)\n\n"
                    f"**Your Primary Market ({st.session_state['primary_mkt']}):**\n"
                    f"• {pm_p:,.3f} {PM['currency']} / {PM['unit_label']}\n"
                    f"• Purity: {st.session_state['purity'].split('—')[0].strip()}\n\n"
                    f"**Reference prices (24K):**\n"
                    f"• Per gram: ${gold/31.1035:,.4f}\n"
                    f"• Per tola: ${gold/2.6667:,.3f}\n"
                    f"• Per kg: ${gold*32.1507:,.2f}\n\n"
                    f"📊 **Context:** 52W High ${hi52:,.0f} · 52W Low ${lo52:,.0f}"
                )

        # RSI
        elif any(w in q for w in ["rsi","مؤشر القوة","قوة نسبية","relative strength"]):
            trend = ("منطقة شراء" if rsi<30 else "ذروة شراء" if rsi>70 else "محايد") if use_ar else \
                    ("Oversold" if rsi<30 else "Overbought" if rsi>70 else "Neutral")
            if use_ar:
                return (f"📐 **مؤشر القوة النسبية RSI (14)**\n\nالقيمة الحالية: **{rsi:.1f}**\nالحالة: **{trend}**\n\n"
                        f"• RSI < 30: ذروة بيع — إشارة شراء قوية\n• RSI 30–50: منطقة ضعف — مراقبة\n"
                        f"• RSI 50–70: منطقة صحية — اتجاه صاعد\n• RSI > 70: ذروة شراء — تحذير من التصحيح\n\n"
                        f"للذهب الآن عند {rsi:.1f}: {'فرصة شراء محتملة 🟢' if rsi<30 else 'تحذير من التصحيح 🔴' if rsi>70 else 'الاتجاه صحي 🟡'}")
            else:
                return (f"📐 **RSI (14) Indicator**\n\nCurrent value: **{rsi:.1f}**\nStatus: **{trend}**\n\n"
                        f"• RSI < 30: Oversold — strong buy signal\n• RSI 30–50: Weak zone — watch\n"
                        f"• RSI 50–70: Healthy — bullish trend\n• RSI > 70: Overbought — correction risk\n\n"
                        f"Gold at RSI {rsi:.1f}: {'Potential buy opportunity 🟢' if rsi<30 else 'Overbought — correction risk 🔴' if rsi>70 else 'Healthy trend 🟡'}")

        # MACD
        elif any(w in q for w in ["macd","ماكد"]):
            cross = ("إيجابي — زخم صاعد 🟢" if macd>sig else "سلبي — زخم هابط 🔴") if use_ar else \
                    ("Positive — bullish momentum 🟢" if macd>sig else "Negative — bearish momentum 🔴")
            if use_ar:
                return (f"📊 **مؤشر MACD**\n\n• MACD: {macd:+.2f}\n• Signal: {sig:+.2f}\n• Histogram: {macd-sig:+.2f}\n\nالحالة: **{cross}**\n\n"
                        f"{'MACD تجاوز خط الإشارة للأعلى = إشارة شراء. انتظر تأكيداً فوق الخط الصفري.' if macd>sig else 'MACD أسفل خط الإشارة = ضغط هبوطي. احذر من مراكز الشراء الجديدة.'}")
            else:
                return (f"📊 **MACD Indicator**\n\n• MACD: {macd:+.2f}\n• Signal: {sig:+.2f}\n• Histogram: {macd-sig:+.2f}\n\nStatus: **{cross}**\n\n"
                        f"{'MACD crossed above signal = buy signal. Wait for confirmation above zero line.' if macd>sig else 'MACD below signal line = bearish pressure. Be cautious with new longs.'}")

        # SUPPORT / RESISTANCE
        elif any(w in q for w in ["support","resistance","مقاومة","دعم","level","مستوى"]):
            if use_ar:
                return (f"🎯 **مستويات الدعم والمقاومة**\n\n"
                        f"**مقاومات:**\n• المقاومة 1: ${bb_u:,.2f} (بولينجر العلوي)\n• المقاومة 2: ${hi52:,.0f} (أعلى 52 أسبوع)\n• MA200: ${ma200:,.0f}\n\n"
                        f"**دعومات:**\n• الدعم 1: ${bb_l:,.2f} (بولينجر السفلي)\n• الدعم 2: ${lo52:,.0f} (أدنى 52 أسبوع)\n• MA50: ${ma50:,.0f}\n\n"
                        f"**ATR (التقلب اليومي):** ${atr:,.2f}\n"
                        f"• وقف الخسارة المقترح: ${sl:,.2f} (1.5× ATR)\n"
                        f"• جني الأرباح 1: ${tp1:,.2f} | 2: ${tp2:,.2f}")
            else:
                return (f"🎯 **Support & Resistance Levels**\n\n"
                        f"**Resistances:**\n• R1: ${bb_u:,.2f} (Bollinger Upper)\n• R2: ${hi52:,.0f} (52W High)\n• MA200: ${ma200:,.0f}\n\n"
                        f"**Supports:**\n• S1: ${bb_l:,.2f} (Bollinger Lower)\n• S2: ${lo52:,.0f} (52W Low)\n• MA50: ${ma50:,.0f}\n\n"
                        f"**ATR (daily range):** ${atr:,.2f}\n"
                        f"• Suggested stop loss: ${sl:,.2f} (1.5× ATR)\n"
                        f"• Take profit 1: ${tp1:,.2f} | TP2: ${tp2:,.2f}")

        # VIX / FEAR
        elif any(w in q for w in ["vix","fear","خوف","تقلب","volatility"]):
            lvl = ("مرتفع — ذعر" if vix>30 else "مرتفع — قلق" if vix>22 else "منخفض — هدوء") if use_ar else \
                  ("High — panic" if vix>30 else "Elevated — fear" if vix>22 else "Low — calm")
            if use_ar:
                return (f"😨 **مؤشر الخوف VIX**\n\nالقيمة: **{vix:.1f}** | الحالة: **{lvl}**\n\n"
                        f"• VIX < 15: سوق هادئ — طلب منخفض على الذهب كملاذ آمن\n"
                        f"• VIX 15–25: قلق معتدل — بعض الطلب على الذهب\n"
                        f"• VIX > 25: خوف مرتفع — الذهب يتفوق عادةً\n"
                        f"• VIX > 40: ذعر تام — أداء الذهب التاريخي ممتاز\n\n"
                        f"الآن: {'الذهب في وضع قوي كملاذ آمن 🟢' if vix>22 else 'أسواق هادئة — الذهب أقل جاذبية كملاذ 🟡'}")
            else:
                return (f"😨 **VIX Fear Index**\n\nValue: **{vix:.1f}** | Level: **{lvl}**\n\n"
                        f"• VIX < 15: Calm — low safe-haven demand for gold\n"
                        f"• VIX 15–25: Moderate fear — some gold demand\n"
                        f"• VIX > 25: High fear — gold typically outperforms\n"
                        f"• VIX > 40: Panic — gold historically excels\n\n"
                        f"Currently: {'Gold well positioned as safe haven 🟢' if vix>22 else 'Calm markets — less urgency for gold 🟡'}")

        # DXY / DOLLAR
        elif any(w in q for w in ["dxy","dollar","دولار","usd","الدولار"]):
            if use_ar:
                return (f"💵 **مؤشر الدولار الأمريكي (DXY)**\n\nالقيمة: **{dxy:.2f}** | التغير: {dxy_ch:+.2f}%\n\n"
                        f"الذهب والدولار مرتبطان عكسياً بنسبة ~80%.\n\n"
                        f"• دولار ضعيف (DXY ينخفض) → الذهب يرتفع 🟢\n• دولار قوي (DXY يرتفع) → ضغط على الذهب 🔴\n\n"
                        f"اليوم: DXY {dxy_ch:+.2f}% → {'يدعم الذهب 🟢' if dxy_ch<-0.3 else 'يضغط على الذهب 🔴' if dxy_ch>0.3 else 'محايد 🟡'}\n\n"
                        f"مستويات رئيسية للدولار: دعم {dxy*0.97:.1f} | مقاومة {dxy*1.03:.1f}")
            else:
                return (f"💵 **US Dollar Index (DXY)**\n\nValue: **{dxy:.2f}** | Change: {dxy_ch:+.2f}%\n\n"
                        f"Gold and the dollar have ~80% inverse correlation.\n\n"
                        f"• Weak dollar (DXY falling) → Gold rises 🟢\n• Strong dollar (DXY rising) → Gold under pressure 🔴\n\n"
                        f"Today: DXY {dxy_ch:+.2f}% → {'Supports gold 🟢' if dxy_ch<-0.3 else 'Headwind for gold 🔴' if dxy_ch>0.3 else 'Neutral 🟡'}\n\n"
                        f"Key DXY levels: Support {dxy*0.97:.1f} | Resistance {dxy*1.03:.1f}")

        # INFLATION / CPI
        elif any(w in q for w in ["inflation","cpi","تضخم","أسعار المستهلك"]):
            if use_ar:
                return (f"🧾 **التضخم وعلاقته بالذهب**\n\n"
                        f"الذهب هو التحوط الكلاسيكي ضد التضخم منذ آلاف السنين.\n\n"
                        f"• تضخم مرتفع (CPI > 3%) → الذهب يرتفع تاريخياً\n"
                        f"• معدلات فائدة حقيقية سلبية → أفضل بيئة للذهب\n"
                        f"• المعدل الحقيقي = عائد 10Y - CPI\n\n"
                        f"السياق الحالي:\n"
                        f"• عائد 10Y: {us10y:.2f}%\n"
                        f"• إذا كان التضخم أعلى من {us10y:.1f}%، فالمعدل الحقيقي سلبي = صعودي للذهب 🟢")
            else:
                return (f"🧾 **Inflation & Gold**\n\n"
                        f"Gold is the classic inflation hedge — trusted for thousands of years.\n\n"
                        f"• High inflation (CPI > 3%) → Gold historically outperforms\n"
                        f"• Negative real rates → Best environment for gold\n"
                        f"• Real rate = 10Y yield - CPI\n\n"
                        f"Current context:\n"
                        f"• US 10Y yield: {us10y:.2f}%\n"
                        f"• If inflation > {us10y:.1f}%, real rate is negative = bullish for gold 🟢")

        # GOLD-SILVER RATIO
        elif any(w in q for w in ["silver","فضة","gold silver","ratio","نسبة"]):
            if use_ar:
                return (f"⚖️ **نسبة الذهب / الفضة (GSR)**\n\nالذهب: ${gold:,.2f} | الفضة: ${slv:,.2f}\nالنسبة الحالية: **{gsr:.1f}**\n\n"
                        f"• GSR < 60: الذهب رخيص نسبياً مقابل الفضة\n• GSR 60–80: نطاق عادي\n"
                        f"• GSR > 80: الذهب غالٍ نسبياً / الفضة رخيصة\n\n"
                        f"الآن {gsr:.1f}: {'الذهب رخيص نسبياً — ميزة للشراء 🟢' if gsr<65 else 'الذهب غالٍ نسبياً مقابل الفضة 🟡' if gsr>80 else 'في النطاق الطبيعي'}")
            else:
                return (f"⚖️ **Gold/Silver Ratio (GSR)**\n\nGold: ${gold:,.2f} | Silver: ${slv:,.2f}\nCurrent ratio: **{gsr:.1f}**\n\n"
                        f"• GSR < 60: Gold cheap vs silver\n• GSR 60–80: Normal range\n"
                        f"• GSR > 80: Gold expensive / silver cheap\n\n"
                        f"At {gsr:.1f}: {'Gold relatively cheap — buying advantage 🟢' if gsr<65 else 'Gold expensive vs silver 🟡' if gsr>80 else 'In normal range'}")

        # ISLAMIC / HALAL
        elif any(w in q for w in ["halal","حلال","islamic","إسلام","ربا","riba","شرع","sharia"]):
            if use_ar:
                _static_ar = ("☽ **الذهب في الإسلام والتمويل الإسلامي**\n\n"
                              "✅ **مباح شرعاً:**\n• شراء وبيع الذهب المادي (مسكوك، سبائك) بالقبض الفوري\n"
                              "• حسابات الذهب في البنوك الإسلامية (قبض فوري)\n• صناديق الذهب المدعومة بذهب مادي (ETF Shari'a-compliant)\n\n")
                return (_static_ar +
                        "⚠️ **يحتاج إلى دراسة شرعية:**\n• العقود الآجلة للذهب (Futures) — قد تنطوي على ربا\n"
                        "• عقود CFD — تداول بدون امتلاك فعلي، قد لا يكون حلالاً\n"
                        "• الذهب الورقي (ETF غير مدعوم بذهب مادي)\n\n"
                        "📌 **النصيحة:** استشر عالماً متخصصاً في فقه المعاملات المالية قبل تداول أي أداة مالية متعلقة بالذهب.")
            else:
                _static_en = ("☽ **Gold in Islamic Finance**\n\n"
                              "✅ **Generally permissible:**\n• Physical gold (coins, bars) with immediate delivery\n"
                              "• Islamic bank gold savings accounts (immediate possession)\n• Shari'a-compliant gold ETFs backed by physical gold\n\n")
                return (_static_en +
                        "⚠️ **Requires scholarly review:**\n• Gold futures — may involve riba (interest)\n"
                        "• CFDs — trading without ownership, may not be halal\n"
                        "• Paper gold ETFs not backed by physical gold\n\n"
                        "📌 Consult a qualified Islamic finance scholar before trading any gold derivative.")

        # SIGNAL / OVERALL
        elif any(w in q for w in ["signal","إشارة","overall","trend","اتجاه","تحليل","analysis","outlook","توقع"]):
            if use_ar:
                return (
                    f"📊 **التحليل الشامل للذهب الآن**\n\n"
                    f"الإشارة الكلية: **{overall[1]}** ({bull_pts} صاعد / {bear_pts} هابط)\n"
                    f"السعر: **${gold:,.2f}** ({ch_pct:+.2f}% اليوم)\n\n"
                    f"**المؤشرات الفنية:**\n"
                    f"• RSI {rsi:.1f} → {'ذروة بيع 🟢' if rsi<30 else 'ذروة شراء 🔴' if rsi>70 else 'محايد'}\n"
                    f"• MACD {'صاعد 🟢' if macd>sig else 'هابط 🔴'} ({macd:+.2f})\n"
                    f"• MA50 ${ma50:,.0f} — السعر {'فوقه 🟢' if gold>ma50 else 'تحته 🔴'}\n"
                    f"• MA200 ${ma200:,.0f} — السعر {'فوقه 🟢' if gold>ma200 else 'تحته 🔴'}\n\n"
                    f"**الاقتصاد الكلي:**\n"
                    f"• DXY {dxy:.1f} ({dxy_ch:+.2f}%) {'🟢' if dxy_ch<-0.3 else '🔴' if dxy_ch>0.3 else '🟡'}\n"
                    f"• VIX {vix:.1f} {'— خوف مرتفع 🟢' if vix>22 else '— هدوء 🟡'}\n"
                    f"• S&P500 {spx_ch:+.2f}% {'— هبوط يدعم الذهب 🟢' if spx_ch<-1 else '— صعود يقلل الطلب 🔴' if spx_ch>1 else '🟡'}\n\n"
                    f"**مستويات رئيسية:**\n"
                    f"• دعم: ${bb_l:,.0f} | ${ma50:,.0f}\n"
                    f"• مقاومة: ${bb_u:,.0f} | ${hi52:,.0f}\n"
                    f"• وقف الخسارة: ${sl:,.0f} | TP1: ${tp1:,.0f} | TP2: ${tp2:,.0f}\n\n"
                    f"⚠️ تحليل تعليمي فقط."
                )
            else:
                return (
                    f"📊 **Comprehensive Gold Analysis**\n\n"
                    f"Overall signal: **{overall[0]}** ({bull_pts} bullish / {bear_pts} bearish)\n"
                    f"Price: **${gold:,.2f}** ({ch_pct:+.2f}% today)\n\n"
                    f"**Technical Indicators:**\n"
                    f"• RSI {rsi:.1f} → {'Oversold 🟢' if rsi<30 else 'Overbought 🔴' if rsi>70 else 'Neutral'}\n"
                    f"• MACD {'bullish 🟢' if macd>sig else 'bearish 🔴'} ({macd:+.2f})\n"
                    f"• MA50 ${ma50:,.0f} — price {'above 🟢' if gold>ma50 else 'below 🔴'}\n"
                    f"• MA200 ${ma200:,.0f} — price {'above 🟢' if gold>ma200 else 'below 🔴'}\n\n"
                    f"**Macro:**\n"
                    f"• DXY {dxy:.1f} ({dxy_ch:+.2f}%) {'🟢' if dxy_ch<-0.3 else '🔴' if dxy_ch>0.3 else '🟡'}\n"
                    f"• VIX {vix:.1f} {'— elevated fear 🟢' if vix>22 else '— calm 🟡'}\n"
                    f"• S&P500 {spx_ch:+.2f}% {'— falling, gold demand up 🟢' if spx_ch<-1 else '— rising, less gold demand 🔴' if spx_ch>1 else '🟡'}\n\n"
                    f"**Key levels:**\n"
                    f"• Support: ${bb_l:,.0f} | ${ma50:,.0f}\n"
                    f"• Resistance: ${bb_u:,.0f} | ${hi52:,.0f}\n"
                    f"• Stop loss: ${sl:,.0f} | TP1: ${tp1:,.0f} | TP2: ${tp2:,.0f}\n\n"
                    f"⚠️ Educational analysis only."
                )

        # DEFAULT — general gold info
        else:
            topics_ar = ("**يمكنني الإجابة عن:**\n• سعر الذهب الآن\n• هل أشتري أم أبيع؟\n"
                         "• مستويات الدعم والمقاومة\n• مؤشر RSI / MACD / VIX / DXY\n"
                         "• نسبة الذهب/الفضة\n• التضخم والذهب\n• الذهب والإسلام (حلال؟)\n"
                         "• التحليل الشامل للسوق")
            topics_en = ("**I can answer about:**\n• Current gold price\n• Should I buy or sell?\n"
                         "• Support & resistance levels\n• RSI / MACD / VIX / DXY\n"
                         "• Gold/Silver ratio\n• Inflation & gold\n• Islamic finance & gold\n"
                         "• Full market analysis")
            if use_ar:
                return (f"☽ مرحباً! أنا **مستشار مانسا** للذهب — يعمل بالكامل بدون اتصال بأي خدمة خارجية.\n\n"
                        f"السعر الحالي: **${gold:,.2f}** | الإشارة: **{overall[1]}**\n\n{topics_ar}")
            else:
                return (f"☽ Hello! I'm **MANSA Gold Advisor** — works completely offline, no external API needed.\n\n"
                        f"Current price: **${gold:,.2f}** | Signal: **{overall[0]}**\n\n{topics_en}")

    # ── Build TA data ─────────────────────────────────────────────────────────
    ta_data = get_ta_advisor()

    # ── Welcome message ───────────────────────────────────────────────────────
    if not st.session_state["chat_init_done"]:
        overall_sig = advisor_answer("signal analysis", ta_data)
        welcome = (
            "☽ **أهلاً بك في مستشار مانسا للذهب**\n\n"
            "أنا مستشار ذكي يعمل بالكامل بدون مفتاح API أو رسوم — أجيب بناءً على بيانات السوق الحية.\n\n"
            "💡 اسأل مثلاً: *هل أشتري الذهب الآن؟* أو *ما مستويات الدعم؟*"
        ) if use_ar else (
            "☽ **Welcome to MANSA Gold Advisor**\n\n"
            "I'm a smart advisor that works completely free — no API key, no charges — answering based on live market data.\n\n"
            "💡 Try asking: *Should I buy gold now?* or *What are the support levels?*"
        )
        st.session_state["chat_history"] = [{"role":"assistant","content":welcome}]
        st.session_state["chat_init_done"] = True

    # ── Quick action buttons ───────────────────────────────────────────────────
    st.markdown(f"<div class='section-label'>⚡ {L['quick_questions']}</div>", unsafe_allow_html=True)
    quick_q = (["هل أشتري الآن؟","ما سعر الذهب؟","مستويات الدعم والمقاومة","تحليل RSI","الذهب والإسلام","تحليل شامل"]
               if use_ar else
               ["Should I buy now?","Current gold price","Support & resistance","RSI analysis","Gold & Islamic finance","Full market analysis"])
    q_cols = st.columns(3)
    triggered_q = None
    for i, qq in enumerate(quick_q):
        with q_cols[i % 3]:
            if st.button(qq, key=f"qq_{i}", use_container_width=True):
                triggered_q = qq
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Chat history — scrollable bubble UI ─────────────────────────────────
    msgs_html = ""
    for msg in st.session_state["chat_history"]:
        is_user  = msg["role"] == "user"
        align    = "flex-end" if is_user else "flex-start"
        bg_style = "rgba(122,90,4,0.25)" if is_user else C["card2"]
        bdr      = rgba(C["gold"], 0.6) if is_user else C["border2"]
        avatar   = "\U0001f464" if is_user else "\u262d"
        radius   = "14px 14px 4px 14px" if is_user else "14px 14px 14px 4px"
        rtl_s    = "direction:rtl;text-align:right;" if use_ar and not is_user else ""
        who_lbl  = ("\u0623\u0646\u062a" if use_ar else "YOU") if is_user else "MANSA \u262d"
        who_col  = C["muted"] if is_user else C["gold"]
        content  = msg["content"].replace("\n","<br>").replace("**","<b>",100)
        msgs_html += f"""
        <div style='display:flex;justify-content:{align};margin-bottom:14px;'>
          <div style='max-width:83%;background:{bg_style};border:1px solid {bdr};
                      border-radius:{radius};padding:12px 16px;'>
            <div style='font-size:10px;letter-spacing:.12em;color:{who_col};
                        font-weight:700;margin-bottom:6px;'>
              {avatar} &nbsp; {who_lbl}
            </div>
            <div style='font-family:Cairo,serif;font-size:{"15px" if use_ar else "14px"};
                        color:{C["text"]};line-height:1.8;{rtl_s}'>
              {content}
            </div>
          </div>
        </div>"""
    st.markdown(
        f"<div style='max-height:500px;overflow-y:auto;padding:12px;"
        f"border:1px solid {C['border2']};border-radius:10px;"
        f"background:{C['bg']};margin-bottom:12px;' id='cmsg'>"
        f"{msgs_html}</div>"
        f"<script>var e=document.getElementById('cmsg');if(e)e.scrollTop=e.scrollHeight;</script>",
        unsafe_allow_html=True
    )

    # ── Input row ─────────────────────────────────────────────────────────────
    inp_c, clr_c = st.columns([9, 1])
    with inp_c:
        user_input = st.text_input(
            "", placeholder=L.get("chatbot_placeholder","اكتب سؤالك هنا... اضغط Enter للإرسال"),
            key="chat_input", label_visibility="collapsed"
        )
        send = st.button(
            ("▶ " + L.get("chatbot_send","إرسال")),
            key="chat_send_btn", type="primary", use_container_width=True
        )
    with clr_c:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🗑️", key="chat_clear_btn",
                     help="مسح المحادثة" if use_ar else "Clear chat"):
            st.session_state["chat_history"]  = []
            st.session_state["chat_init_done"]= False
            st.rerun()

    # ── Process message ───────────────────────────────────────────────────────
    msg_to_send = triggered_q or (user_input.strip() if send and user_input else None)
    if msg_to_send:
        st.session_state["chat_history"].append({"role":"user","content":msg_to_send})

        api_key = get_api_key()

        if api_key:
            # ── Claude API path — unlimited questions ─────────────────────────
            with st.spinner("☽ " + ("جاري التفكير..." if use_ar else "Thinking...")):
                try:
                    import urllib.request as _ur2, json as _j2

                    # Build system prompt with live market context
                    lang_name = {"العربية 🇸🇦":"Arabic","English 🇬🇧":"English",
                                 "Français 🇫🇷":"French","Türkçe 🇹🇷":"Turkish",
                                 "اردو 🇵🇰":"Urdu"}.get(st.session_state["lang"],"English")
                    ma50_  = ta_data.get("ma50")  or 0
                    ma200_ = ta_data.get("ma200") or 0
                    rsi_   = ta_data.get("rsi")   or 50
                    macd_  = ta_data.get("macd")  or 0
                    sig_   = ta_data.get("sig")   or 0
                    bb_u_  = ta_data.get("bb_u")  or 0
                    bb_l_  = ta_data.get("bb_l")  or 0
                    atr_   = ta_data.get("atr")   or 0

                    system_p = f"""You are MANSA, an expert AI gold trading advisor.
Always respond in {lang_name}. Match the user's language exactly.

LIVE MARKET DATA RIGHT NOW:
- Gold Spot: ${g_ref:,.2f} USD/oz (24K) | Change: {live['gold']['pct']:+.2f}% today
- Silver: ${live['silver']['price']:,.2f} | Oil: ${live['oil']['price']:,.2f} | DXY: {live['dxy']['price']:,.2f}
- VIX: {live['vix']['price']:,.1f} | US10Y: {live['us10y']['price']:,.2f}% | S&P500 change: {live['spx']['pct']:+.2f}%

TECHNICAL ANALYSIS:
- MA50: ${ma50_:,.0f} | MA200: ${ma200_:,.0f} | Price vs MA50: {'ABOVE ✅' if g_ref>ma50_ else 'BELOW ⚠️'}
- RSI(14): {rsi_:.1f} | {'OVERSOLD 🟢' if rsi_<30 else 'OVERBOUGHT 🔴' if rsi_>70 else 'NEUTRAL'}
- MACD: {macd_:+.2f} vs Signal {sig_:+.2f} | {'BULLISH 🟢' if macd_>sig_ else 'BEARISH 🔴'}
- Bollinger: Upper ${bb_u_:,.0f} | Lower ${bb_l_:,.0f}
- ATR(14): ${atr_:,.2f} | Stop loss suggestion: ${g_ref-1.5*atr_:,.0f} | TP1: ${g_ref+1.5*atr_:,.0f}

PRIMARY MARKET: {st.session_state['primary_mkt']}

Rules:
- Give specific, actionable advice with exact price levels
- Always mention actual numbers from the live data above
- For Islamic finance questions, be scholarly and careful
- End with a one-line disclaimer that this is educational only
- Be concise: 3-5 paragraphs max"""

                    # Only send last 10 exchanges to keep tokens manageable
                    hist_for_api = st.session_state["chat_history"][-20:]  # last 10 exchanges
                    api_msgs = [{"role":m["role"],"content":m["content"]}
                                for m in hist_for_api
                                if m["role"] in ("user","assistant")]

                    payload = _j2.dumps({
                        "model":      "claude-sonnet-4-20250514",
                        "max_tokens": 800,
                        "system":     system_p,
                        "messages":   api_msgs,
                    }).encode()

                    req2 = _ur2.Request(
                        "https://api.anthropic.com/v1/messages",
                        data=payload,
                        headers={
                            "Content-Type":      "application/json",
                            "x-api-key":         api_key,
                            "anthropic-version": "2023-06-01",
                        },
                        method="POST"
                    )
                    with _ur2.urlopen(req2, timeout=25) as resp2:
                        d2 = _j2.loads(resp2.read().decode())
                    reply = "".join(b["text"] for b in d2.get("content",[])
                                    if b.get("type")=="text")
                    if not reply:
                        raise ValueError("empty response")

                except Exception as ex:
                    err_str = str(ex)
                    # Friendly error messages
                    if "401" in err_str:
                        err_msg = ("⚠️ مفتاح API غير صحيح. تحقق من مفتاحك في console.anthropic.com"
                                   if use_ar else
                                   "⚠️ Invalid API key. Check your key at console.anthropic.com")
                    elif "429" in err_str:
                        err_msg = ("⚠️ تجاوزت حد الاستخدام. انتظر قليلاً وأعد المحاولة."
                                   if use_ar else
                                   "⚠️ Rate limit reached. Please wait a moment and try again.")
                    elif "timeout" in err_str.lower() or "timed out" in err_str.lower():
                        err_msg = ("⚠️ انتهت مهلة الاتصال. سأستخدم التحليل المحلي بدلاً من ذلك:"
                                   if use_ar else
                                   "⚠️ Connection timeout. Using local analysis instead:")
                        err_msg += "\n\n" + advisor_answer(msg_to_send, ta_data)
                    else:
                        err_msg = ("⚠️ خطأ: " if use_ar else "⚠️ Error: ") + err_str[:120]
                        err_msg += "\n\n" + advisor_answer(msg_to_send, ta_data)
                    reply = err_msg

        else:
            # ── Offline rule engine — no API key ─────────────────────────────
            reply = advisor_answer(msg_to_send, ta_data)

        st.session_state["chat_history"].append({"role":"assistant","content":reply})
        st.rerun()

    # ── Disclaimer ────────────────────────────────────────────────────────────
    api_key_set = bool(get_api_key())
    st.markdown(f"""
    <div style='background:{C['card']};border:1px solid {C['border']};border-radius:4px;
                padding:8px 16px;display:flex;justify-content:space-between;align-items:center;margin-top:1rem;'>
      <div style='font-size:11px;font-style:italic;color:{C['dim']};'>
        ⚠️ {L['disclaimer']}
      </div>
      <div style='font-size:10px;color:{"#52B788" if api_key_set else C['muted']};'>
        {"🟢 Claude API · أسئلة غير محدودة" if api_key_set and use_ar else
         "🟢 Claude API · Unlimited questions" if api_key_set else
         "🟡 وضع محلي · بدون مفتاح API" if use_ar else
         "🟡 Offline mode · No API key set"}
      </div>
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: ABOUT MANSA
# ═══════════════════════════════════════════════════════════════════════════════
elif nav == L["nav_about"]:
    use_ar = is_rtl()
    ph("🏛️  " + L.get("nav_about","About Mansa").replace("🏛️  ",""))

    # ── Navigation overview tabs ──────────────────────────────────────────────
    tab_overview, tab_story, tab_features, tab_logos, tab_credits, tab_changelog = st.tabs([
        "🏠 " + ("نظرة عامة" if use_ar else "Overview"),
        "☽ " + ("قصة مانسا" if use_ar else "Mansa's Story"),
        "⚙️ " + ("المزايا"    if use_ar else "Features"),
        "🎨 " + ("الشعارات"   if use_ar else "Logos"),
        "👤 " + ("عن المطور"  if use_ar else "Developer"),
        "📋 " + ("الإصدارات"  if use_ar else "Changelog"),
    ])

    # ── TAB 1: Overview — all pages listed ───────────────────────────────────
    with tab_overview:
        st.markdown("<br>", unsafe_allow_html=True)
        rtl_s = "direction:rtl;text-align:right;" if use_ar else ""

        PAGE_GROUPS = [
            ("📊 " + ("التحليل المباشر" if use_ar else "Live Analysis"), [
                ("🏠", L["nav_dashboard"],   "أسعار مباشرة · 16 سوقاً · لوحة بيانات متكاملة" if use_ar else "Live prices · 16 markets · complete dashboard"),
                ("🌍", L["nav_markets"],     "الأسواق العربية والدولية · جميع العيارات" if use_ar else "Arab & international markets · all purities"),
                ("📈", L["nav_charts"],      "رسوم بيانية تفاعلية · مؤشرات تقنية" if use_ar else "Interactive charts · technical indicators"),
                ("🔭", L["nav_simulator"],   "محاكاة تأثير العوامل الاقتصادية على الذهب" if use_ar else "Simulate macro factor impact on gold price"),
            ]),
            ("🤖 " + ("الذكاء الاصطناعي" if use_ar else "AI & Predictions"), [
                ("🤖", L["nav_predictions"],"6 نماذج AI لتوقع سعر الذهب غداً" if use_ar else "6 AI/ML models predicting tomorrow's gold price"),
                ("💬", L["nav_advisor"],    "مستشار تداول ذكي مدعوم بكلود" if use_ar else "AI trading advisor powered by Claude"),
                ("📡", L["nav_signals"],    "10 إشارات تقنية مباشرة مع إشارة مانسا الإجمالية" if use_ar else "10 live technical signals + MANSA overall signal"),
                ("🌡️", L["nav_sentiment"],  "مؤشر الخوف والجشع للذهب" if use_ar else "Gold Fear & Greed Index"),
            ]),
            ("💼 " + ("إدارة المحفظة" if use_ar else "Portfolio Management"), [
                ("💼", L["nav_portfolio"],  "تتبع محفظتك الذهبية مع ربح/خسارة مباشر" if use_ar else "Track your gold portfolio with live P&L"),
                ("🎯", L["nav_demo"],       "تداول تجريبي بأموال وهمية — بدون مخاطر" if use_ar else "Paper trading with virtual money — zero risk"),
                ("🪙", L["nav_savings"],    "خطة ادخار شهرية بالذهب · مقارنة بالنقد" if use_ar else "Monthly gold savings plan · vs cash comparison"),
                ("📓", L["nav_journal"],    "سجل صفقاتك وحللها مع مساعدة الذكاء الاصطناعي" if use_ar else "Log & analyse your trades with AI assistance"),
                ("🧮", L["nav_calculator"], "حاسبة حجم الصفقة · الربح/الخسارة · الهامش · التعادل" if use_ar else "Position size · P&L · margin · break-even"),
            ]),
            ("🌍 " + ("الأسواق والبيانات" if use_ar else "Markets & Data"), [
                ("📂", L["nav_data"],       "استكشاف بيانات التدريب · 35 سنة من بيانات الذهب" if use_ar else "Training data explorer · 35 years of gold data"),
                ("📊", L["nav_heatmap"],    "مصفوفة الارتباط بين الذهب والأصول الأخرى" if use_ar else "Correlation matrix: gold vs other assets"),
                ("📈", L["nav_compare"],    "مقارنة أداء الأصول عبر الزمن" if use_ar else "Compare asset performance over time") if L.get("nav_compare") else None,
                ("⛏️", L["nav_supply"],     "العرض والطلب · إنتاج المناجم · صناديق ETF" if use_ar else "Supply & demand · mine production · ETFs"),
                ("🏦", L["nav_cb"],         "احتياطيات البنوك المركزية · بيانات WGC 2025" if use_ar else "Central bank reserves · WGC 2025 data"),
                ("💱", L["nav_currency"],   "محول العملات بسعر الذهب الحي" if use_ar else "Currency converter at live gold rate"),
            ]),
            ("📅 " + ("التخطيط والتنبيهات" if use_ar else "Planning & Alerts"), [
                ("🔔", L["nav_alerts"],     "تنبيهات سعرية تظهر عند بلوغ الهدف" if use_ar else "Price alerts triggered when target is reached"),
                ("📅", L["nav_calendar"],   "التقويم الاقتصادي مع الأحداث العربية" if use_ar else "Economic calendar with Arab market events"),
                ("☪️", L["nav_zakat"],      "حاسبة الزكاة على الذهب تلقائياً" if use_ar else "Automatic Zakat calculator on gold holdings"),
                ("📉", L["nav_drawdown"],   "اختبار الضغط · تحليل أسوأ السيناريوهات" if use_ar else "Stress test · worst-case scenario analysis"),
            ]),
            ("📋 " + ("التقارير والمعلومات" if use_ar else "Reports & Info"), [
                ("📋", L["nav_report"],     "تقرير أسبوعي تلقائي قابل للتحميل" if use_ar else "Auto-generated weekly report · downloadable"),
                ("🏆", L["nav_mansa_score"],"مؤشر مانسا الذهبي — نقطة واحدة تلخص السوق" if use_ar else "MANSA Gold Score — one number summarising the market"),
                ("🕐", L["nav_sessions"],   "جلسات التداول العالمية مع التوقيت" if use_ar else "Global trading sessions with live countdown"),
                ("🌍", L["nav_goldmap"],    "خريطة الذهب العالمية" if use_ar else "Global gold map"),
            ]),
        ]

        for group_title, pages in PAGE_GROUPS:
            st.markdown(f"<div class='section-label'>{group_title}</div>",
                        unsafe_allow_html=True)
            cols = st.columns(2)
            col_idx = 0
            for item in pages:
                if item is None:
                    continue
                icon, nav_lbl, desc = item
                with cols[col_idx % 2]:
                    st.markdown(f"""
                    <div style='background:{C["card"]};border:1px solid {C["border"]};
                                border-radius:5px;padding:10px 14px;margin-bottom:6px;
                                {rtl_s}'>
                      <div style='font-size:13px;font-weight:700;color:{C["gold_pale"]};'>
                        {icon} {nav_lbl}
                      </div>
                      <div style='font-size:11px;color:{C["muted"]};margin-top:3px;'>
                        {desc}
                      </div>
                    </div>""", unsafe_allow_html=True)
                col_idx += 1
            st.markdown("<br>", unsafe_allow_html=True)

    # ── TAB 2: Mansa's Story ──────────────────────────────────────────────────
    with tab_story:
        st.markdown("<br>", unsafe_allow_html=True)
        rtl_s = "direction:rtl;text-align:right;" if use_ar else ""
        bdr_s = f"border-right:3px solid {C['gold']};" if use_ar else f"border-left:3px solid {C['gold']};"
        st.markdown(f"""
        <div style='background:{C["card2"]};border:1px solid {C["border2"]};
                    border-radius:8px;padding:26px 32px;{bdr_s}'>
          <div style='font-family:"Cairo",{C["font_h"]},serif;font-size:24px;
                      font-weight:900;color:{C["gold_hi"]};margin-bottom:16px;{rtl_s}'>
            ☽ {L["mansa_title"]}
          </div>
          <div style='font-family:"Cairo",sans-serif;font-size:{"15px" if use_ar else "14px"};
                      color:{C["text"]};line-height:2.0;{rtl_s}'>
            {"مانسا موسى (1280–1337م) هو الإمبراطور العاشر لإمبراطورية مالي الإسلامية في غرب أفريقيا، ويُعدّ أثرى إنسان عرفه التاريخ على الإطلاق. في عام 1324م، قاد حجّه الأسطوري إلى مكة المكرمة في موكب بلغ أكثر من 60,000 شخص يحملون قوافل من الذهب. سُمِّيت هذه المنصة تيمّنًا به، إذ تجمع بين الإرث الذهبي الإسلامي العريق والذكاء المالي الحديث." if use_ar else
             "Mansa Musa (1280–1337 CE) was the tenth emperor of the Mali Empire in West Africa and is widely regarded as the wealthiest individual in recorded history. In 1324 CE, he embarked on a legendary pilgrimage to Mecca with over 60,000 people and vast quantities of gold, causing gold prices to collapse across Egypt and the Middle East for a decade. This platform is named MANSA in his honour."}
          </div>
          <div style='margin-top:14px;font-size:13px;color:{C["gold"]};
                      font-style:italic;{rtl_s}'>
            ❝ {"ثروته المُقدَّرة بأكثر من 400 مليار دولار — أثرى من أي شخص على قيد الحياة اليوم" if use_ar else
               "Estimated wealth exceeds $400 billion in modern values — richer than any person alive today"} ❞
          </div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        # Historical highlights
        st.markdown(f"<div class='section-label'>{L['hist_highlights']}</div>",
                    unsafe_allow_html=True)
        highlights = [
            ("1280", "☽", "ولادة مانسا موسى في إمبراطورية مالي" if use_ar else "Birth of Mansa Musa in the Mali Empire"),
            ("1312", "👑", "توليه عرش إمبراطورية مالي" if use_ar else "Ascended to the throne of the Mali Empire"),
            ("1324", "🕋", "حجه الأسطوري إلى مكة بـ 60,000 شخص وقوافل الذهب" if use_ar else "Legendary hajj to Mecca with 60,000 people and gold caravans"),
            ("1327", "🕌", "بناء مسجد جنغريبر في تمبكتو" if use_ar else "Built Djinguereber Mosque in Timbuktu"),
            ("1337", "✨", "وفاته تاركاً إرثاً لا يُنسى في تاريخ الذهب" if use_ar else "Passed away, leaving an unforgettable legacy in gold history"),
        ]
        for year, icon, text in highlights:
            st.markdown(f"""
            <div style='display:flex;align-items:flex-start;gap:14px;
                        margin-bottom:10px;{"direction:rtl;text-align:right;" if use_ar else ""}'>
              <div style='background:{C["gold_dark"]}44;border:1px solid {C["gold"]}44;
                          border-radius:4px;padding:4px 10px;font-family:{C["font_m"]},monospace;
                          font-size:12px;color:{C["gold"]};white-space:nowrap;min-width:50px;
                          text-align:center;'>{year}</div>
              <div style='font-size:22px;'>{icon}</div>
              <div style='font-size:13px;color:{C["text"]};line-height:1.7;'>{text}</div>
            </div>""", unsafe_allow_html=True)

    # ── TAB 3: Features ───────────────────────────────────────────────────────
    with tab_features:
        st.markdown("<br>", unsafe_allow_html=True)
        rtl_s = "direction:rtl;text-align:right;" if use_ar else ""
        features_list = [
            ("🌍", ("دعم 5 لغات" if use_ar else "5 Languages"),
             ("العربية · الإنجليزية · الفرنسية · التركية · الأردوية مع دعم RTL" if use_ar else
              "Arabic · English · French · Turkish · Urdu with full RTL support")),
            ("🎨", ("3 مظاهر بصرية" if use_ar else "3 Visual Themes"),
             ("الحضارة الإسلامية · العملة الذهبية القديمة · قاعة التداول" if use_ar else
              "Islamic Civilization · Ancient Gold Coin · Trading Floor")),
            ("⚡", ("بيانات مباشرة" if use_ar else "Live Data"),
             ("Twelve Data (أقل من دقيقة) أو Yahoo Finance احتياطياً" if use_ar else
              "Twelve Data (<1 min) or Yahoo Finance fallback")),
            ("☁️", ("مزامنة Supabase" if use_ar else "Supabase Sync"),
             ("المحفظة والتنبيهات والسجل محفوظة عبر الجلسات" if use_ar else
              "Portfolio, alerts & journal persist across sessions")),
            ("🤖", ("6 نماذج AI" if use_ar else "6 AI Models"),
             ("Linear Regression · Random Forest · Gradient Boosting · XGBoost · LSTM · Prophet" )),
            ("☪️", ("حاسبة الزكاة" if use_ar else "Zakat Calculator"),
             ("حساب تلقائي بناءً على سعر الذهب الحي" if use_ar else
              "Auto-calculated based on live gold spot price")),
            ("📱", ("تطبيق PWA" if use_ar else "PWA App"),
             ("قابل للتثبيت على الهاتف كتطبيق محلي" if use_ar else
              "Installable as a native-feeling mobile app")),
            ("🔌", ("أداة تضمين" if use_ar else "Embed Widget"),
             ("أضف سعر الذهب لموقعك بسطر HTML واحد" if use_ar else
              "Add live gold price to any website with one HTML line")),
        ]
        feat_cols = st.columns(2)
        for i, (icon, title, desc) in enumerate(features_list):
            with feat_cols[i % 2]:
                st.markdown(f"""
                <div style='background:{C["card2"]};border:1px solid {C["border2"]};
                            border-radius:6px;padding:14px 16px;margin-bottom:8px;{rtl_s}'>
                  <div style='font-size:22px;margin-bottom:6px;'>{icon}</div>
                  <div style='font-size:13px;font-weight:700;color:{C["gold_pale"]};
                              margin-bottom:4px;'>{title}</div>
                  <div style='font-size:11px;color:{C["muted"]};line-height:1.6;'>{desc}</div>
                </div>""", unsafe_allow_html=True)

    # ── TAB 4: Logos ──────────────────────────────────────────────────────────
    with tab_logos:
        st.markdown(f"<div class='section-label'>{L['brand_logos']}</div>",
                    unsafe_allow_html=True)
        logo_cols = st.columns(3)
        for col, theme_key in zip(logo_cols, list(THEMES.keys())):
            svg = get_logo_svg(theme_key, width=200)
            th  = THEMES[theme_key]
            with col:
                st.components.v1.html(f"""
                <div style="text-align:center;padding:12px 0;background:transparent;">
                  {svg}
                  <div style="font-family:Georgia,serif;font-size:11px;
                              color:{th['muted']};margin-top:6px;">
                    {th['desc']}
                  </div>
                </div>""", height=int(200*0.85)+50, scrolling=False)

    # ── TAB 5: Developer ──────────────────────────────────────────────────────
    with tab_credits:
        st.markdown("<br>", unsafe_allow_html=True)
        rtl_s = "direction:rtl;text-align:right;" if use_ar else ""
        st.markdown(f"""
        <div style='background:{C["card2"]};border:1px solid {C["gold"]}44;border-radius:8px;
                    padding:24px 28px;max-width:600px;margin:0 auto;{rtl_s}'>
          <div style='font-size:40px;text-align:center;margin-bottom:14px;'>👤</div>
          <div style='font-family:{C["font_h"]},serif;font-size:20px;font-weight:900;
                      color:{C["gold_pale"]};text-align:center;letter-spacing:.1em;'>
            OWN AL ANSARI
          </div>
          <div style='font-size:12px;color:{C["muted"]};text-align:center;
                      margin:6px 0 16px;font-style:italic;'>
            {L["developer"]}
          </div>
          <div style='font-family:"Cairo",sans-serif;font-size:13px;color:{C["text"]};
                      line-height:1.9;'>
            {"بنى هذه المنصة استلهامًا من إرث مانسا موسى الذهبي، ودمجه بأحدث تقنيات الذكاء الاصطناعي لخدمة المتداولين العرب." if use_ar else
             "Built this platform inspired by the golden legacy of Mansa Musa, fused with modern AI to serve Arab gold traders."}
          </div>
          <div style='margin-top:16px;text-align:center;'>
            <div style='font-size:11px;color:{C["dim"]};font-style:italic;'>
              {L["copyright"]} · {L["platform_name"]}
            </div>
            <div style='font-size:10px;color:{C["dim"]};margin-top:4px;'>
              Inspired by Mansa Musa · The Golden King of Mali · 1312 CE ☽
            </div>
          </div>
        </div>""", unsafe_allow_html=True)

    # ── TAB 6: Changelog ─────────────────────────────────────────────────────
    with tab_changelog:
        st.markdown("<br>", unsafe_allow_html=True)
        rtl_s = "direction:rtl;text-align:right;" if use_ar else ""
        versions = [
            ("v4.0.0", "2026-03", [
                ("🎯", "Demo Trading — paper trading with virtual $10,000"),
                ("🪙", "Gold Savings Plan — monthly DCA tracker vs cash"),
                ("📋", "Weekly Report — auto-generated downloadable summary"),
                ("🏦", "Central Bank Reserves — dedicated page with Arab insights"),
                ("⚡", "Twelve Data real-time API (<1 min latency)"),
                ("☁️", "Supabase persistence — data survives browser close"),
            ]),
            ("v3.0.0", "2025-12", [
                ("🌐", "5-language support: Arabic, English, French, Turkish, Urdu"),
                ("🎨", "3 visual themes with full CSS animation suite"),
                ("🤖", "6 AI/ML prediction models with R² scoring"),
                ("📡", "10 live technical signals + MANSA Score"),
                ("🔔", "Price alerts, trade journal, portfolio tracker"),
                ("🎮", "Mansa runner game"),
            ]),
            ("v2.0.0", "2025-06", [
                ("📊", "27 pages: Charts, Simulator, Zakat, Economic Calendar"),
                ("🌍", "16 Arab & international gold markets"),
                ("⛏️", "Supply & demand data, ETF holdings"),
            ]),
            ("v1.0.0", "2025-01", [
                ("🏠", "Initial release — dashboard, markets, live prices"),
            ]),
        ]
        for ver, date, changes in versions:
            with st.expander(f"**{ver}** · {date}", expanded=(ver == "v4.0.0")):
                for icon, desc in changes:
                    st.markdown(f"""
                    <div style='display:flex;gap:10px;padding:4px 0;{rtl_s}'>
                      <span style='font-size:16px;'>{icon}</span>
                      <span style='font-size:12px;color:{C["text"]};'>{desc}</span>
                    </div>""", unsafe_allow_html=True)


elif nav == L["nav_portfolio"]:
    use_ar = is_rtl()
    ph("💼  " + L.get("nav_portfolio","Portfolio").replace("💼  ",""),
       "تتبع محفظتك الذهبية · الربح والخسارة · العملات المتعددة" if use_ar else
       "Track your gold portfolio · P&L · Multi-currency")

    rtl_p = "direction:rtl;text-align:right;" if use_ar else ""

    # ── Add position form ──────────────────────────────────────────────────────
    with st.expander("➕ " + ("إضافة صفقة جديدة" if use_ar else "Add New Position"), expanded=len(st.session_state["portfolio_entries"])==0):
        f1,f2,f3 = st.columns(3)
        with f1:
            p_label   = st.text_input("🏷️ " + ("اسم/وصف" if use_ar else "Label"), placeholder="e.g. ذهب 21K منزلي", key="p_label")
            p_qty     = st.number_input("⚖️ " + ("الكمية" if use_ar else "Quantity"), min_value=0.001, value=10.0, step=0.1, key="p_qty")
            p_unit    = st.selectbox("📐 " + ("الوحدة" if use_ar else "Unit"), ["gram","oz","kg","tola"], key="p_unit")
        with f2:
            p_buy     = st.number_input("💰 " + ("سعر الشراء" if use_ar else "Buy Price"), min_value=0.01, value=float(g_ref/31.1035), step=0.01, key="p_buy",
                                         help="Price per unit in your currency")
            p_curr    = st.selectbox("💱 " + ("العملة" if use_ar else "Currency"),
                                      ["USD","JOD","SAR","AED","EGP","KWD","QAR","BHD","GBP","EUR","TRY"], key="p_curr")
            p_date    = st.date_input("📅 " + ("تاريخ الشراء" if use_ar else "Buy Date"),
                                       value=datetime.date.today(), key="p_date")
        with f3:
            st.markdown("<br><br>", unsafe_allow_html=True)
            if st.button("✅ " + ("حفظ الصفقة" if use_ar else "Save Position"),
                         type="primary", use_container_width=True, key="p_save"):
                st.session_state["portfolio_entries"].append({
                    "label": p_label or f"Gold {p_unit}", "qty": p_qty, "unit": p_unit,
                    "buy_price": p_buy, "currency": p_curr,
                    "date": str(p_date), "id": len(st.session_state["portfolio_entries"])
                })
                _sb_save()
                st.success("✅ " + ("تمت الإضافة" if use_ar else "Position added"))
                st.rerun()
            if st.button("🗑️ " + ("مسح الكل" if use_ar else "Clear All"),
                         use_container_width=True, key="p_clear"):
                st.session_state["portfolio_entries"] = []
                _sb_save()
                st.rerun()
        st.markdown(f"""
        <div style='font-size:9px;color:{C["dim"]};margin-top:6px;text-align:center;'>
          {"💡 لتصدير التقرير: استخدم طباعة الصفحة (Ctrl+P) واختر 'حفظ كـ PDF'" if use_ar else
           "💡 To export: use browser Print (Ctrl+P) → Save as PDF"}
        </div>""", unsafe_allow_html=True)

    entries = st.session_state["portfolio_entries"]
    if not entries:
        st.markdown(f"""
        <div style='background:{C['card2']};border:2px dashed {C['gold']}44;border-radius:8px;
                    padding:40px;text-align:center;'>
          <div style='font-size:48px;'>💼</div>
          <div style='font-size:{"16px" if use_ar else "14px"};color:{C['muted']};margin-top:12px;{rtl_p}'>
            {'أضف صفقاتك الذهبية أعلاه لتتبع الربح والخسارة في الوقت الفعلي' if use_ar else
             'Add your gold positions above to track real-time P&L across all markets'}
          </div>
        </div>""", unsafe_allow_html=True)
    else:
        # FX rates for conversion
        # Live FX rates (all as USD per 1 unit of currency)
        FX = {
            "USD": 1.0,
            "JOD": 1.0 / fetch_fx("USDJOD=X", False, 0.709),   # JOD is stronger than USD
            "SAR": 1.0 / fetch_fx("USDSAR=X", False, 3.75),
            "AED": 1.0 / fetch_fx("USDAED=X", False, 3.6725),
            "EGP": 1.0 / fetch_fx("USDEGP=X", False, 50.9),
            "KWD": 1.0 / fetch_fx("USDKWD=X", False, 0.307),
            "QAR": 1.0 / fetch_fx("USDQAR=X", False, 3.64),
            "BHD": 1.0 / fetch_fx("USDBHD=X", False, 0.377),
            "GBP": fetch_fx("GBPUSD=X",  True,  1.27),
            "EUR": fetch_fx("EURUSD=X",  True,  1.08),
            "TRY": 1.0 / fetch_fx("USDTRY=X", False, 32.0),
        }

        def to_usd_oz(qty, unit, price_per_unit, currency):
            """Convert a position's cost to USD per oz equivalent."""
            unit_to_oz = {"gram":1/31.1035,"oz":1.0,"kg":1000/31.1035,"tola":11.6638/31.1035}
            qty_oz   = qty * unit_to_oz.get(unit, 1/31.1035)
            cost_usd = price_per_unit * qty * (1.0 / FX.get(currency, 1.0))
            return qty_oz, cost_usd

        total_cost_usd = 0; total_qty_oz = 0; total_pnl_usd = 0
        rows = []
        for e in entries:
            qty_oz, cost_usd = to_usd_oz(e["qty"], e["unit"], e["buy_price"], e["currency"])
            current_val_usd  = qty_oz * g_ref
            pnl_usd          = current_val_usd - cost_usd
            pnl_pct          = (pnl_usd / cost_usd * 100) if cost_usd > 0 else 0
            buy_usd_oz       = cost_usd / qty_oz if qty_oz > 0 else 0
            total_cost_usd  += cost_usd
            total_qty_oz    += qty_oz
            total_pnl_usd   += pnl_usd
            rows.append({
                "label": e["label"], "qty_oz": qty_oz, "cost_usd": cost_usd,
                "current_val": current_val_usd, "pnl": pnl_usd, "pnl_pct": pnl_pct,
                "buy_usd_oz": buy_usd_oz, "date": e["date"], "id": e["id"]
            })

        total_val_usd   = total_qty_oz * g_ref
        total_pnl_pct   = (total_pnl_usd / total_cost_usd * 100) if total_cost_usd > 0 else 0
        breakeven_price = total_cost_usd / total_qty_oz if total_qty_oz > 0 else 0

        # Summary hero cards
        sc1,sc2,sc3,sc4 = st.columns(4)
        for col, lbl, val, sub, col_c in [
            (sc1, "💰 " + ("إجمالي القيمة" if use_ar else "Total Value"),
             f"${total_val_usd:,.2f}", "USD", C["gold_hi"]),
            (sc2, "📈 " + ("إجمالي الربح/الخسارة" if use_ar else "Total P&L"),
             f"${total_pnl_usd:+,.2f}", f"{total_pnl_pct:+.2f}%",
             C["green"] if total_pnl_usd >= 0 else C["red"]),
            (sc3, "⚖️ " + ("إجمالي الوزن" if use_ar else "Total Weight"),
             f"{total_qty_oz*31.1035:,.2f}g", f"{total_qty_oz:.4f} oz", C["gold_pale"]),
            (sc4, "🎯 " + ("سعر التعادل" if use_ar else "Break-even"),
             f"${breakeven_price:,.2f}", "USD/oz", C["muted"]),
        ]:
            with col:
                st.markdown(f"""<div class='stat-card' style='text-align:center;border-color:{col_c}44;'>
                  <div class='stat-label'>{lbl}</div>
                  <div style='font-family:{C['font_m']},monospace;font-size:20px;
                              font-weight:700;color:{col_c};'>{val}</div>
                  <div style='font-size:11px;color:{C['dim']};'>{sub}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # P&L chart (equity curve across positions)
        if len(rows) > 1:
            fig_pnl = go.Figure()
            fig_pnl.add_trace(go.Bar(
                x=[r["label"] for r in rows],
                y=[r["pnl"] for r in rows],
                marker_color=[C["green"] if r["pnl"]>=0 else C["red"] for r in rows],
                text=[f"${r['pnl']:+,.0f}<br>{r['pnl_pct']:+.1f}%" for r in rows],
                textposition="auto", name="P&L"
            ))
            fig_pnl.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=C["card"],
                font=dict(color=C["text"], size=10),
                height=260, margin=dict(l=0,r=0,t=20,b=0),
                xaxis=dict(gridcolor=C["border2"]), yaxis=dict(gridcolor=C["border2"]),
                title=dict(text=("الربح والخسارة لكل صفقة" if use_ar else "P&L per Position"),
                           font=dict(color=C["gold"],size=12))
            )
            st.plotly_chart(fig_pnl, use_container_width=True)

        # Positions table
        st.markdown(f"<div class='section-label'>{L['position_details']}</div>", unsafe_allow_html=True)
        for r in rows:
            pnl_col = C["green"] if r["pnl"]>=0 else C["red"]
            pnl_icon= "▲" if r["pnl"]>=0 else "▼"
            del_key = f"del_{r['id']}"
            dc1, dc2 = st.columns([6,1])
            with dc1:
                st.markdown(f"""
                <div class='stat-card' style='margin-bottom:6px;'>
                  <div style='display:flex;justify-content:space-between;flex-wrap:wrap;gap:8px;'>
                    <div>
                      <div style='font-size:14px;font-weight:700;color:{C['gold_pale']};'>{r['label']}</div>
                      <div style='font-size:11px;color:{C['dim']};margin-top:2px;'>
                        {r['qty_oz']*31.1035:.2f}g · {r['qty_oz']:.4f}oz · Buy: ${r['buy_usd_oz']:,.2f}/oz · {r['date']}
                      </div>
                    </div>
                    <div style='text-align:right;'>
                      <div style='font-family:{C['font_m']},monospace;font-size:16px;
                                  color:{C['gold_hi']};'>${r['current_val']:,.2f}</div>
                      <div style='font-family:{C['font_m']},monospace;font-size:13px;color:{pnl_col};'>
                        {pnl_icon} ${r['pnl']:+,.2f} ({r['pnl_pct']:+.2f}%)
                      </div>
                    </div>
                  </div>
                </div>""", unsafe_allow_html=True)
            with dc2:
                if st.button("🗑️", key=del_key):
                    st.session_state["portfolio_entries"] = [
                        e for e in st.session_state["portfolio_entries"] if e["id"] != r["id"]
                    ]
                    _sb_save()
                    st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: TRADING CALCULATOR 🧮
# ═══════════════════════════════════════════════════════════════════════════════
elif nav == L["nav_calculator"]:
    use_ar = is_rtl()
    ph("🧮  " + L.get("nav_calculator","Calculator").replace("🧮  ",""),
       "حاسبة الحجم · الربح/الخسارة · الهامش · نقطة التعادل" if use_ar else
       "Position Size · P&L · Margin · Break-even · All in one")

    tab_labels = (["📐 حجم الصفقة","💰 الربح/الخسارة","📊 الهامش","🎯 نقطة التعادل"]
                  if use_ar else
                  ["📐 Position Size","💰 P&L Calculator","📊 Margin","🎯 Break-even"])
    t1,t2,t3,t4 = st.tabs(tab_labels)

    with t1:  # Position Size
        st.markdown(f"<div class='section-label'>{L['calc_pos_sub']}</div>", unsafe_allow_html=True)
        c1,c2 = st.columns(2)
        with c1:
            ps_capital   = st.number_input("💰 " + ("رأس المال" if use_ar else "Account Capital"), value=10000.0, step=100.0, key="ps_cap")
            ps_risk_pct  = st.slider("⚠️ " + ("نسبة المخاطرة %" if use_ar else "Risk %"), 0.5, 5.0, 2.0, 0.5, key="ps_risk")
            ps_entry     = st.number_input("📥 " + ("سعر الدخول" if use_ar else "Entry Price"), value=float(g_ref), step=1.0, key="ps_entry")
            ps_sl        = st.number_input("🛑 " + (L['stop_loss_lbl']), value=float(g_ref*0.99), step=1.0, key="ps_sl")
            ps_curr      = st.selectbox("💱 " + ("العملة" if use_ar else "Currency"), ["USD","JOD","SAR","AED"], key="ps_curr2")
        with c2:
            risk_usd   = ps_capital * ps_risk_pct / 100
            sl_dist    = abs(ps_entry - ps_sl)
            pos_oz     = risk_usd / sl_dist if sl_dist > 0 else 0
            pos_grams  = pos_oz * 31.1035
            pos_cost   = pos_oz * ps_entry
            tp1        = ps_entry + 1.5 * sl_dist if ps_entry > ps_sl else ps_entry - 1.5*sl_dist
            tp2        = ps_entry + 3.0 * sl_dist if ps_entry > ps_sl else ps_entry - 3.0*sl_dist
            rr_ratio   = 1.5

            st.markdown(f"""
            <div style='background:{C['card2']};border:1px solid {C['gold']}55;border-radius:8px;padding:20px;'>
              <div style='font-size:12px;color:{C['muted']};margin-bottom:12px;{("direction:rtl;text-align:right;" if use_ar else "")}'>
                {L['results']}
              </div>
              <div style='display:grid;grid-template-columns:1fr 1fr;gap:10px;'>
                <div class='stat-card' style='text-align:center;padding:10px;'>
                  <div class='stat-label'>{L['risk_amount']}</div>
                  <div style='font-size:18px;color:{C["red"]};font-weight:700;'>${risk_usd:,.2f}</div>
                </div>
                <div class='stat-card' style='text-align:center;padding:10px;'>
                  <div class='stat-label'>{L['position_size_lbl']}</div>
                  <div style='font-size:18px;color:{C["gold_hi"]};font-weight:700;'>{pos_oz:.4f} oz</div>
                  <div style='font-size:11px;color:{C["dim"]};'>{pos_grams:.2f}g</div>
                </div>
                <div class='stat-card' style='text-align:center;padding:10px;'>
                  <div class='stat-label'>{L['position_cost']}</div>
                  <div style='font-size:18px;color:{C["gold_pale"]};font-weight:700;'>${pos_cost:,.2f}</div>
                </div>
                <div class='stat-card' style='text-align:center;padding:10px;'>
                  <div class='stat-label'>{L['risk_reward']}</div>
                  <div style='font-size:18px;color:{C["green"]};font-weight:700;'>1:{rr_ratio}</div>
                </div>
              </div>
              <div style='margin-top:12px;padding:10px;background:{C["card"]};border-radius:4px;'>
                <div style='font-size:12px;color:{C["muted"]};{"direction:rtl;text-align:right;" if use_ar else ""}'>
                  🛑 SL: ${ps_sl:,.2f} &nbsp;|&nbsp;
                  🎯 TP1: ${tp1:,.2f} &nbsp;|&nbsp;
                  🎯 TP2: ${tp2:,.2f}
                </div>
              </div>
            </div>""", unsafe_allow_html=True)

    with t2:  # P&L Calculator
        st.markdown(f"<div class='section-label'>{L['calc_pnl_sub']}</div>", unsafe_allow_html=True)
        p1,p2 = st.columns(2)
        with p1:
            pl_entry   = st.number_input("📥 " + ("سعر الدخول" if use_ar else "Entry Price"), value=float(g_ref), step=1.0, key="pl_entry")
            pl_exit    = st.number_input("📤 " + ("سعر الخروج" if use_ar else "Exit Price"), value=float(g_ref*1.01), step=1.0, key="pl_exit")
            pl_qty     = st.number_input("⚖️ " + ("الكمية (جرام)" if use_ar else "Quantity (grams)"), value=100.0, step=10.0, key="pl_qty")
            pl_curr    = st.selectbox("💱", ["USD","JOD","SAR","AED","EGP","KWD","QAR","BHD"], key="pl_curr")
        with p2:
            pl_oz    = pl_qty / 31.1035
            pl_cost  = pl_entry * pl_oz
            pl_val   = pl_exit  * pl_oz
            pl_gross = pl_val - pl_cost
            pl_pct   = (pl_gross / pl_cost * 100) if pl_cost > 0 else 0
            pl_col   = C["green"] if pl_gross >= 0 else C["red"]
            pl_icon  = "▲ PROFIT" if pl_gross >= 0 else "▼ LOSS"
            st.markdown(f"""
            <div style='background:{C['card2']};border:2px solid {pl_col}55;border-radius:8px;padding:24px;text-align:center;margin-top:20px;'>
              <div style='font-size:11px;letter-spacing:.2em;color:{C['muted']};margin-bottom:8px;'>
                {pl_icon if not use_ar else ('▲ ربح' if pl_gross>=0 else '▼ خسارة')}
              </div>
              <div style='font-family:{C['font_m']},monospace;font-size:36px;font-weight:900;color:{pl_col};'>
                ${abs(pl_gross):,.2f}
              </div>
              <div style='font-size:18px;color:{pl_col};margin-top:4px;'>{pl_pct:+.2f}%</div>
              <div style='margin-top:16px;font-size:12px;color:{C['dim']};'>
                {L['cost_lbl']}: ${pl_cost:,.2f} →
                {L['value_lbl']}: ${pl_val:,.2f}
              </div>
              <div style='font-size:11px;color:{C['dim']};margin-top:4px;'>
                {pl_oz:.4f} oz · {pl_qty:.2f}g
              </div>
            </div>""", unsafe_allow_html=True)

    with t3:  # Margin Calculator
        st.markdown(f"<div class='section-label'>{L['calc_margin_sub']}</div>", unsafe_allow_html=True)
        m1,m2 = st.columns(2)
        with m1:
            mg_size    = st.number_input("⚖️ " + ("حجم الصفقة (أوقية)" if use_ar else "Position Size (oz)"), value=1.0, step=0.1, key="mg_size")
            mg_lev     = st.select_slider("🔧 " + ("الرافعة المالية" if use_ar else "Leverage"), options=[1,2,5,10,20,50,100,200], value=10, key="mg_lev")
            mg_price   = st.number_input("💲 " + ("السعر الحالي" if use_ar else "Current Price"), value=float(g_ref), step=1.0, key="mg_price")
        with m2:
            mg_notional = mg_size * mg_price
            mg_margin   = mg_notional / mg_lev
            mg_pip      = mg_size * 1.0  # $1 move per oz
            mg_mc       = mg_price * 0.5  # hypothetical margin call
            st.markdown(f"""
            <div style='background:{C['card2']};border:1px solid {C['gold']}44;border-radius:8px;padding:20px;'>
              <div style='display:grid;grid-template-columns:1fr 1fr;gap:10px;'>
                <div class='stat-card' style='padding:10px;text-align:center;'>
                  <div class='stat-label'>{L['notional']}</div>
                  <div style='font-size:17px;color:{C["gold_pale"]};font-weight:700;'>${mg_notional:,.2f}</div>
                </div>
                <div class='stat-card' style='padding:10px;text-align:center;'>
                  <div class='stat-label'>{L['req_margin']}</div>
                  <div style='font-size:17px;color:{C["gold_hi"]};font-weight:700;'>${mg_margin:,.2f}</div>
                </div>
                <div class='stat-card' style='padding:10px;text-align:center;'>
                  <div class='stat-label'>{L['per_dollar']}</div>
                  <div style='font-size:17px;color:{C["green"]};font-weight:700;'>${mg_pip:,.2f}</div>
                </div>
                <div class='stat-card' style='padding:10px;text-align:center;'>
                  <div class='stat-label'>{L['leverage_lbl']}</div>
                  <div style='font-size:17px;color:{C["accent"]};font-weight:700;'>1:{mg_lev}</div>
                </div>
              </div>
              <div style='margin-top:10px;padding:8px 12px;background:{C["red"]}18;
                          border:1px solid {C["red"]}44;border-radius:4px;'>
                <div style='font-size:11px;color:{C["red"]};'>
                  ⚠️ {'الرافعة العالية تزيد المخاطرة. تأكد من وجود هامش كافٍ.' if use_ar else
                       'High leverage amplifies both gains and losses. Always use stop-loss orders.'}
                </div>
              </div>
            </div>""", unsafe_allow_html=True)

    with t4:  # Break-even
        st.markdown(f"<div class='section-label'>{L['calc_be_sub']}</div>", unsafe_allow_html=True)
        b1,b2 = st.columns(2)
        with b1:
            be_entry    = st.number_input("📥 " + ("سعر الدخول" if use_ar else "Entry Price"), value=float(g_ref), step=1.0, key="be_entry")
            be_spread   = st.number_input("📊 " + ("الفارق (نقاط)" if use_ar else "Spread (USD)"), value=0.5, step=0.1, key="be_spread")
            be_comm_pct = st.number_input("💸 " + ("العمولة %" if use_ar else "Commission %"), value=0.1, step=0.05, key="be_comm")
            be_swap     = st.number_input("🔄 " + ("تكلفة الترحيل/يوم" if use_ar else "Swap/day (USD)"), value=0.5, step=0.1, key="be_swap")
            be_days     = st.number_input("📅 " + ("عدد الأيام" if use_ar else "Days held"), value=1, step=1, key="be_days", min_value=1, max_value=365)
        with b2:
            be_comm_usd = be_entry * be_comm_pct / 100
            be_total_cost = be_spread + be_comm_usd + be_swap * be_days
            be_price    = be_entry + be_total_cost
            be_pct      = be_total_cost / be_entry * 100
            st.markdown(f"""
            <div style='background:{C['card2']};border:1px solid {C['gold']}44;border-radius:8px;padding:24px;text-align:center;'>
              <div class='stat-label' style='margin-bottom:8px;'>{L['breakeven']}</div>
              <div style='font-family:{C['font_m']},monospace;font-size:32px;font-weight:900;color:{C['gold_hi']};'>
                ${be_price:,.3f}
              </div>
              <div style='font-size:14px;color:{C['muted']};margin-top:6px;'>
                +${be_total_cost:,.3f} ({be_pct:.3f}%) {L['above_entry']}
              </div>
              <div style='margin-top:16px;text-align:left;'>
                <div style='font-size:11px;color:{C['dim']};margin:3px 0;'>• {L['spread_lbl']}: ${be_spread:,.3f}</div>
                <div style='font-size:11px;color:{C['dim']};margin:3px 0;'>• {L['commission_lbl']}: ${be_comm_usd:,.3f}</div>
                <div style='font-size:11px;color:{C['dim']};margin:3px 0;'>• {L['swap_lbl']}: ${be_swap*be_days:,.3f} ({be_days}d)</div>
              </div>
            </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: ECONOMIC CALENDAR 📅
# ═══════════════════════════════════════════════════════════════════════════════
elif nav == L["nav_calendar"]:
    use_ar = is_rtl()
    ph("📅  " + L.get("nav_calendar","Economic Calendar").replace("📅  ",""),
       "الأحداث الاقتصادية المؤثرة على الذهب · محدّث شهريًا" if use_ar else
       "Key economic events that move gold prices · Updated monthly")

    today = datetime.date.today()

    # ── Live Gold News from RSS (auto-refreshes every 15 min) ─────────────────
    @st.cache_data(ttl=CACHE_TTL_SCORE)  # 15 min cache — news refreshes automatically
    def fetch_gold_news():
        """Fetch live gold news from free RSS feeds — no API key needed."""
        import urllib.request as _ur, xml.etree.ElementTree as _ET
        feeds = [
            ("Reuters Commodities", "https://feeds.reuters.com/reuters/companyNews"),
            ("Kitco Gold News",     "https://www.kitco.com/rss/"),
            ("FXStreet",            "https://www.fxstreet.com/rss"),
        ]
        articles = []
        gold_keywords = ["gold","ذهب","XAU","bullion","precious","yellow metal","مانسا","commodity"]
        for source, url in feeds:
            try:
                req = _ur.Request(url, headers={"User-Agent":"Mozilla/5.0"})
                with _ur.urlopen(req, timeout=8) as r:
                    xml_data = r.read()
                root = _ET.fromstring(xml_data)
                items = root.findall(".//item") or root.findall(".//{http://www.w3.org/2005/Atom}entry")
                for item in items[:6]:
                    title_el = item.find("title")
                    link_el  = item.find("link")
                    pub_el   = item.find("pubDate") or item.find("{http://www.w3.org/2005/Atom}published")
                    title = title_el.text.strip() if title_el is not None and title_el.text else ""
                    if any(kw.lower() in title.lower() for kw in gold_keywords):
                        articles.append({
                            "source": source,
                            "title":  title,
                            "link":   link_el.text.strip() if link_el is not None and link_el.text else "#",
                            "pub":    pub_el.text.strip()[:16] if pub_el is not None and pub_el.text else "—"
                        })
            except Exception:
                continue
        return articles[:12]

    news_articles = fetch_gold_news()

    news_expanded = len(news_articles) > 0
    with st.expander("📰 " + ("أخبار الذهب المباشرة · تحديث تلقائي كل 15 دقيقة" if use_ar else
                              "Live Gold News · Auto-refreshes every 15 minutes"),
                     expanded=news_expanded):
        if news_articles:
            for art in news_articles:
                sentiment_icon = "🟢" if any(w in art["title"].lower() for w in
                    ["rise","gain","high","rally","surge","climb","up","bullish","strong"]) else                                  "🔴" if any(w in art["title"].lower() for w in
                    ["fall","drop","low","plunge","decline","down","bearish","weak","pressure"]) else "⚪"
                st.markdown(f"""
                <div style='padding:8px 12px;border-bottom:1px solid {C['border']}22;'>
                  <div style='display:flex;gap:8px;align-items:flex-start;'>
                    <span style='font-size:14px;margin-top:1px;'>{sentiment_icon}</span>
                    <div>
                      <div style='font-size:13px;color:{C['text']};font-weight:500;line-height:1.4;'>
                        {art['title']}
                      </div>
                      <div style='font-size:10px;color:{C['dim']};margin-top:3px;'>
                        📡 {art['source']} · 🕐 {art['pub']}
                      </div>
                    </div>
                  </div>
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style='padding:12px;text-align:center;'>
              <div style='font-size:12px;color:{C['muted']};'>
                {'جاري تحميل الأخبار... تأكد من الاتصال بالإنترنت' if use_ar else
                 'Loading news... ensure internet connection is available'}
              </div>
              <div style='font-size:10px;color:{C['dim']};margin-top:4px;'>
                {'المصادر: Reuters · Kitco Gold · FXStreet' if use_ar else
                 'Sources: Reuters · Kitco Gold · FXStreet'}
              </div>
            </div>""", unsafe_allow_html=True)
        if st.button("🔄 " + ("تحديث الأخبار الآن" if use_ar else "Refresh News Now"),
                     key="news_refresh"):
            fetch_gold_news.clear()
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # Economic events that move gold — static + Arab markets
    EVENTS = [
        {"date":"2026-03-18","time":"19:00","event":"FOMC Meeting Minutes","impact":"HIGH","currency":"USD",
         "forecast":"Hawkish tone expected","actual":"—","effect":"bearish_gold","icon":"🏦"},
        {"date":"2026-03-19","time":"09:00","event":"EU CPI (Feb)","impact":"MED","currency":"EUR",
         "forecast":"2.3% YoY","actual":"—","effect":"bullish_gold","icon":"📊"},
        {"date":"2026-03-21","time":"13:30","event":"US Existing Home Sales","impact":"LOW","currency":"USD",
         "forecast":"4.12M","actual":"—","effect":"neutral","icon":"🏠"},
        {"date":"2026-03-25","time":"13:30","event":"US Durable Goods Orders","impact":"MED","currency":"USD",
         "forecast":"-1.2%","actual":"—","effect":"neutral","icon":"🏭"},
        {"date":"2026-03-28","time":"13:30","event":"US Core PCE Price Index","impact":"HIGH","currency":"USD",
         "forecast":"2.7% YoY","actual":"—","effect":"bullish_gold","icon":"💹"},
        {"date":"2026-04-01","time":"13:30","event":"US NFP (Non-Farm Payrolls)","impact":"HIGH","currency":"USD",
         "forecast":"195K","actual":"—","effect":"bearish_gold","icon":"👷"},
        {"date":"2026-04-04","time":"07:00","event":"Saudi Arabia PMI","impact":"MED","currency":"SAR",
         "forecast":"57.2","actual":"—","effect":"neutral","icon":"🇸🇦"},
        {"date":"2026-04-10","time":"13:30","event":"US CPI (March)","impact":"HIGH","currency":"USD",
         "forecast":"3.1% YoY","actual":"—","effect":"bullish_gold","icon":"📈"},
        {"date":"2026-04-15","time":"20:00","event":"Fed Chair Speech (Powell)","impact":"HIGH","currency":"USD",
         "forecast":"Rate guidance","actual":"—","effect":"bearish_gold","icon":"🎤"},
        {"date":"2026-04-17","time":"13:30","event":"US Retail Sales","impact":"MED","currency":"USD",
         "forecast":"+0.4%","actual":"—","effect":"neutral","icon":"🛒"},
        {"date":"2026-04-22","time":"18:00","event":"ECB Rate Decision","impact":"HIGH","currency":"EUR",
         "forecast":"Hold at 3.75%","actual":"—","effect":"bullish_gold","icon":"🏛️"},
        {"date":"2026-05-01","time":"14:00","event":"FOMC Rate Decision","impact":"HIGH","currency":"USD",
         "forecast":"Hold at 4.25-4.5%","actual":"—","effect":"bullish_gold","icon":"🏦"},
        {"date":"2026-05-09","time":"13:30","event":"US CPI (April)","impact":"HIGH","currency":"USD",
         "forecast":"2.9% YoY","actual":"—","effect":"bullish_gold","icon":"📈"},
        {"date":"2026-05-15","time":"08:00","event":"Jordan Gold Imports (Q1)","impact":"LOW","currency":"JOD",
         "forecast":"12.4 tons","actual":"—","effect":"neutral","icon":"🇯🇴"},
        {"date":"2026-06-11","time":"19:00","event":"FOMC Rate Decision","impact":"HIGH","currency":"USD",
         "forecast":"Possible -25bp cut","actual":"—","effect":"very_bullish_gold","icon":"🏦"},
        # ── Arab Market Events ────────────────────────────────────────────────
        {"date":"2026-03-20","time":"06:00","event":"Jordan Central Bank Rate Decision","impact":"MED","currency":"JOD",
         "forecast":"Hold at 7.5%","actual":"—","effect":"neutral","icon":"🇯🇴"},
        {"date":"2026-03-26","time":"08:00","event":"Jordan Gold Trade Balance (Feb)","impact":"LOW","currency":"JOD",
         "forecast":"Import surplus","actual":"—","effect":"bullish_gold","icon":"🇯🇴"},
        {"date":"2026-04-03","time":"08:00","event":"UAE PMI (March)","impact":"MED","currency":"AED",
         "forecast":"55.1","actual":"—","effect":"neutral","icon":"🇦🇪"},
        {"date":"2026-04-06","time":"07:00","event":"Saudi Arabia Monetary Policy Meeting","impact":"HIGH","currency":"SAR",
         "forecast":"Hold — follow Fed","actual":"—","effect":"neutral","icon":"🇸🇦"},
        {"date":"2026-04-08","time":"08:00","event":"Kuwait Gold Reserve Update (Q1)","impact":"LOW","currency":"KWD",
         "forecast":"79 tons","actual":"—","effect":"bullish_gold","icon":"🇰🇼"},
        {"date":"2026-04-12","time":"09:00","event":"Egypt CPI (March)","impact":"MED","currency":"EGP",
         "forecast":"24.1% YoY","actual":"—","effect":"bullish_gold","icon":"🇪🇬"},
        {"date":"2026-04-20","time":"08:00","event":"Dubai Gold & Jewellery Group Report","impact":"MED","currency":"AED",
         "forecast":"Q1 trade data","actual":"—","effect":"bullish_gold","icon":"🇦🇪"},
        {"date":"2026-04-25","time":"10:00","event":"Jordan Dept of Statistics: Gold Prices","impact":"LOW","currency":"JOD",
         "forecast":"March average","actual":"—","effect":"neutral","icon":"🇯🇴"},
        {"date":"2026-05-05","time":"07:00","event":"Saudi Arabia CPI (April)","impact":"MED","currency":"SAR",
         "forecast":"2.4% YoY","actual":"—","effect":"neutral","icon":"🇸🇦"},
        {"date":"2026-05-12","time":"08:00","event":"Qatar Central Bank Monetary Meeting","impact":"MED","currency":"QAR",
         "forecast":"Hold","actual":"—","effect":"neutral","icon":"🇶🇦"},
        {"date":"2026-05-20","time":"09:00","event":"Bahrain Gold Bullion Bank Report","impact":"LOW","currency":"BHD",
         "forecast":"Q1 holdings","actual":"—","effect":"bullish_gold","icon":"🇧🇭"},
        {"date":"2026-06-01","time":"08:00","event":"UAE Gold Trade Report (Q2 Start)","impact":"MED","currency":"AED",
         "forecast":"Dubai exports","actual":"—","effect":"bullish_gold","icon":"🇦🇪"},
    ]

    # Filter/sort options
    fc1,fc2,fc3 = st.columns(3)
    with fc1:
        cal_filter = st.selectbox(
            ("تصفية حسب التأثير" if use_ar else "Filter by Impact"),
            (["الكل","عالي","متوسط","منخفض"] if use_ar else ["All","HIGH","MED","LOW"]),
            key="cal_filter"
        )
    with fc2:
        cal_currency = st.selectbox(
            ("تصفية حسب العملة" if use_ar else "Filter by Currency"),
            ["All","USD","EUR","JOD","SAR","AED","EGP","KWD","QAR","BHD"],
            key="cal_currency"
        )
    with fc3:
        cal_upcoming = st.toggle(
            ("الأحداث القادمة فقط" if use_ar else "Upcoming only"),
            value=True, key="cal_upcoming"
        )

    # Apply filters
    filtered = EVENTS.copy()
    if cal_upcoming:
        filtered = [e for e in filtered if datetime.date.fromisoformat(e["date"]) >= today]
    if cal_filter not in ["All","الكل"]:
        impact_map = {"عالي":"HIGH","متوسط":"MED","منخفض":"LOW"}
        fi = impact_map.get(cal_filter, cal_filter)
        filtered = [e for e in filtered if e["impact"] == fi]
    if cal_currency != "All":
        filtered = [e for e in filtered if e["currency"] == cal_currency]

    st.markdown("<br>", unsafe_allow_html=True)

    if not filtered:
        st.info("لا توجد أحداث مطابقة" if use_ar else "No matching events found")
    else:
        impact_colors = {"HIGH":C["red"],"MED":C["gold"],"LOW":C["muted"]}
        effect_colors = {
            "bullish_gold":      C["green"],
            "very_bullish_gold": C["green"],
            "bearish_gold":      C["red"],
            "neutral":           C["muted"],
        }
        effect_labels_ar = {
            "bullish_gold":"📈 صعودي للذهب","very_bullish_gold":"🚀 صعودي قوي جداً",
            "bearish_gold":"📉 هبوطي للذهب","neutral":"➡️ محايد",
        }
        effect_labels_en = {
            "bullish_gold":"📈 Bullish for Gold","very_bullish_gold":"🚀 Very Bullish Gold",
            "bearish_gold":"📉 Bearish for Gold","neutral":"➡️ Neutral",
        }

        # Group by date
        from itertools import groupby
        for ev_date_str, group in groupby(filtered, key=lambda x: x["date"]):
            ev_date = datetime.date.fromisoformat(ev_date_str)
            days_away = (ev_date - today).days
            if days_away == 0:
                date_badge = f"<span style='color:{C['green']};font-weight:700;'>{L['today_lbl']}</span>"
            elif days_away == 1:
                date_badge = f"<span style='color:{C['gold']};'>{L['tomorrow_lbl']}</span>"
            elif days_away < 0:
                date_badge = f"<span style='color:{C['dim']};'>{'منذ ' + str(abs(days_away)) + ' يوم' if use_ar else str(abs(days_away)) + 'd ago'}</span>"
            else:
                date_badge = f"<span style='color:{C['muted']};'>{'خلال ' + str(days_away) + L['days_lbl'] + str(days_away) + 'd'}</span>"

            st.markdown(f"""
            <div style='display:flex;align-items:center;gap:10px;margin:16px 0 6px;'>
              <div style='font-family:{C['font_m']},monospace;font-size:13px;font-weight:700;
                          color:{C['gold_pale']};'>{ev_date.strftime('%a %d %b %Y')}</div>
              <div style='font-size:12px;'>{date_badge}</div>
            </div>""", unsafe_allow_html=True)

            for ev in group:
                ic  = impact_colors.get(ev["impact"], C["muted"])
                ef  = effect_colors.get(ev["effect"], C["muted"])
                elbl= (effect_labels_ar if use_ar else effect_labels_en).get(ev["effect"],"—")
                st.markdown(f"""
                <div style='background:{C['card2']};border:1px solid {C['border2']};
                            border-left:4px solid {ic};border-radius:6px;
                            padding:12px 16px;margin-bottom:6px;
                            display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;'>
                  <div>
                    <div style='display:flex;align-items:center;gap:8px;margin-bottom:4px;'>
                      <span style='font-size:18px;'>{ev['icon']}</span>
                      <span style='font-size:14px;font-weight:700;color:{C['text']};'>{ev['event']}</span>
                      <span style='background:{ic}22;color:{ic};font-size:9px;letter-spacing:.1em;
                                   padding:2px 8px;border-radius:10px;border:1px solid {ic}44;'>
                        {ev['impact']}
                      </span>
                      <span style='font-size:11px;color:{C['dim']};'>{ev['currency']}</span>
                    </div>
                    <div style='font-size:11px;color:{C['dim']};'>
                      🕐 {ev['time']} UTC &nbsp;·&nbsp;
                      {L['forecast_lbl']} {ev['forecast']} &nbsp;·&nbsp;
                      {L['actual_lbl']} {ev['actual']}
                    </div>
                  </div>
                  <div style='text-align:right;'>
                    <div style='font-size:12px;color:{ef};font-weight:700;'>{elbl}</div>
                  </div>
                </div>""", unsafe_allow_html=True)

    # Explanation box
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='background:{C['card']};border:1px solid {C['border']};border-radius:6px;padding:14px 18px;'>
      <div style='font-size:11px;color:{C['muted']};{"direction:rtl;text-align:right;" if use_ar else ""}'>
        {'💡 كيف تؤثر الأحداث على الذهب: البيانات التضخمية المرتفعة (CPI/PCE) تدعم الذهب · قرارات رفع الفائدة تضغط عليه · الأزمات الجيوسياسية تدفعه للأعلى · تقارير التوظيف القوية تضعفه عادةً.' if use_ar else
         '💡 How events affect gold: High inflation data (CPI/PCE) supports gold · Rate hike decisions pressure it · Geopolitical crises push it higher · Strong employment reports usually weaken it.'}
      </div>
    </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: FEAR & GREED SENTIMENT 🌡️
# ═══════════════════════════════════════════════════════════════════════════════
elif nav == L["nav_sentiment"]:
    use_ar = is_rtl()
    ph("🌡️  " + L.get("nav_sentiment","Sentiment").replace("🌡️  ",""),
       "مؤشر الخوف والجشع للذهب · مبني على بيانات السوق الحية" if use_ar else
       "Gold Fear & Greed Index · Built from live market data")

    # Build the composite score from live indicators — 1y lookback for accuracy
    @st.cache_data(ttl=CACHE_TTL_HISTORY)
    def get_sentiment_data():
        """Compute the composite Gold Fear & Greed Index from live market data.
        
            Returns
            -------
            dict
                Keys: composite (0-100), rsi, price, ma50, ma200, vix, dxy_pct,
                us10y, bb_pct and component-level scores.
            """
        # Use 1y for proper MA200 and volatility baseline
        df = fetch_history("1y","GC=F")
        if df.empty: return {}
        cl = find_col(df,["Close","Close_GC=F"])
        if not cl: return {}
        c = df[cl].dropna()
        if len(c) < 15: return {}

        # ── RSI(14) ──────────────────────────────────────────────────────────
        d  = c.diff()
        ag = d.clip(lower=0).rolling(14).mean()
        al = (-d.clip(upper=0)).rolling(14).mean()
        rsi = float((100-(100/(1+ag/al.replace(0,1e-9)))).iloc[-1])

        # ── Moving averages ───────────────────────────────────────────────────
        ma50  = float(c.rolling(50).mean().iloc[-1])  if len(c)>=50  else float(c.mean())
        ma200 = float(c.rolling(200).mean().iloc[-1]) if len(c)>=200 else float(c.mean())
        price = float(c.iloc[-1])

        # ── Momentum: 30d, 90d ───────────────────────────────────────────────
        mom30 = float((c.iloc[-1]/c.iloc[-30]-1)*100) if len(c)>=31 else 0.0
        mom90 = float((c.iloc[-1]/c.iloc[-90]-1)*100) if len(c)>=91 else 0.0

        # ── Volatility: current ATR vs 1-year average ATR ────────────────────
        atr_series = c.diff().abs().rolling(14).mean()
        atr_now    = float(atr_series.iloc[-1])
        atr_avg    = float(atr_series.mean())
        vol_ratio  = atr_now / atr_avg if atr_avg > 0 else 1.0

        # ── Bollinger Band position (% inside band) ───────────────────────────
        bm  = c.rolling(20).mean()
        bs  = c.rolling(20).std()
        bb_upper = float((bm + 2*bs).iloc[-1])
        bb_lower = float((bm - 2*bs).iloc[-1])
        bb_pct   = (price - bb_lower) / (bb_upper - bb_lower) if (bb_upper - bb_lower) > 0 else 0.5

        # ── Volume trend (proxy: high-low range as volume proxy) ─────────────
        hl_col = find_col(df, ["High","High_GC=F"])
        ll_col = find_col(df, ["Low","Low_GC=F"])
        hl_range = 0.5  # default neutral
        if hl_col and ll_col:
            hl = df[hl_col].dropna(); ll = df[ll_col].dropna()
            if len(hl)>=20:
                recent_range = float((hl-ll).tail(5).mean())
                avg_range    = float((hl-ll).mean())
                hl_range     = recent_range / avg_range if avg_range > 0 else 1.0

        return dict(rsi=rsi, ma50=ma50, ma200=ma200, price=price,
                    mom30=mom30, mom90=mom90, vol_ratio=vol_ratio,
                    bb_pct=bb_pct, hl_range=hl_range,
                    bb_upper=bb_upper, bb_lower=bb_lower)

    sd = get_sentiment_data()
    rsi_s    = sd.get("rsi",   50)
    ma50_s   = sd.get("ma50",  g_ref)
    ma200_s  = sd.get("ma200", g_ref)
    price_s  = sd.get("price", g_ref)
    mom30_s  = sd.get("mom30", 0)
    mom90_s  = sd.get("mom90", 0)
    vol_s    = sd.get("vol_ratio", 1)
    bb_pct_s = sd.get("bb_pct",   0.5)
    hl_rng_s = sd.get("hl_range", 1.0)
    vix_s    = live["vix"]["price"]
    dxy_s    = live["dxy"]["pct"]
    us10y_s  = live["us10y"]["price"]
    # Gold/Silver ratio — high ratio = gold expensive vs silver, slight bearish signal
    gsr      = g_ref / live["silver"]["price"] if live["silver"]["price"] > 0 else 80

    def clamp(v, lo=0, hi=100): return max(lo, min(hi, v))

    # ── 9 Components, each 0–100 ──────────────────────────────────────────────
    # 1. RSI(14): <30=extreme fear/buy, >70=extreme greed/sell
    rsi_score   = clamp((rsi_s - 25) / 50 * 100)

    # 2. Price vs MA50 (trend health): deviation mapped ±5% → 0-100
    ma50_score  = clamp(50 + (price_s - ma50_s) / ma50_s * 1000)

    # 3. Price vs MA200 (long-term trend): deviation mapped ±10% → 0-100
    ma200_score = clamp(50 + (price_s - ma200_s) / ma200_s * 500)

    # 4. 30-day momentum: ±10% maps to 0-100
    mom30_score = clamp(50 + mom30_s * 5)

    # 5. 90-day momentum (longer trend confirmation): ±20% → 0-100
    mom90_score = clamp(50 + mom90_s * 2.5)

    # 6. VIX Fear Index: high VIX → market panic → gold demand → bullish
    #    VIX 10=calm(greed), VIX 40=panic(fear) — FOR GOLD: high VIX = bullish
    vix_score   = clamp((vix_s - 10) / 30 * 100)

    # 7. DXY change: falling dollar = bullish gold
    dxy_score   = clamp(50 - dxy_s * 20)

    # 8. US10Y Yield: higher real rates = bearish for gold
    #    Yield 3%=neutral, 6%=bearish, 1%=bullish
    us10y_score = clamp(50 - (us10y_s - 3.5) * 20)

    # 9. Bollinger Band position: >80% = overbought, <20% = oversold
    bb_score    = clamp((1 - bb_pct_s) * 100)   # inverted: near lower = fear = buy

    # Weighted composite (weights sum to 1.0)
    weights = [0.18, 0.12, 0.10, 0.15, 0.10, 0.12, 0.10, 0.08, 0.05]
    scores  = [rsi_score, ma50_score, ma200_score, mom30_score, mom90_score,
               vix_score, dxy_score, us10y_score, bb_score]
    composite = clamp(sum(w*s for w,s in zip(weights, scores)))

    # Label
    if composite < 20:
        label_en = "Extreme Fear";   label_ar = "خوف شديد";    gauge_col = "#FF2244"
    elif composite < 40:
        label_en = "Fear";           label_ar = "خوف";         gauge_col = "#FF7744"
    elif composite < 60:
        label_en = "Neutral";        label_ar = "محايد";       gauge_col = C["gold"]
    elif composite < 80:
        label_en = "Greed";          label_ar = "جشع";         gauge_col = "#44CC88"
    else:
        label_en = "Extreme Greed";  label_ar = "جشع شديد";   gauge_col = "#00FF88"

    label = label_ar if use_ar else label_en

    # ── Gauge visualization ───────────────────────────────────────────────────
    import math
    needle_angle = -150 + composite * 3  # -150° to +150° range
    nx = 160 + 100 * math.cos(math.radians(needle_angle))
    ny = 155 - 100 * math.sin(math.radians(needle_angle))

    gauge_html = f"""
    <div style='text-align:center;padding:10px 0;'>
      <svg viewBox="0 0 320 200" width="100%" xmlns="http://www.w3.org/2000/svg">
        <defs>
          <linearGradient id="gauge_grad" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%"   stop-color="#FF2244"/>
            <stop offset="25%"  stop-color="#FF7744"/>
            <stop offset="50%"  stop-color="{C['gold']}"/>
            <stop offset="75%"  stop-color="#44CC88"/>
            <stop offset="100%" stop-color="#00FF88"/>
          </linearGradient>
        </defs>
        <!-- Gauge arc background (grey) -->
        <path d="M 60 155 A 100 100 0 0 1 260 155"
              fill="none" stroke="{C['border2']}" stroke-width="18" stroke-linecap="round"/>
        <!-- Colored gauge arc -->
        <path d="M 60 155 A 100 100 0 0 1 260 155"
              fill="none" stroke="url(#gauge_grad)" stroke-width="18" stroke-linecap="round"
              stroke-dasharray="314" stroke-dashoffset="{314*(1-composite/100):.1f}"/>
        <!-- Needle -->
        <line x1="160" y1="155" x2="{nx:.1f}" y2="{ny:.1f}"
              stroke="{gauge_col}" stroke-width="3" stroke-linecap="round"/>
        <circle cx="160" cy="155" r="8" fill="{C['card']}" stroke="{gauge_col}" stroke-width="2"/>
        <!-- Score -->
        <text x="160" y="140" text-anchor="middle"
              font-family="monospace" font-size="32" font-weight="900"
              fill="{gauge_col}">{composite:.0f}</text>
        <!-- Label -->
        <text x="160" y="178" text-anchor="middle"
              font-family="Cairo,serif" font-size="16" font-weight="700"
              fill="{gauge_col}">{label}</text>
        <!-- Zone labels -->
        <text x="52"  y="175" text-anchor="middle" font-size="9" fill="#FF2244" font-family="serif">
          {L['fear_lbl']}
        </text>
        <text x="268" y="175" text-anchor="middle" font-size="9" fill="#00FF88" font-family="serif">
          {L['greed_lbl']}
        </text>
        <text x="160" y="72"  text-anchor="middle" font-size="8"  fill="{C['muted']}" font-family="serif">
          {L['neutral_sig']}
        </text>
      </svg>
    </div>"""

    g1, g2 = st.columns([2,3], gap="large")
    with g1:
        st.components.v1.html(gauge_html, height=220, scrolling=False)

    with g2:
        st.markdown(f"""
        <div style='background:{C['card2']};border:1px solid {gauge_col}44;border-radius:8px;
                    padding:18px 22px;margin-top:10px;'>
          <div style='font-size:12px;color:{C['muted']};margin-bottom:12px;
                      {"direction:rtl;text-align:right;" if use_ar else ""}'>
            {L['what_means_gold']}
          </div>
          {"<div style='font-size:13px;color:"+C['text']+";direction:rtl;text-align:right;line-height:1.8;'>" if use_ar else "<div style='font-size:13px;color:"+C['text']+";line-height:1.8;'>"}
            { ('🟢 المستثمرون خائفون — الذهب قد يكون فرصة شراء. الخوف يدفع المال نحو الملاذات الآمنة.' if composite < 40 else
               '🟡 السوق محايدة — انتظر إشارات أوضح قبل اتخاذ قرار.' if composite < 60 else
               '🔴 المستثمرون جشعون — السوق ممتدة. قد يكون الوقت مناسبًا لجني الأرباح أو تضييق وقف الخسارة.') if use_ar else
             ('🟢 Investors are fearful — gold may be a buying opportunity. Fear drives capital to safe havens.' if composite < 40 else
              '🟡 Market is neutral — wait for clearer signals before acting.' if composite < 60 else
              '🔴 Investors are greedy — market may be overextended. Consider taking profits or tightening stop-losses.')
            }
          </div>
        </div>""", unsafe_allow_html=True)

    # Component breakdown
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"<div class='section-label'>{L['index_components']}</div>", unsafe_allow_html=True)
    components = [
        ("RSI(14)",       rsi_score,   f"{rsi_s:.1f}",
         ('ذروة بيع 🟢' if rsi_s<30 else 'ذروة شراء 🔴' if rsi_s>70 else 'محايد') if use_ar else ('Oversold 🟢' if rsi_s<30 else 'Overbought 🔴' if rsi_s>70 else 'Neutral')),
        ("MA50 Trend",    ma50_score,  f"${ma50_s:,.0f}",
         ('فوق MA50 ✅' if price_s>ma50_s else 'تحت MA50 ⚠️') if use_ar else ('Above MA50 ✅' if price_s>ma50_s else 'Below MA50 ⚠️')),
        ("MA200 Trend",   ma200_score, f"${ma200_s:,.0f}",
         ('اتجاه صاعد ✅' if price_s>ma200_s else 'اتجاه هابط ⚠️') if use_ar else ('Bull trend ✅' if price_s>ma200_s else 'Bear trend ⚠️')),
        ("Momentum 30d",  mom30_score, f"{mom30_s:+.2f}%", ""),
        ("Momentum 90d",  mom90_score, f"{mom90_s:+.2f}%", ""),
        ("VIX Fear",      vix_score,   f"{vix_s:.1f}",
         ('خوف عالٍ ↑ ذهب 🟢' if vix_s>22 else 'هدوء 🟡') if use_ar else ('High fear ↑ gold 🟢' if vix_s>22 else 'Calm 🟡')),
        ("DXY Change",    dxy_score,   f"{dxy_s:+.2f}%",
         ('دولار ضعيف ↑ ذهب 🟢' if dxy_s<-0.2 else 'دولار قوي ↓ ذهب 🔴' if dxy_s>0.2 else '—') if use_ar else ('Weak USD ↑ gold 🟢' if dxy_s<-0.2 else 'Strong USD ↓ gold 🔴' if dxy_s>0.2 else '—')),
        ("US10Y Yield",   us10y_score, f"{us10y_s:.2f}%",
         ('عائد منخفض ↑ ذهب 🟢' if us10y_s<4 else 'عائد مرتفع ↓ ذهب 🔴') if use_ar else ('Low yield ↑ gold 🟢' if us10y_s<4 else 'High yield ↓ gold 🔴')),
        ("Bollinger",     bb_score,    f"{bb_pct_s*100:.0f}%",
         ('قرب الحد السفلي 🟢' if bb_pct_s<0.25 else 'قرب الحد العلوي 🔴' if bb_pct_s>0.75 else '—') if use_ar else ('Near lower band 🟢' if bb_pct_s<0.25 else 'Near upper band 🔴' if bb_pct_s>0.75 else '—')),
    ]
    crows = [components[i:i+4] for i in range(0,len(components),4)]
    for crow in crows:
        cc = st.columns(len(crow))
        for col,(name,score,val,note) in zip(cc,crow):
            bar_col = "#00FF88" if score>65 else ("#FF2244" if score<35 else C["gold"])
            with col:
                st.markdown(f"""
                <div class='stat-card' style='text-align:center;padding:10px;'>
                  <div class='stat-label' style='font-size:9px;'>{name}</div>
                  <div style='font-family:{C['font_m']},monospace;font-size:16px;
                              font-weight:700;color:{bar_col};margin:4px 0;'>{score:.0f}</div>
                  <div style='width:100%;height:4px;background:{C['border']};border-radius:2px;'>
                    <div style='width:{score:.0f}%;height:100%;background:{bar_col};border-radius:2px;'></div>
                  </div>
                  <div style='font-size:9px;color:{C['dim']};margin-top:4px;'>{val}</div>
                  <div style='font-size:9px;color:{C['muted']};'>{note}</div>
                </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: MARKET SESSIONS CLOCK 🕐
# ═══════════════════════════════════════════════════════════════════════════════
elif nav == L["nav_sessions"]:
    use_ar = is_rtl()
    ph("🕐  " + L.get("nav_sessions","Market Sessions").replace("🕐  ",""),
       "ساعات عمل أسواق الذهب العالمية · المناطق الزمنية المباشرة" if use_ar else
       "Global gold market trading hours · Live time zones")

    now_utc = datetime.datetime.utcnow()

    SESSIONS = [
        # ── Arab Markets (shown first as primary) ────────────────────────────
        {"name":"Amman",     "name_ar":"عمّان",    "flag":"🇯🇴","open_h":7, "close_h":15,"tz_off":3,
         "color":"#F5C830","desc_en":"Jordan Jewellery Market (JJM) — PRIMARY","desc_ar":"السوق الذهبي الأردني (JJM) — الرئيسي",
         "primary":True},
        {"name":"Riyadh",    "name_ar":"الرياض",   "flag":"🇸🇦","open_h":7, "close_h":15,"tz_off":3,
         "color":"#00C9A7","desc_en":"Saudi Gold & Jewellery Market","desc_ar":"سوق الذهب السعودي"},
        {"name":"Dubai",     "name_ar":"دبي",      "flag":"🇦🇪","open_h":6, "close_h":15,"tz_off":4,
         "color":"#FFB347","desc_en":"Dubai Gold Souk — world's largest physical market","desc_ar":"سوق دبي للذهب — أكبر سوق فيزيائي"},
        {"name":"Kuwait",    "name_ar":"الكويت",   "flag":"🇰🇼","open_h":8, "close_h":14,"tz_off":3,
         "color":"#9B59B6","desc_en":"Kuwait Gold Market","desc_ar":"سوق الذهب الكويتي"},
        {"name":"Cairo",     "name_ar":"القاهرة",  "flag":"🇪🇬","open_h":8, "close_h":16,"tz_off":2,
         "color":"#E67E22","desc_en":"Egypt Gold Exchange","desc_ar":"بورصة الذهب المصرية"},
        {"name":"Doha",      "name_ar":"الدوحة",   "flag":"🇶🇦","open_h":7, "close_h":15,"tz_off":3,
         "color":"#1ABC9C","desc_en":"Qatar Gold Souk","desc_ar":"سوق الذهب القطري"},
        # ── International ────────────────────────────────────────────────────
        {"name":"London",    "name_ar":"لندن",     "flag":"🇬🇧","open_h":8, "close_h":17,"tz_off":0,
         "color":"#E74C3C","desc_en":"LBMA — price benchmark for ALL markets","desc_ar":"LBMA — المرجع الرسمي لأسعار الذهب العالمية"},
        {"name":"New York",  "name_ar":"نيويورك",  "flag":"🇺🇸","open_h":13,"close_h":22,"tz_off":-5,
         "color":"#2ECC71","desc_en":"COMEX futures — most traded gold contracts","desc_ar":"COMEX — أكثر عقود الذهب الآجلة تداولاً"},
        {"name":"Tokyo",     "name_ar":"طوكيو",    "flag":"🇯🇵","open_h":0, "close_h":9, "tz_off":9,
         "color":"#FF6B6B","desc_en":"TOCOM — Asian futures","desc_ar":"TOCOM — عقود آجلة آسيوية"},
        {"name":"Shanghai",  "name_ar":"شنغهاي",   "flag":"🇨🇳","open_h":1, "close_h":9, "tz_off":8,
         "color":"#4DA6FF","desc_en":"SGE — world's largest physical gold exchange","desc_ar":"SGE — أكبر بورصة ذهب فيزيائي"},
    ]

    def is_session_open(sess, now_utc):
        """Return True when a trading session is currently open.
        
            Parameters
            ----------
            sess : dict
                Session config with ``open`` and ``close`` keys (UTC hour floats).
            now_utc : datetime.datetime
                Current UTC time.
            """
        local_h = (now_utc.hour + sess["tz_off"]) % 24
        o, c = sess["open_h"], sess["close_h"]
        if o < c:
            return o <= local_h < c
        else:
            return local_h >= o or local_h < c

    def local_time(sess, now_utc):
        import datetime as _dt
        local = now_utc + _dt.timedelta(hours=sess["tz_off"])
        return local.strftime("%H:%M")

    def minutes_to_event(sess, now_utc):
        local_h = (now_utc.hour + sess["tz_off"]) % 24
        local_m = now_utc.minute
        total_m = local_h * 60 + local_m
        o, c = sess["open_h"] * 60, sess["close_h"] * 60
        open_in  = (o - total_m) % (24*60)
        close_in = (c - total_m) % (24*60)
        return open_in, close_in

    # Big session clocks
    st.markdown(f"<div class='section-label'>{L['market_status_now']} · {now_utc.strftime('%H:%M UTC')}</div>", unsafe_allow_html=True)

    # Show Arab markets first (separate section), then international
    arab_sess  = [s for s in SESSIONS if s.get("primary") or s["flag"] in ["🇯🇴","🇸🇦","🇦🇪","🇰🇼","🇪🇬","🇶🇦"]]
    intl_sess  = [s for s in SESSIONS if s not in arab_sess]

    st.markdown(f"<div class='section-label'>{L['arab_mkts_primary']}</div>", unsafe_allow_html=True)
    sc = st.columns(3)
    for i, sess in enumerate(arab_sess):
        is_open   = is_session_open(sess, now_utc)
        loc_time  = local_time(sess, now_utc)
        oi, ci    = minutes_to_event(sess, now_utc)
        status_lbl= ("مفتوح ●" if is_open else "مغلق ○") if use_ar else ("OPEN ●" if is_open else "CLOSED ○")
        status_col= sess["color"] if is_open else C["dim"]
        next_event= (f"{L['closes_in']} {ci//60}h {ci%60}m" if is_open else
                     f"{L['opens_in']} {oi//60}h {oi%60}m")
        hours_str = f"{sess['open_h']:02d}:00 – {sess['close_h']:02d}:00 UTC"
        with sc[i % 3]:
            st.markdown(f"""
            <div style='background:{C['card2']};border:2px solid {status_col}66;border-radius:10px;
                        padding:18px;text-align:center;margin-bottom:12px;
                        {"box-shadow:0 0 16px "+sess["color"]+"44;" if is_open else ""}'>
              <div style='font-size:28px;'>{sess['flag']}</div>
              <div style='font-family:{C['font_h']},serif;font-size:16px;font-weight:700;
                          color:{C['text']};margin-top:6px;'>
                {sess['name_ar'] if use_ar else sess['name']}
              </div>
              <div style='font-family:{C['font_m']},monospace;font-size:22px;font-weight:900;
                          color:{status_col};margin:6px 0;'>{loc_time}</div>
              <div style='font-size:13px;font-weight:700;color:{status_col};
                          letter-spacing:.1em;margin-bottom:6px;'>{status_lbl}</div>
              <div style='font-size:10px;color:{C['muted']};'>{next_event}</div>
              <div style='font-size:9px;color:{C['dim']};margin-top:4px;'>{hours_str}</div>
              <div style='font-size:10px;color:{C['muted']};font-style:italic;margin-top:4px;'>
                {sess['desc_ar'] if use_ar else sess['desc_en']}
              </div>
            </div>""", unsafe_allow_html=True)

    st.markdown(f"<div class='section-label'>{L['intl_mkts_lbl']}</div>", unsafe_allow_html=True)
    sc2 = st.columns(4)
    for i, sess in enumerate(intl_sess):
        is_open   = is_session_open(sess, now_utc)
        loc_time  = local_time(sess, now_utc)
        oi, ci    = minutes_to_event(sess, now_utc)
        status_lbl= ("مفتوح ●" if is_open else "مغلق ○") if use_ar else ("OPEN ●" if is_open else "CLOSED ○")
        status_col= sess["color"] if is_open else C["dim"]
        next_event= (f"{L['closes_in']} {ci//60}h {ci%60}m" if is_open else
                     f"{L['opens_in']} {oi//60}h {oi%60}m")
        hours_str = f"{sess['open_h']:02d}:00 – {sess['close_h']:02d}:00 UTC"
        with sc2[i % 4]:
            st.markdown(f"""
            <div style='background:{C['card2']};border:2px solid {status_col}55;border-radius:10px;
                        padding:14px;text-align:center;margin-bottom:10px;
                        {"box-shadow:0 0 14px "+sess["color"]+"33;" if is_open else ""}'>
              <div style='font-size:24px;'>{sess['flag']}</div>
              <div style='font-size:13px;font-weight:700;color:{C['text']};margin-top:4px;'>
                {sess['name_ar'] if use_ar else sess['name']}</div>
              <div style='font-family:{C['font_m']},monospace;font-size:18px;font-weight:900;
                          color:{status_col};margin:4px 0;'>{loc_time}</div>
              <div style='font-size:11px;font-weight:700;color:{status_col};'>{status_lbl}</div>
              <div style='font-size:9px;color:{C['muted']};'>{next_event}</div>
              <div style='font-size:8px;color:{C['dim']};margin-top:2px;'>{hours_str}</div>
              <div style='font-size:9px;color:{C['muted']};font-style:italic;margin-top:3px;'>
                {sess['desc_ar'] if use_ar else sess['desc_en']}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 24h timeline bar — now uses all sessions
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"<div class='section-label'>{L['timeline_24h']}</div>", unsafe_allow_html=True)

    timeline_html = f"""<div style='position:relative;height:80px;background:{C['card']};
                                   border-radius:8px;overflow:hidden;border:1px solid {C['border2']};'>"""
    for sess in (arab_sess + intl_sess):
        o, c = sess["open_h"], sess["close_h"]
        if o < c:
            left = o/24*100; width=(c-o)/24*100
            timeline_html += f"""<div style='position:absolute;top:8px;left:{left:.1f}%;
                width:{width:.1f}%;height:24px;background:{sess['color']}55;
                border:1px solid {sess['color']}88;border-radius:3px;
                display:flex;align-items:center;justify-content:center;
                font-size:9px;color:{sess['color']};font-weight:700;overflow:hidden;'>
              {sess['flag']}
            </div>"""
        else:
            w1=(24-o)/24*100; w2=c/24*100
            timeline_html += f"""
            <div style='position:absolute;top:8px;left:{o/24*100:.1f}%;width:{w1:.1f}%;height:24px;
                background:{sess['color']}55;border:1px solid {sess['color']}88;border-radius:3px;
                display:flex;align-items:center;justify-content:center;
                font-size:9px;color:{sess['color']};font-weight:700;'>{sess['flag']}</div>
            <div style='position:absolute;top:8px;left:0%;width:{w2:.1f}%;height:24px;
                background:{sess['color']}55;border:1px solid {sess['color']}88;border-radius:3px;'></div>"""

    # Current time needle
    needle_pct = now_utc.hour/24*100 + now_utc.minute/1440*100
    timeline_html += f"""
      <div style='position:absolute;top:0;left:{needle_pct:.2f}%;width:2px;height:100%;
                  background:{C['gold_hi']};opacity:0.9;'></div>
      <div style='position:absolute;top:36px;left:{needle_pct:.2f}%;transform:translateX(-50%);
                  font-size:9px;color:{C['gold_hi']};font-weight:700;white-space:nowrap;'>
        {now_utc.strftime('%H:%M')}
      </div>"""

    # Hour labels
    for h in [0,4,8,12,16,20]:
        timeline_html += f"""<div style='position:absolute;bottom:4px;left:{h/24*100:.1f}%;
            font-size:8px;color:{C['dim']};transform:translateX(-50%);'>{h:02d}:00</div>"""

    timeline_html += "</div>"
    st.markdown(timeline_html, unsafe_allow_html=True)

    # Best trading times note
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='background:{C['card']};border:1px solid {C['border']};border-radius:6px;padding:14px 18px;'>
      <div style='font-size:12px;color:{C['muted']};{"direction:rtl;text-align:right;" if use_ar else ""}'>
        💡 {'أفضل أوقات تداول الذهب: تداخل لندن ونيويورك (13:00–17:00 UTC) يشهد أعلى حجم تداول وأكبر حركة أسعار · تداخل طوكيو وأوروبا (07:00–09:00 UTC) مناسب للأسواق العربية · عادةً أقل تقلبًا خلال جلسة آسيا.' if use_ar else
         'Best gold trading times: London/New York overlap (13:00–17:00 UTC) sees highest volume and biggest moves · Tokyo/Europe overlap (07:00–09:00 UTC) is convenient for Arab timezone traders · Generally quieter during Asian session.'}
      </div>
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: PRICE ALERTS 🔔
# ═══════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: PRICE ALERTS 🔔
# ═══════════════════════════════════════════════════════════════════════════════
elif nav == L["nav_alerts"]:
    use_ar = is_rtl()
    ph("🔔  " + L.get("nav_alerts","Price Alerts").replace("🔔  ",""),
       "ضع تنبيهاً عند سعر محدد — يُعرض عند فتح التطبيق" if use_ar else
       "Set alerts at target prices — triggered on next app load")

    if "price_alerts" not in st.session_state: st.session_state["price_alerts"] = []

    # Add alert form
    with st.expander("➕ " + (L['add_new_alert']), expanded=True):
        ac1, ac2, ac3 = st.columns(3)
        with ac1:
            al_price = st.number_input("💲 " + (L['target_price']),
                                        min_value=100.0, max_value=99999.0, value=float(round(g_ref,-1)+50), step=10.0, key="al_price")
            al_dir   = st.radio("📍 " + (L['direction_lbl']),
                                 (["عند الوصول أو الارتفاع","عند الانخفاض إلى"] if use_ar else ["At or Above","At or Below"]),
                                 horizontal=True, key="al_dir")
        with ac2:
            al_label = st.text_input("🏷️ " + (L['alert_label']),
                                      placeholder=("مثال: هدف الربح الأول" if use_ar else "e.g. TP1 target"), key="al_label")
            al_note  = st.text_area("📝 " + ("ملاحظة" if use_ar else "Note"),
                                     placeholder=(L['why_level']),
                                     height=80, key="al_note")
        with ac3:
            st.markdown("<br><br>", unsafe_allow_html=True)
            if st.button("✅ " + (L['save_alert']),
                         type="primary", use_container_width=True, key="al_save"):
                direction = "above" if (al_dir in ["عند الوصول أو الارتفاع","At or Above"]) else "below"
                st.session_state["price_alerts"].append({
                    "id": len(st.session_state["price_alerts"]),
                    "price": al_price, "direction": direction,
                    "label": al_label or f"Alert ${al_price:,.0f}",
                    "note": al_note, "triggered": False,
                    "created": datetime.datetime.now().strftime("%d %b %Y %H:%M")
                })
                _sb_save()
                st.success("🔔 " + ("تم حفظ التنبيه!" if use_ar else "Alert saved!"))
                st.rerun()

    # Check active alerts vs current price
    triggered_now = []
    for alert in st.session_state["price_alerts"]:
        if not alert["triggered"]:
            if alert["direction"] == "above" and g_ref >= alert["price"]:
                triggered_now.append(alert)
                alert["triggered"] = True
            elif alert["direction"] == "below" and g_ref <= alert["price"]:
                triggered_now.append(alert)
                alert["triggered"] = True

    # Show triggered alerts prominently
    for triggered_alert in triggered_now:
        st.markdown(f"""
        <div style='background:{C["green"]}22;border:2px solid {C["green"]};border-radius:8px;
                    padding:16px 20px;margin-bottom:8px;animation:pulse 1s;'>
          <div style='font-size:24px;font-weight:900;color:{C["green"]};'>
            🔔 {L['alert_triggered_lbl']}
          </div>
          <div style='font-size:16px;color:{C["text"]};margin-top:6px;'>
            {triggered_alert["label"]} — ${triggered_alert["price"]:,.2f} |
            {"السعر الحالي" if use_ar else "Current"}: ${g_ref:,.2f}
          </div>
          {f"<div style='font-size:12px;color:{C["muted"]};margin-top:4px;'>{triggered_alert["note"]}</div>" if triggered_alert["note"] else ""}
        </div>""", unsafe_allow_html=True)

    # Alert list
    alerts = st.session_state["price_alerts"]
    active   = [a for a in alerts if not a["triggered"]]
    done     = [a for a in alerts if a["triggered"]]

    if not alerts:
        st.markdown(f"""
        <div style='background:{C["card2"]};border:2px dashed {C["gold"]}44;border-radius:8px;
                    padding:40px;text-align:center;'>
          <div style='font-size:48px;'>🔔</div>
          <div style='font-size:14px;color:{C["muted"]};margin-top:12px;'>
            {"لا توجد تنبيهات نشطة — أضف تنبيهاً بسعر مستهدف أعلاه" if use_ar else
             "No alerts set — add a target price above to get notified"}
          </div>
        </div>""", unsafe_allow_html=True)
    else:
        # Active
        if active:
            st.markdown(f"<div class='section-label'>{'🟡 تنبيهات نشطة · ' + str(len(active)) if use_ar else '🟡 Active Alerts · ' + str(len(active))}</div>", unsafe_allow_html=True)
            for a in active:
                dist    = a["price"] - g_ref
                dist_pct= dist / g_ref * 100
                d_col   = C["green"] if (a["direction"]=="above" and dist>0) or (a["direction"]=="below" and dist<0) else C["red"]
                arr     = "▲" if a["direction"]=="above" else "▼"
                dc1, dc2 = st.columns([6,1])
                with dc1:
                    st.markdown(f"""
                    <div class='stat-card' style='margin-bottom:6px;border-color:{C["gold"]}44;'>
                      <div style='display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;'>
                        <div>
                          <div style='font-size:15px;font-weight:700;color:{C["gold_pale"]};'>{arr} {a["label"]}</div>
                          <div style='font-size:11px;color:{C["dim"]};margin-top:3px;'>
                            {L['alert_at']} ${a["price"]:,.2f} ·
                            {L['distance_lbl']}: <span style='color:{d_col};'>${abs(dist):,.2f} ({dist_pct:+.2f}%)</span>
                            · {a["created"]}
                          </div>
                          {f"<div style='font-size:11px;color:{C["muted"]};font-style:italic;margin-top:3px;'>{a["note"]}</div>" if a["note"] else ""}
                        </div>
                        <div style='font-family:{C["font_m"]},monospace;font-size:18px;font-weight:700;color:{C["gold_hi"]};'>
                          ${a["price"]:,.2f}
                        </div>
                      </div>
                    </div>""", unsafe_allow_html=True)
                with dc2:
                    if st.button("🗑️", key=f"del_al_{a['id']}"):
                        st.session_state["price_alerts"] = [x for x in alerts if x["id"] != a["id"]]
                        st.rerun()

        # Triggered
        if done:
            with st.expander(f"✅ {'التنبيهات المُفعَّلة · ' + str(len(done)) if use_ar else 'Triggered Alerts · ' + str(len(done))}"):
                for a in done:
                    st.markdown(f"""
                    <div style='padding:8px 12px;border-bottom:1px solid {C["border"]}44;opacity:0.6;'>
                      <span style='font-size:12px;color:{C["green"]};'>✅ {a["label"]}</span>
                      <span style='font-size:11px;color:{C["dim"]};margin-left:12px;'>${a["price"]:,.2f} · {a["created"]}</span>
                    </div>""", unsafe_allow_html=True)
                if st.button("🗑️ " + (L['clear_triggered']), key="clear_triggered"):
                    st.session_state["price_alerts"] = active
                    _sb_save()
                    st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: CORRELATION HEATMAP 📊
# ═══════════════════════════════════════════════════════════════════════════════
elif nav == L["nav_heatmap"]:
    use_ar = is_rtl()
    ph("📊  " + L.get("nav_heatmap","Correlation Heatmap").replace("📊  ",""),
       "ارتباط الذهب مع الأصول الأخرى · كلما اقترب من 1 زاد الترابط · كلما اقترب من -1 زاد التعاكس" if use_ar else
       "How gold moves relative to other assets · +1=move together · -1=move opposite")

    hm_period = st.select_slider(
        ("الفترة الزمنية" if use_ar else "Time Period"),
        options=["1mo","3mo","6mo","1y","2y"], value="1y", key="hm_period"
    )

    ASSETS = {
        "Gold (XAU)": "GC=F", "Silver (XAG)":"SI=F", "Crude Oil":"CL=F",
        "S&P 500":"^GSPC", "USD Index":"DX-Y.NYB", "VIX":"^VIX",
        "US 10Y":"^TNX", "Bitcoin":"BTC-USD", "Platinum":"PL=F",
    }

    @st.cache_data(ttl=CACHE_TTL_HISTORY * 12)
    def get_corr_data(period):
        """Compute the correlation matrix across major assets for the heatmap.
        
            Parameters
            ----------
            period : str
                yfinance period string.
        
            Returns
            -------
            pd.DataFrame
                Square correlation matrix with asset names as index/columns.
            """
        dfs = {}
        for name, ticker in ASSETS.items():
            try:
                df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
                if df.empty: continue
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = ['_'.join(str(c) for c in col).strip('_') for col in df.columns]
                cc = next((c for c in df.columns if c.lower().startswith("close")), None)
                if cc:
                    dfs[name] = df[cc].dropna().pct_change().dropna()
            except Exception:
                    _log.debug("Suppressed", exc_info=True)
        if len(dfs) < 2: return pd.DataFrame()
        combined = pd.DataFrame(dfs).dropna()
        return combined.corr().round(3)

    with st.spinner("⏳ " + ("جاري حساب الارتباطات..." if use_ar else "Computing correlations...")):
        corr = get_corr_data(hm_period)

    if corr.empty:
        st.error("لم يتم تحميل البيانات" if use_ar else "Could not load data")
    else:
        # Build heatmap with plotly
        import plotly.graph_objects as _pgo
        fig_hm = _pgo.Figure(data=_pgo.Heatmap(
            z=corr.values,
            x=list(corr.columns),
            y=list(corr.index),
            colorscale=[[0.0,"#E74C3C"],[0.3,"#FF8C42"],[0.5,C["card2"]],
                        [0.7,"#52B788"],[1.0,"#00C9A7"]],
            zmid=0, zmin=-1, zmax=1,
            text=[[f"{v:.2f}" for v in row] for row in corr.values],
            texttemplate="%{text}",
            textfont={"size":11,"color":C["text"]},
            hovertemplate="%{y} ↔ %{x}<br>r = %{z:.3f}<extra></extra>",
            showscale=True,
            colorbar=dict(
                tickvals=[-1,-0.5,0,0.5,1],
                ticktext=["-1.0 Inverse","-0.5","0 Neutral","+0.5","+1.0 Positive"],
                tickfont=dict(color=C["muted"],size=9),
            )
        ))
        fig_hm.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=C["card"],
            font=dict(color=C["text"],size=10),
            height=520, margin=dict(l=0,r=0,t=30,b=0),
            title=dict(text=(f"خريطة الارتباط · فترة {hm_period}" if use_ar else f"Correlation Matrix · {hm_period}"),
                       font=dict(color=C["gold"],size=13))
        )
        st.plotly_chart(fig_hm, use_container_width=True)

        # Gold row breakdown
        if "Gold (XAU)" in corr:
            gold_row = corr["Gold (XAU)"].drop("Gold (XAU)").sort_values()
            st.markdown(f"<div class='section-label'>{L['gold_corr_assets']}</div>", unsafe_allow_html=True)
            cr_cols = st.columns(len(gold_row))
            for col, (asset, val) in zip(cr_cols, gold_row.items()):
                bar_col = C["green"] if val > 0.3 else (C["red"] if val < -0.3 else C["muted"])
                label   = ("ترابط قوي 🟢" if val>0.5 else "ترابط معتدل 🟡" if val>0.2 else
                           "تعاكس معتدل 🟡" if val>-0.2 else "تعاكس قوي 🔴") if use_ar else                           ("Strong +ve 🟢" if val>0.5 else "Mild +ve 🟡" if val>0.2 else
                           "Mild -ve 🟡" if val>-0.2 else "Strong -ve 🔴")
                with col:
                    st.markdown(f"""
                    <div class='stat-card' style='text-align:center;padding:10px;'>
                      <div class='stat-label' style='font-size:8px;'>{asset.split(" ")[0]}</div>
                      <div style='font-family:{C["font_m"]},monospace;font-size:20px;
                                  font-weight:700;color:{bar_col};'>{val:+.2f}</div>
                      <div style='font-size:9px;color:{C["dim"]};margin-top:3px;'>{label}</div>
                    </div>""", unsafe_allow_html=True)

        # Explanation
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style='background:{C["card"]};border:1px solid {C["border"]};border-radius:6px;padding:14px 18px;'>
          <div style='font-size:11px;color:{C["muted"]};{"direction:rtl;text-align:right;" if use_ar else ""}'>
            {"💡 كيف تقرأ الخريطة: الأحمر = تعاكس (عندما يرتفع أحدهما ينخفض الآخر) · الأخضر = ترابط (يتحركان معاً) · الصفر = لا علاقة بينهما. الذهب عادةً معاكس للدولار ومعاكس لـ S&P 500 في أوقات الأزمات." if use_ar else
             "💡 How to read: Red = inverse movement · Green = move together · Near zero = no relationship. Gold typically has negative correlation with USD and inverse relationship with S&P 500 during crises."}
          </div>
        </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: MANSA SCORE 🏆
# ═══════════════════════════════════════════════════════════════════════════════
elif nav == L["nav_mansa_score"]:
    use_ar = is_rtl()
    ph("🏆  " + L.get("nav_mansa_score","MANSA Score").replace("🏆  ",""),
       "مؤشر مانسا الذهبي · درجة واحدة تلخص كل ما تحتاجه لقرار التداول" if use_ar else
       "The MANSA Gold Score · One number summarising the entire market")

    # ── Reuse sentiment + live data to build 5 pillars ───────────────────────
    @st.cache_data(ttl=CACHE_TTL_HISTORY)
    def get_mansa_components():
        """Compute technical components used by the MANSA Score calculator.
        
            Returns
            -------
            dict
                Keys: rsi, ma50, ma200, price, mom30, bb_pct, bb_u, bb_l.
            """
        df = fetch_history("1y","GC=F")
        if df.empty: return {}
        cl = find_col(df,["Close","Close_GC=F"])
        if not cl: return {}
        c = df[cl].dropna()
        if len(c)<15: return {}
        d=c.diff(); ag=d.clip(lower=0).rolling(14).mean(); al=(-d.clip(upper=0)).rolling(14).mean()
        rsi = float((100-(100/(1+ag/al.replace(0,1e-9)))).iloc[-1])
        ma50  = float(c.rolling(50).mean().iloc[-1]) if len(c)>=50 else float(c.mean())
        ma200 = float(c.rolling(200).mean().iloc[-1]) if len(c)>=200 else float(c.mean())
        mom30 = float((c.iloc[-1]/c.iloc[-30]-1)*100) if len(c)>=31 else 0
        bm=c.rolling(20).mean(); bs=c.rolling(20).std()
        bb_u = float((bm+2*bs).iloc[-1]); bb_l = float((bm-2*bs).iloc[-1])
        bb_pct = (float(c.iloc[-1])-bb_l)/(bb_u-bb_l) if (bb_u-bb_l)>0 else 0.5
        return dict(rsi=rsi,ma50=ma50,ma200=ma200,price=float(c.iloc[-1]),
                    mom30=mom30,bb_pct=bb_pct,bb_u=bb_u,bb_l=bb_l)

    mc = get_mansa_components()
    clamp = lambda v,lo=0,hi=100: max(lo,min(hi,v))

    # PILLAR 1: Technical (RSI, MA, Bollinger) → 0-100
    rsi_v   = mc.get("rsi",50)
    ma50_v  = mc.get("ma50",g_ref)
    ma200_v = mc.get("ma200",g_ref)
    bb_p    = mc.get("bb_pct",0.5)
    rsi_p   = clamp((rsi_v-30)/40*100)  # standard 30/70 levels
    ma50_p  = clamp(50+(g_ref-ma50_v)/ma50_v*800)
    ma200_p = clamp(50+(g_ref-ma200_v)/ma200_v*400)
    bb_p2   = clamp((1-bb_p)*100)
    tech_score = (rsi_p*0.35 + ma50_p*0.30 + ma200_p*0.20 + bb_p2*0.15)

    # PILLAR 2: Macro (VIX, DXY, US10Y) → 0-100
    vix_v   = live["vix"]["price"]
    dxy_ch  = live["dxy"]["pct"]
    us10y_v = live["us10y"]["price"]
    vix_p   = clamp((vix_v-10)/30*100)
    dxy_p   = clamp(50-dxy_ch*20)
    rate_p  = clamp(50-(us10y_v-3.5)*20)
    macro_score = (vix_p*0.35 + dxy_p*0.35 + rate_p*0.30)

    # PILLAR 3: Momentum (30d, daily change) → 0-100
    mom30_v = mc.get("mom30",0)
    daily_v = live["gold"]["pct"]
    mom30_p = clamp(50+mom30_v*4)
    daily_p = clamp(50+daily_v*10)
    mom_score = (mom30_p*0.70 + daily_p*0.30)

    # PILLAR 4: Commodity (Gold/Silver, Gold/Oil ratios) → 0-100
    gsr     = g_ref/live["silver"]["price"] if live["silver"]["price"]>0 else 80
    gor     = g_ref/live["oil"]["price"]    if live["oil"]["price"]>0    else 25
    gsr_p   = clamp(50-(gsr-80)*2)   # GSR>80 = gold expensive vs silver = slight bearish
    gor_p   = clamp(50-(gor-25)*3)   # GOR>25 = gold expensive vs oil
    comm_score = (gsr_p*0.5+gor_p*0.5)

    # PILLAR 5: Market Structure (52W position) → 0-100
    @st.cache_data(ttl=CACHE_TTL_HISTORY * 12)
    def get_52w_range():
        """Return the 52-week (low, high) price range for gold.
        
            Returns
            -------
            tuple[float, float]
                *(low, high)* in USD per troy ounce.
            """
        df2 = fetch_history("1y","GC=F")
        if df2.empty: return g_ref*0.9, g_ref*1.1
        cl2 = find_col(df2,["Close","Close_GC=F"])
        if not cl2: return g_ref*0.9, g_ref*1.1
        c2 = df2[cl2].dropna()
        return float(c2.min()), float(c2.max())
    lo52, hi52 = get_52w_range()
    struct_score = clamp((g_ref-lo52)/(hi52-lo52)*100 if hi52>lo52 else 50)

    # COMPOSITE MANSA SCORE
    w = [0.28, 0.25, 0.22, 0.13, 0.12]
    p = [tech_score, macro_score, mom_score, comm_score, struct_score]
    mansa_score = clamp(sum(wi*pi for wi,pi in zip(w,p)))

    # Grade
    if mansa_score >= 80:
        grade="A+"; grade_ar="ممتاز"; grade_col=C["green"]; grade_desc_en="Strong Buy — All signals aligned bullishly"; grade_desc_ar="شراء قوي — جميع المؤشرات صاعدة"
    elif mansa_score >= 65:
        grade="A";  grade_ar="جيد جداً"; grade_col="#52B788"; grade_desc_en="Buy — Most signals positive"; grade_desc_ar="شراء — معظم المؤشرات إيجابية"
    elif mansa_score >= 55:
        grade="B";  grade_ar="جيد"; grade_col=C["gold"]; grade_desc_en="Cautious Buy — Mixed signals, trend up"; grade_desc_ar="شراء بحذر — إشارات مختلطة"
    elif mansa_score >= 45:
        grade="C";  grade_ar="محايد"; grade_col=C["muted"]; grade_desc_en="Neutral — Wait for clearer signal"; grade_desc_ar="محايد — انتظر إشارة أوضح"
    elif mansa_score >= 35:
        grade="D";  grade_ar="ضعيف"; grade_col="#FF8C42"; grade_desc_en="Caution — Bearish pressure building"; grade_desc_ar="تحذير — ضغط هبوطي يتراكم"
    else:
        grade="F";  grade_ar="سلبي"; grade_col=C["red"]; grade_desc_en="Avoid / Sell — Strong bearish signals"; grade_desc_ar="تجنب/بيع — إشارات هبوطية قوية"

    # ── Big score display ─────────────────────────────────────────────────────
    ms1, ms2 = st.columns([1,2], gap="large")
    with ms1:
        st.components.v1.html(f"""
        <div style='text-align:center;padding:20px 10px;'>
          <div style='font-family:Georgia,serif;font-size:11px;letter-spacing:.3em;
                      color:#C9960C;text-transform:uppercase;margin-bottom:8px;'>
            {"مؤشر مانسا الذهبي" if use_ar else "MANSA GOLD SCORE"}
          </div>
          <div style='font-family:Georgia,serif;font-size:90px;font-weight:900;
                      color:{grade_col};line-height:1;'>{mansa_score:.0f}</div>
          <div style='font-family:Georgia,serif;font-size:32px;font-weight:900;
                      color:{grade_col};margin-top:4px;'>{grade}</div>
          <div style='font-family:Cairo,serif;font-size:16px;color:{grade_col};margin-top:8px;'>
            {grade_ar if use_ar else grade_desc_en.split("—")[0].strip()}
          </div>
          <div style='margin-top:16px;font-size:11px;color:#666;font-style:italic;'>
            ${g_ref:,.2f} · {datetime.datetime.now().strftime("%d %b %Y %H:%M")}
          </div>
        </div>""", height=280, scrolling=False)

    with ms2:
        st.markdown(f"""
        <div style='background:{C["card2"]};border:2px solid {grade_col}55;border-radius:10px;
                    padding:20px 24px;'>
          <div style='font-family:"Cairo",serif;font-size:{"16px" if use_ar else "15px"};
                      color:{C["text"]};line-height:1.8;{"direction:rtl;text-align:right;" if use_ar else ""}'>
            <b style='color:{grade_col};'>{grade_ar if use_ar else grade_desc_en.split("—")[0].strip()}</b>
            {'<br>' + grade_desc_ar if use_ar else '<br>' + grade_desc_en}
          </div>
          <hr style='border-color:{C["border2"]};margin:14px 0;'>
          <div style='font-size:12px;color:{C["muted"]};{"direction:rtl;text-align:right;" if use_ar else ""}'>
            {"المؤشر مبني على 5 ركائز: التحليل الفني (28%) · المؤشرات الكلية (25%) · الزخم (22%) · نسب السلع (13%) · الهيكل السوقي (12%)" if use_ar else
             "Score built on 5 pillars: Technical Analysis (28%) · Macro Indicators (25%) · Momentum (22%) · Commodity Ratios (13%) · Market Structure (12%)"}
          </div>
        </div>""", unsafe_allow_html=True)

    # 5 Pillars breakdown
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"<div class='section-label'>{L['five_pillars']}</div>", unsafe_allow_html=True)
    pillars = [
        ("⚙️", "التحليل الفني" if use_ar else "Technical",   tech_score,   "RSI · MA50 · MA200 · Bollinger"),
        ("🌍", "المؤشرات الكلية" if use_ar else "Macro",      macro_score,  "VIX · DXY · US10Y Yield"),
        ("📈", "الزخم" if use_ar else "Momentum",            mom_score,    "30d · Daily change"),
        ("⚖️", "نسب السلع" if use_ar else "Commodities",    comm_score,   "Gold/Silver · Gold/Oil"),
        ("🏗️", "الهيكل" if use_ar else "Structure",         struct_score, "52W Range Position"),
    ]
    wts = [0.28, 0.25, 0.22, 0.13, 0.12]
    pc = st.columns(5)
    for col, (icon, name, score, detail), wt in zip(pc, pillars, wts):
        sc = C["green"] if score>65 else (C["red"] if score<35 else C["gold"])
        with col:
            st.markdown(f"""
            <div class='stat-card' style='text-align:center;padding:12px;'>
              <div style='font-size:22px;'>{icon}</div>
              <div style='font-size:10px;color:{C["muted"]};margin:4px 0;'>{name}</div>
              <div style='font-family:{C["font_m"]},monospace;font-size:24px;
                          font-weight:900;color:{sc};'>{score:.0f}</div>
              <div style='width:100%;height:5px;background:{C["border"]};border-radius:3px;margin:6px 0;'>
                <div style='width:{score:.0f}%;height:100%;background:{sc};border-radius:3px;'></div>
              </div>
              <div style='font-size:9px;color:{C["dim"]};'>{detail}</div>
              <div style='font-size:9px;color:{C["muted"]};margin-top:2px;'>{"وزن" if use_ar else "Weight"}: {wt*100:.0f}%</div>
            </div>""", unsafe_allow_html=True)

    # Share/copy score
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='background:{C["card"]};border:1px solid {grade_col}44;border-radius:8px;
                padding:14px 20px;text-align:center;'>
      <div style='font-size:13px;color:{C["text"]};font-weight:500;'>
        {"📤 شارك هذا التحليل:" if use_ar else "📤 Share this analysis:"}
      </div>
      <div style='font-family:{C["font_m"]},monospace;font-size:12px;color:{grade_col};margin-top:8px;'>
        🏆 MANSA Gold Score: {mansa_score:.0f}/100 ({grade}) | Gold: ${g_ref:,.2f} |
        {datetime.datetime.now().strftime("%d %b %Y")} | mansa.app
      </div>
    </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: ZAKAT CALCULATOR ☪️
# ═══════════════════════════════════════════════════════════════════════════════
elif nav == L["nav_zakat"]:
    use_ar = is_rtl()
    ph("☪️  " + L.get("nav_zakat","Zakat Calculator").replace("☪️  ",""),
       "حاسبة زكاة الذهب · تلقائياً حسب سعر السوق اليوم" if use_ar else
       "Gold Zakat Calculator · Automatically based on today's market price")

    rtl = "direction:rtl;text-align:right;" if use_ar else ""

    # ── Quran verse (At-Tawbah 34-35) ────────────────────────────────────────
    st.markdown(f"""
    <div style='background:linear-gradient(135deg,{C["card2"]},{C["card"]});
                border:2px solid {C["gold"]}77;border-radius:10px;
                padding:22px 28px;margin-bottom:20px;position:relative;overflow:hidden;'>
      <div style='position:absolute;top:0;left:0;right:0;height:2px;
                  background:linear-gradient(90deg,transparent,{C["gold_hi"]},{C["gold"]},transparent);'></div>
      <!-- Quran verse -->
      <div style='font-family:"Cairo",serif;font-size:17px;font-weight:600;
                  color:{C["gold_pale"]};line-height:2.1;direction:rtl;text-align:right;
                  margin-bottom:14px;'>
        ﴿ وَالَّذِينَ يَكْنِزُونَ الذَّهَبَ وَالْفِضَّةَ وَلَا يُنفِقُونَهَا فِي سَبِيلِ اللَّهِ
        فَبَشِّرْهُم بِعَذَابٍ أَلِيمٍ ۝ يَوْمَ يُحْمَىٰ عَلَيْهَا فِي نَارِ جَهَنَّمَ
        فَتُكْوَىٰ بِهَا جِبَاهُهُمْ وَجُنُوبُهُمْ وَظُهُورُهُمْ ۖ هَٰذَا مَا كَنَزْتُمْ
        لِأَنفُسِكُمْ فَذُوقُوا مَا كُنتُمْ تَكْنِزُونَ ﴾
      </div>
      <div style='font-size:12px;color:{C["gold"]};direction:rtl;text-align:right;
                  margin-bottom:14px;font-weight:700;'>
        ― سورة التوبة: الآيتان 34-35
      </div>
      <!-- English translation -->
      <div style='font-family:Georgia,serif;font-size:13px;color:{C["text"]};
                  line-height:1.8;font-style:italic;margin-bottom:10px;
                  border-top:1px solid {C["gold"]}33;padding-top:12px;'>
        "And those who hoard gold and silver and do not spend it in the cause of Allah —
        give them tidings of a painful punishment. The Day it will be heated in the fire
        of Hell and their foreheads, their sides, and their backs will be branded with it,
        [and it will be said], 'This is what you hoarded for yourselves,
        so taste what you used to hoard.'"
      </div>
      <div style='font-size:11px;color:{C["gold"]};font-weight:600;'>
        — Surah At-Tawbah [9:34-35]
      </div>
    </div>""", unsafe_allow_html=True)

    # ── Zakat info box ─────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style='background:{C["card2"]};border:1px solid {C["gold"]}44;border-radius:8px;
                padding:16px 20px;margin-bottom:16px;'>
      <div style='font-size:13px;color:{C["text"]};line-height:1.85;{rtl}'>
        {"📖 <b>الزكاة واجبة</b> على الذهب إذا بلغ النصاب وحال عليه الحول (سنة هجرية كاملة = 354 يومًا). وقد حدّد النبي ﷺ النصاب بـ <b>87.48 جرامًا</b> من الذهب الخالص (20 مثقالًا)، والمقدار الواجب <b>2.5%</b> من إجمالي القيمة السوقية. يُستحب أداؤها في رمضان لتضاعف الأجر." if use_ar else
         "📖 <b>Zakat is obligatory</b> on gold that reaches the nisab threshold and has been held for one complete lunar year (hawl = 354 days). The Prophet ﷺ set the nisab at <b>87.48 grams</b> of pure gold (20 mithqals). The amount due is <b>2.5%</b> (1/40th) of total market value. Many Muslims choose to pay during Ramadan for increased reward."}
      </div>
    </div>""", unsafe_allow_html=True)

    # Nisab standard selector
    nisab_std = st.radio(
        ("معيار النصاب" if use_ar else "Nisab Standard"),
        (["معيار الذهب (87.48 جرام) — المستخدم للذهب",
          "معيار الفضة (612.36 جرام) — الأشمل والأكثر توصية"]
         if use_ar else
         ["Gold standard (87.48g) — use when assets are mainly gold",
          "Silver standard (612.36g) — recommended by most scholars (lower threshold)"]),
        key="nisab_std", horizontal=True
    )
    use_silver_nisab = "فضة" in nisab_std or "Silver" in nisab_std

    # Nisab calculation — corrected to 87.48g (Prophet's ﷺ established weight)
    nisab_grams  = 87.48 if not use_silver_nisab else (612.36 * (g_ref/31.1035) / (live["silver"]["price"]/31.1035 if live["silver"]["price"]>0 else 1))
    nisab_usd    = (87.48 * g_ref / 31.1035) if not use_silver_nisab else (612.36 * (live["silver"]["price"]/31.1035 if live["silver"]["price"]>0 else 0.95))
    pm_data      = MARKETS[st.session_state["primary_mkt"]]
    pm_fx        = fetch_fx(pm_data["fx_ticker"], pm_data.get("fx_inverse",False), pm_data["fx_approx"])
    pm_fx_rate   = 1.0/pm_fx if pm_data.get("fx_inverse",False) else pm_fx
    nisab_local  = nisab_usd * pm_fx_rate
    price_per_g_24k = g_ref / 31.1035

    zc1, zc2 = st.columns(2, gap="large")
    with zc1:
        st.markdown(f"<div class='section-label'>{L['enter_holdings']}</div>", unsafe_allow_html=True)

        z_24k = st.number_input(f"{L['gold24k']}", min_value=0.0, value=0.0, step=1.0, key="z_24k")
        z_22k = st.number_input(f"{L['gold22k']}", min_value=0.0, value=0.0, step=1.0, key="z_22k")
        z_21k = st.number_input(f"{L['gold21k']}", min_value=0.0, value=100.0, step=1.0, key="z_21k")
        z_18k = st.number_input(f"{L['gold18k']}", min_value=0.0, value=0.0, step=1.0, key="z_18k")
        z_14k = st.number_input(f"{L['gold14k']}", min_value=0.0, value=0.0, step=1.0, key="z_14k")
        z_curr= st.selectbox("💱 " + ("عملة الزكاة" if use_ar else "Zakat Currency"),
                              ["USD","JOD","SAR","AED","EGP","KWD","QAR","BHD"], key="z_curr")

    with zc2:
        # Convert all to 24K equivalent grams
        equiv_24k = z_24k + z_22k*(22/24) + z_21k*(21/24) + z_18k*(18/24) + z_14k*(14/24)
        total_value_usd = equiv_24k * price_per_g_24k

        # Local currency value
        z_fx = {"USD":1.0,"JOD":1/0.709,"SAR":1/3.75,"AED":1/3.6725,"EGP":1/50.9,
                "KWD":1/0.307,"QAR":1/3.64,"BHD":1/0.377}
        z_rate       = 1.0 / z_fx.get(z_curr, 1.0)  # USD per unit of z_curr
        total_local  = total_value_usd / (z_rate if z_rate > 0 else 1)
        nisab_local_z = nisab_usd / (z_rate if z_rate > 0 else 1)

        zakat_due_usd   = total_value_usd * 0.025
        zakat_local     = total_local * 0.025
        above_nisab     = equiv_24k >= nisab_grams

        st.markdown(f"<div class='section-label'>{L['zakat_result']}</div>", unsafe_allow_html=True)

        nisab_col = C["green"] if above_nisab else C["red"]
        status_msg_ar = "✅ بلغت النصاب — الزكاة واجبة" if above_nisab else "❌ لم تبلغ النصاب — لا زكاة واجبة"
        status_msg_en = "✅ Nisab reached — Zakat is due" if above_nisab else "❌ Below nisab — No zakat due yet"

        st.markdown(f"""
        <div style='background:{nisab_col}18;border:2px solid {nisab_col}66;border-radius:10px;padding:20px;'>
          <div style='font-family:"Cairo",serif;font-size:{"17px" if use_ar else "15px"};font-weight:700;
                      color:{nisab_col};margin-bottom:14px;{rtl}'>
            {status_msg_ar if use_ar else status_msg_en}
          </div>
          <div style='display:grid;grid-template-columns:1fr 1fr;gap:10px;'>
            <div class='stat-card' style='padding:10px;text-align:center;'>
              <div class='stat-label'>{L['total_gold_lbl']}</div>
              <div style='font-size:18px;color:{C["gold_hi"]};font-weight:700;'>{equiv_24k:.2f}g</div>
              <div style='font-size:10px;color:{C["dim"]};'>24K {"معادل" if use_ar else "equivalent"}</div>
            </div>
            <div class='stat-card' style='padding:10px;text-align:center;'>
              <div class='stat-label'>{L['nisab_lbl']}</div>
              <div style='font-size:18px;color:{nisab_col};font-weight:700;'>{nisab_grams:.2f}g</div>
              <div style='font-size:10px;color:{C["dim"]};'>${nisab_usd:,.2f} USD</div>
            </div>
            <div class='stat-card' style='padding:10px;text-align:center;'>
              <div class='stat-label'>{L['total_value_lbl']}</div>
              <div style='font-size:18px;color:{C["gold_pale"]};font-weight:700;'>${total_value_usd:,.2f}</div>
              <div style='font-size:10px;color:{C["dim"]};'>{total_local:,.2f} {z_curr}</div>
            </div>
            <div class='stat-card' style='padding:10px;text-align:center;border-color:{nisab_col}55;'>
              <div class='stat-label'>{L['zakat_due']}</div>
              <div style='font-size:22px;color:{nisab_col};font-weight:900;'>
                {zakat_local:,.2f} {z_curr if above_nisab else "—"}
              </div>
              <div style='font-size:10px;color:{C["dim"]};'>
                {"=" if above_nisab else ""} ${zakat_due_usd:,.2f} USD
              </div>
            </div>
          </div>
          {"<div style='margin-top:12px;padding:10px;background:"+C["gold"]+"18;border-radius:4px;font-size:12px;color:"+C["muted"]+";"+rtl+"'>" + ("💡 هذا الحساب تقديري. تأكد من السعر الفعلي في يوم أداء الزكاة وراجع عالمًا شرعيًا للتأكد." if use_ar else "💡 This is an estimate. Verify the actual price on the day of payment and consult an Islamic scholar.") + "</div>" if above_nisab else ""}
        </div>""", unsafe_allow_html=True)

        # Progress bar to nisab
        if not above_nisab:
            pct = min(equiv_24k/nisab_grams*100, 100)
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"""
            <div style='font-size:12px;color:{C["muted"]};margin-bottom:6px;{rtl}'>
              {"الوصول للنصاب:" if use_ar else "Progress to Nisab:"}
              {equiv_24k:.1f} / {nisab_grams:.0f}g ({pct:.1f}%)
            </div>
            <div style='height:8px;background:{C["border"]};border-radius:4px;'>
              <div style='width:{pct:.1f}%;height:100%;background:linear-gradient(90deg,{C["gold_dark"]},{C["gold_hi"]});border-radius:4px;'></div>
            </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: ASSET COMPARISON 📈
# ═══════════════════════════════════════════════════════════════════════════════
elif nav == L["nav_journal"]:
    use_ar = is_rtl()
    ph("📓  " + L.get("nav_journal","Trade Journal").replace("📓  ",""),
       "سجّل صفقاتك وتتبع أداء توصيات الذكاء الاصطناعي" if use_ar else
       "Log your trades and track AI recommendation performance")

    if "trade_journal" not in st.session_state: st.session_state["trade_journal"] = []

    with st.expander("➕ " + ("تسجيل صفقة جديدة" if use_ar else "Log New Trade"),
                     expanded=len(st.session_state["trade_journal"])==0):
        tj1, tj2, tj3 = st.columns(3)
        with tj1:
            tj_dir    = st.radio("📍",["BUY 🟢","SELL 🔴"], horizontal=True, key="tj_dir")
            tj_entry  = st.number_input("📥 " + ("دخول" if use_ar else "Entry"), value=float(g_ref), step=1.0, key="tj_entry")
            tj_exit   = st.number_input("📤 " + ("خروج" if use_ar else "Exit"), value=float(g_ref*1.01), step=1.0, key="tj_exit")
        with tj2:
            tj_qty    = st.number_input("⚖️ " + ("الكمية (جرام)" if use_ar else "Qty (grams)"), value=10.0, step=1.0, key="tj_qty")
            tj_date   = st.date_input("📅", value=datetime.date.today(), key="tj_date")
            tj_ai_rec = st.text_input("🤖 " + ("توصية الذكاء الاصطناعي" if use_ar else "AI Recommendation"),
                                       placeholder=("مثال: BUY — RSI=32 oversold" if use_ar else "e.g. BUY — RSI=32 oversold"), key="tj_ai")
        with tj3:
            tj_note   = st.text_area("📝 " + ("ملاحظاتك" if use_ar else "Your Notes"),
                                      height=100, key="tj_note",
                                      placeholder=("لماذا دخلت هذه الصفقة؟" if use_ar else "Why did you enter this trade?"))
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("💾 " + ("حفظ" if use_ar else "Save"), type="primary", use_container_width=True, key="tj_save"):
                qty_oz = tj_qty / 31.1035
                pnl    = (tj_exit - tj_entry) * qty_oz * (1 if "BUY" in tj_dir else -1)
                st.session_state["trade_journal"].append({
                    "id": len(st.session_state["trade_journal"]),
                    "date": str(tj_date), "direction": tj_dir,
                    "entry": tj_entry, "exit": tj_exit,
                    "qty_g": tj_qty, "qty_oz": qty_oz,
                    "pnl": pnl, "pnl_pct": pnl/(tj_entry*qty_oz)*100 if tj_entry*qty_oz>0 else 0,
                    "ai_rec": tj_ai_rec, "note": tj_note,
                    "status": "Win" if pnl>0 else ("Loss" if pnl<0 else "Break-even")
                })
                _sb_save()
                st.success("💾 " + ("تم الحفظ!" if use_ar else "Saved!"))
                st.rerun()

    trades = st.session_state["trade_journal"]
    if not trades:
        st.info("📓 " + ("سجّل أول صفقة لك أعلاه لتتبع أداءك" if use_ar else "Log your first trade above to start tracking performance"))
    else:
        # Stats
        wins   = [t for t in trades if t["status"]=="Win"]
        losses = [t for t in trades if t["status"]=="Loss"]
        total_pnl = sum(t["pnl"] for t in trades)
        win_rate  = len(wins)/len(trades)*100 if trades else 0
        avg_win   = sum(t["pnl"] for t in wins)/len(wins) if wins else 0
        avg_loss  = sum(t["pnl"] for t in losses)/len(losses) if losses else 0
        rr_ratio  = abs(avg_win/avg_loss) if avg_loss else 0

        s1,s2,s3,s4,s5 = st.columns(5)
        for col, lbl, val, vc in [
            (s1,"📊 " + ("الصفقات" if use_ar else "Trades"),      str(len(trades)),  C["gold_pale"]),
            (s2,"🏆 " + ("نسبة الربح" if use_ar else "Win Rate"),  f"{win_rate:.1f}%",C["green"] if win_rate>50 else C["red"]),
            (s3,"💰 " + ("إجمالي P&L" if use_ar else "Total P&L"), f"${total_pnl:+,.2f}", C["green"] if total_pnl>=0 else C["red"]),
            (s4,"📈 " + ("متوسط ربح" if use_ar else "Avg Win"),   f"${avg_win:,.2f}",  C["green"]),
            (s5,"⚖️ " + ("نسبة R/R" if use_ar else "R/R Ratio"), f"{rr_ratio:.2f}",   C["gold"]),
        ]:
            with col:
                st.markdown(f"""<div class='stat-card' style='text-align:center;'>
                  <div class='stat-label' style='font-size:9px;'>{lbl}</div>
                  <div style='font-family:{C["font_m"]},monospace;font-size:18px;
                              font-weight:700;color:{vc};'>{val}</div>
                </div>""", unsafe_allow_html=True)

        # P&L chart
        if len(trades) > 1:
            cum_pnl = []
            running = 0
            for trade in sorted(trades, key=lambda x: x["date"]):
                running += trade["pnl"]
                cum_pnl.append(running)
            fig_j = go.Figure()
            fig_j.add_trace(go.Scatter(
                y=cum_pnl, mode="lines+markers",
                line=dict(color=C["gold"], width=2),
                fill="tozeroy", fillcolor=rgba(C["gold_dark"],0.15),
                name="Cumulative P&L"
            ))
            fig_j.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=C["card"],
                height=220, margin=dict(l=0,r=0,t=20,b=0),
                font=dict(color=C["text"],size=9),
                xaxis=dict(gridcolor=C["border2"]),
                yaxis=dict(gridcolor=C["border2"],title="USD"),
                title=dict(text=("منحنى الأرباح التراكمية" if use_ar else ("منحنى الأرباح التراكمية" if use_ar else "Cumulative P&L Curve")),
                           font=dict(color=C["gold"],size=11))
            )
            st.plotly_chart(fig_j, use_container_width=True)

        # Trade list
        st.markdown(f"<div class='section-label'>{L['trade_log']}</div>", unsafe_allow_html=True)
        for je in reversed(trades[-20:]):
            pnl_c = C["green"] if je["pnl"]>0 else (C["red"] if je["pnl"]<0 else C["muted"])
            pnl_i = "▲" if je["pnl"]>0 else ("▼" if je["pnl"]<0 else "→")
            tc1, tc2 = st.columns([7,1])
            with tc1:
                st.markdown(f"""
                <div class='stat-card' style='margin-bottom:5px;'>
                  <div style='display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:6px;'>
                    <div>
                      <span style='font-size:13px;font-weight:700;color:{C["text"]};'>{je["date"]}</span>
                      <span style='margin:0 8px;background:{"rgba(82,183,136,0.13)" if "BUY" in je["direction"] else "rgba(231,76,60,0.13)"};
                                   color:{"#52B788" if "BUY" in je["direction"] else "#E74C3C"};
                                   padding:2px 10px;border-radius:10px;font-size:11px;font-weight:700;'>
                        {je["direction"]}
                      </span>
                      <span style='font-size:11px;color:{C["dim"]};'>
                        {je["qty_g"]:.1f}g · ${je["entry"]:,.2f} → ${je["exit"]:,.2f}
                        {" · AI: "+je["ai_rec"] if t.get("ai_rec") else ""}
                      </span>
                    </div>
                    <div style='font-family:{C["font_m"]},monospace;font-size:16px;
                                font-weight:700;color:{pnl_c};'>
                      {pnl_i} ${abs(je["pnl"]):,.2f} ({je["pnl_pct"]:+.2f}%)
                    </div>
                  </div>
                  {f"<div style='font-size:10px;color:{C["dim"]};margin-top:4px;font-style:italic;'>{je["note"]}</div>" if t.get("note") else ""}
                </div>""", unsafe_allow_html=True)
            with tc2:
                if st.button("🗑️", key=f"del_tj_{je['id']}"):
                    st.session_state["trade_journal"] = [x for x in trades if x["id"] != je["id"]]
                    st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: GOLD MAP 🌍
# ═══════════════════════════════════════════════════════════════════════════════
elif nav == L["nav_goldmap"]:
    use_ar = is_rtl()
    ph("🌍  " + L.get("nav_goldmap","Gold Map").replace("🌍  ",""),
       "سعر الذهب لكل جرام بالعملة المحلية في أسواق العالم" if use_ar else
       "Gold price per gram in local currency across world markets")

    # Build price table for all markets
    map_data = []
    for mk, cfg in MARKETS.items():
        try:
            p, _ = mkt_price(g_ref, cfg, "21K — 875")  # 21K as standard Arab purity
            map_data.append({
                "market": mk, "flag": cfg["flag"],
                "currency": cfg["currency"], "unit": cfg["unit_label"],
                "price_21k": p, "note": cfg["note"]
            })
        except Exception:
                    _log.debug("Suppressed", exc_info=True)

    # Sort by price in USD equivalent
    map_data.sort(key=lambda x: x["price_21k"])

    st.markdown(f"<div class='section-label'>{'سعر ذهب 21 قيراط لكل ' + map_data[0]['unit'] if map_data and use_ar else 'Gold 21K price per ' + (map_data[0]['unit'] if map_data else 'gram')}</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:11px;color:{C["dim"]};margin-bottom:12px;'>{L['nisab_based']} ${g_ref:,.2f}/oz</div>", unsafe_allow_html=True)

    # Visual cards grid
    cols_per_row = 4
    for row_start in range(0, len(map_data), cols_per_row):
        row_items = map_data[row_start:row_start+cols_per_row]
        row_cols  = st.columns(cols_per_row)
        for col, item in zip(row_cols, row_items):
            is_primary = item["market"] == st.session_state["primary_mkt"]
            bc = C["gold"] if is_primary else C["border2"]
            bw = "2px" if is_primary else "1px"
            with col:
                st.markdown(f"""
                <div style='background:{C["card2"]};border:{bw} solid {bc};border-radius:8px;
                            padding:12px;text-align:center;margin-bottom:8px;
                            {"box-shadow:0 0 10px "+C["gold"]+"33;" if is_primary else ""}'>
                  <div style='font-size:22px;'>{item["flag"]}</div>
                  <div style='font-size:10px;color:{C["muted"]};margin-top:4px;'>
                    {item["market"].split("(")[0].strip()}
                  </div>
                  <div style='font-family:{C["font_m"]},monospace;font-size:17px;
                              font-weight:700;color:{C["gold_hi"] if is_primary else C["gold_pale"]};
                              margin:4px 0;'>
                    {item["price_21k"]:,.3f}
                  </div>
                  <div style='font-size:10px;color:{C["dim"]};'>
                    {item["currency"]}/{item["unit"]}
                  </div>
                  {f"<div style='font-size:9px;color:{C['gold']};margin-top:3px;'>⭐ رئيسي</div>" if is_primary and use_ar else f"<div style='font-size:9px;color:{C['gold']};margin-top:3px;'>⭐ Primary</div>" if is_primary else ""}
                </div>""", unsafe_allow_html=True)

    # Bar chart comparison
    st.markdown("<br>", unsafe_allow_html=True)
    if map_data:
        fig_map = go.Figure(go.Bar(
            x=[d["flag"]+" "+d["market"].split("(")[0].strip() for d in map_data],
            y=[d["price_21k"] for d in map_data],
            marker_color=[C["gold"] if d["market"]==st.session_state["primary_mkt"] else rgba(C["gold_dark"], 0.67) for d in map_data],
            text=[f"{d['price_21k']:.2f} {d['currency']}" for d in map_data],
            textposition="auto",
        ))
        fig_map.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=C["card"],
            height=360, margin=dict(l=0,r=0,t=20,b=60),
            font=dict(color=C["text"],size=9),
            xaxis=dict(tickangle=-35, gridcolor=C["border2"]),
            yaxis=dict(gridcolor=C["border2"]),
            title=dict(text=("مقارنة أسعار الذهب 21K عالمياً" if use_ar else "Gold 21K Price Comparison — Global Markets"),
                       font=dict(color=C["gold"],size=12))
        )
        st.plotly_chart(fig_map, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: STRESS TEST / DRAWDOWN 📉
# ═══════════════════════════════════════════════════════════════════════════════
elif nav == L["nav_drawdown"]:
    use_ar = is_rtl()
    price_per_g_24k = g_ref / 31.1035  # USD per gram at 24K
    ph("📉  " + L.get("nav_drawdown","Stress Test").replace("📉  ",""),
       "اختبر تأثير انخفاض سعر الذهب على محفظتك — كن مستعداً لأي سيناريو" if use_ar else
       "Test how gold price drops affect your portfolio — prepare for any scenario")

    rtl = "direction:rtl;text-align:right;" if use_ar else ""

    # Input
    dd1, dd2 = st.columns(2, gap="large")
    with dd1:
        dd_value  = st.number_input("💰 " + ("قيمة المحفظة (USD)" if use_ar else "Portfolio Value (USD)"),
                                     min_value=100.0, value=10000.0, step=100.0, key="dd_value")
        dd_grams  = st.number_input("⚖️ " + ("الوزن (جرام)" if use_ar else "Weight (grams)"),
                                     min_value=0.0, value=dd_value/price_per_g_24k*0.875, step=10.0, key="dd_grams",
                                     help="21K grams equivalent")
        dd_curr   = st.selectbox("💱", ["USD","JOD","SAR","AED","EGP","KWD"], key="dd_curr")
        dd_fx     = {"USD":1.0,"JOD":0.709,"SAR":3.75,"AED":3.6725,"EGP":50.9,"KWD":0.307}.get(dd_curr,1)
        dd_val_lc = dd_value / dd_fx

        st.markdown(f"""
        <div style='background:{C["card2"]};border:1px solid {C["border2"]};border-radius:6px;padding:12px;margin-top:10px;'>
          <div style='font-size:12px;color:{C["muted"]};{rtl}'>
            {L['current_price_lbl']}: ${g_ref:,.2f}/oz ·
            {dd_grams:.1f}g = ${dd_value:,.2f} = {dd_val_lc:,.2f} {dd_curr}
          </div>
        </div>""", unsafe_allow_html=True)

    with dd2:
        # Scenarios
        scenarios = [
            ("🟡 -5%",  -5),  ("🟠 -10%",-10), ("🔴 -15%",-15),
            ("🔴 -20%",-20),  ("💀 -30%",-30), ("☠️ -50%",-50),
            ("🟢 +10%", 10),  ("🟢 +20%", 20), ("🚀 +30%", 30),
        ]
        sc_header = ["السيناريو","الخسارة/الربح (USD)",f"الخسارة/الربح ({dd_curr})","السعر الجديد"]
        sc_rows   = []
        for label, pct in scenarios:
            new_price = g_ref * (1 + pct/100)
            new_value = dd_grams/31.1035 * new_price
            diff_usd  = new_value - dd_value
            diff_lc   = diff_usd / dd_fx
            sc_rows.append((label, pct, new_price, diff_usd, diff_lc))

        for (label, pct, np_, du, dlc) in sc_rows:
            bar_col = C["green"] if pct>0 else C["red"]
            bar_w   = min(abs(pct)*2, 100)
            bar_dir = "right" if pct > 0 else "left"
            st.markdown(f"""
            <div style='display:flex;justify-content:space-between;align-items:center;
                        padding:6px 10px;border-bottom:1px solid {C["border"]}33;'>
              <div style='font-size:12px;font-weight:700;min-width:70px;'>{label}</div>
              <div style='flex:1;margin:0 10px;height:6px;background:{C["border"]};border-radius:3px;overflow:hidden;'>
                <div style='width:{bar_w}%;height:100%;background:{bar_col};
                            float:{"right" if pct<0 else "left"};border-radius:3px;'></div>
              </div>
              <div style='font-family:{C["font_m"]},monospace;font-size:12px;
                          color:{bar_col};min-width:110px;text-align:right;'>
                ${du:+,.0f} / {dlc:+,.0f} {dd_curr}
              </div>
              <div style='font-size:11px;color:{C["dim"]};min-width:80px;text-align:right;'>
                ${np_:,.0f}
              </div>
            </div>""", unsafe_allow_html=True)

    # Interactive slider
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"<div class='section-label'>{L['interactive_sim']}</div>", unsafe_allow_html=True)
    custom_pct = st.slider(
        ("نسبة التغيير %" if use_ar else "Price Change %"),
        -60.0, 60.0, 0.0, 0.5, key="dd_slider"
    )
    new_price_s  = g_ref * (1 + custom_pct/100)
    new_value_s  = dd_grams/31.1035 * new_price_s
    diff_s       = new_value_s - dd_value
    diff_s_lc    = diff_s / dd_fx
    sc_col       = C["green"] if diff_s >= 0 else C["red"]

    cv1, cv2, cv3, cv4 = st.columns(4)
    for col, lbl, val, vc in [
        (cv1,"🎯 " + ("السعر الجديد" if use_ar else "New Price"),f"${new_price_s:,.2f}",C["gold_pale"]),
        (cv2,"💼 " + ("القيمة الجديدة" if use_ar else "New Value"), f"${new_value_s:,.2f}", vc:=sc_col),
        (cv3,"📊 " + ("التغيير (USD)" if use_ar else "Change USD"),  f"${diff_s:+,.2f}", sc_col),
        (cv4,"💱 " + (f"التغيير ({dd_curr})" if use_ar else f"Change {dd_curr}"), f"{diff_s_lc:+,.2f}", sc_col),
    ]:
        with col:
            st.markdown(f"""<div class='stat-card' style='text-align:center;'>
              <div class='stat-label'>{lbl}</div>
              <div style='font-family:{C["font_m"]},monospace;font-size:18px;
                          font-weight:700;color:{vc};'>{val}</div>
            </div>""", unsafe_allow_html=True)

    # Historical max drawdown note
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='background:{C["card"]};border:1px solid {C["border"]};border-radius:6px;padding:14px 18px;'>
      <div style='font-size:11px;color:{C["muted"]};{rtl}'>
        📉 {"أكبر انخفاضات الذهب تاريخياً: 1980: -65% (في 2 سنة) · 2011-2015: -44% (تصحيح طويل) · 2020 مارس: -12% (كوفيد) · 2022: -15%. متوسط الانخفاض السنوي لا يتجاوز 10% في السنوات العشر الأخيرة." if use_ar else
         "📉 Historical gold max drawdowns: 1980: -65% (over 2 years) · 2011-2015: -44% (long correction) · Mar 2020: -12% (COVID) · 2022: -15%. Average annual drawdown has been <10% over the past decade."}
      </div>
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: GOLD SUPPLY & DEMAND ⛏️
# ═══════════════════════════════════════════════════════════════════════════════
elif nav == L["nav_supply"]:
    use_ar = is_rtl()
    ph("⛏️  " + L.get("nav_supply","Supply & Demand").replace("⛏️  ",""),
       "احتياطيات الذهب العالمية · إنتاج المناجم · مشتريات البنوك المركزية · صناديق ETF" if use_ar else
       "World gold reserves · Mine production · Central bank buying · ETF holdings")

    rtl = "direction:rtl;text-align:right;" if use_ar else ""

    # ── Central Bank Reserves — see dedicated page ────────────────────────────
    st.info(("📌 " + ("لمشاهدة احتياطيات البنوك المركزية بالتفصيل، انتقل إلى: " if use_ar else
             "For detailed Central Bank Reserves, visit: ") + L["nav_cb"]))
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Mine Production ───────────────────────────────────────────────────────
    st.markdown(f"<div class='section-label'>{L['mine_prod']}</div>", unsafe_allow_html=True)
    MINE_DATA = [
        ("🇨🇳","China / الصين",           375),
        ("🇦🇺","Australia / أستراليا",     320),
        ("🇷🇺","Russia / روسيا",           310),
        ("🇨🇦","Canada / كندا",            200),
        ("🇺🇸","USA / أمريكا",             170),
        ("🇬🇭","Ghana / غانا",             130),
        ("🇰🇿","Kazakhstan / كازاخستان",   120),
        ("🇲🇽","Mexico / المكسيك",         115),
        ("🇨🇩","DRC / الكونغو",            110),
        ("🇺🇿","Uzbekistan / أوزبكستان",   105),
    ]
    total_mine = sum(v for _,_,v in MINE_DATA)
    fig_mine = go.Figure(go.Bar(
        y=[f"{f} {n.split('/')[0].strip()}" for f,n,_ in MINE_DATA],
        x=[v for _,_,v in MINE_DATA],
        orientation="h",
        marker=dict(
            color=[C["gold"] if i==0 else rgba(C["gold_dark"], 0.8)
                   for i in range(len(MINE_DATA))],
            line=dict(color=rgba(C["gold"], 0.27), width=0.5)
        ),
        text=[f"{v}t ({v/total_mine*100:.1f}%)" for _,_,v in MINE_DATA],
        textposition="outside",
        textfont=dict(size=9, color=C["muted"]),
    ))
    fig_mine.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=C["card"],
        height=320, margin=dict(l=0,r=80,t=10,b=0),
        font=dict(color=C["text"], size=9),
        xaxis=dict(gridcolor=C["border2"], title=("طن" if use_ar else "Tonnes")),
        yaxis=dict(gridcolor=C["border2"]),
        title=dict(text=(f"الإجمالي: {total_mine:,}طن/سنة · المصدر: WGC 2024" if use_ar else f"Total: {total_mine:,}t/year · Source: WGC 2024"),
                   font=dict(size=9, color=C["dim"]))
    )
    st.plotly_chart(fig_mine, use_container_width=True)


    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"<div class='section-label'>{L['global_demand']}</div>", unsafe_allow_html=True)
    DEMAND_DATA = {
        "💍 Jewellery / المجوهرات":      2168.5,
        "🏦 Central Banks / البنوك المركزية": 1037.4,
        "📊 ETFs / الصناديق":             56.5,
        "🔧 Technology / التكنولوجيا":   327.0,
        "🪙 Bar & Coin / سبائك ومسكوكات": 1186.1,
    }
    fig_dem = go.Figure(go.Pie(
        labels=list(DEMAND_DATA.keys()),
        values=list(DEMAND_DATA.values()),
        hole=0.45,
        marker=dict(colors=[C["gold"], C["green"], "#4DA6FF", "#FF8C42", C["gold_pale"]],
                    line=dict(color=C["bg"], width=2)),
        textfont=dict(size=9, color=C["text"]),
        textinfo="label+percent",
        hovertemplate="%{label}<br>%{value:.0f}t (%{percent})<extra></extra>",
    ))
    fig_dem.add_annotation(
        text=f"{sum(DEMAND_DATA.values()):.0f}t<br>total",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=11, color=C["gold_hi"])
    )
    fig_dem.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", height=320,
        margin=dict(l=0,r=0,t=10,b=0),
        font=dict(color=C["text"], size=9),
        legend=dict(bgcolor=C["card2"], font=dict(size=8),
                    orientation="v", x=1, y=0.5),
        title=dict(text=("المصدر: مجلس الذهب العالمي 2024" if use_ar else ("المصدر: مجلس الذهب العالمي 2024" if use_ar else "Source: World Gold Council 2024")),
                   font=dict(size=9, color=C["dim"]))
    )
    st.plotly_chart(fig_dem, use_container_width=True)

    # ── ETF Holdings ─────────────────────────────────────────────────────────
    st.markdown(f"<div class='section-label'>{L['major_etfs']}</div>", unsafe_allow_html=True)
    ETF_DATA = [
    ("GLD",  "SPDR Gold Shares",         872.5,  -2.3, "NYSE"),
    ("IAU",  "iShares Gold Trust",        478.2,  -0.8, "NYSE"),
    ("GLDM", "SPDR Gold MiniShares",      98.4,   +0.2, "NYSE"),
    ("SGOL", "Aberdeen Physical Gold",    56.3,   -0.1, "NYSE"),
    ("PHAU", "Wisdomtree Physical Gold",  120.8,  +1.2, "LSE"),
    ]
    ec = st.columns(len(ETF_DATA))
    for col, (ticker, name, tonnes, chg, exchange) in zip(ec, ETF_DATA):
        chg_col = C["green"] if chg >= 0 else C["red"]
        with col:
            st.markdown(f"""
        <div class='stat-card' style='text-align:center;'>
          <div style='font-size:15px;font-weight:900;color:{C["gold_hi"]};'>{ticker}</div>
          <div style='font-size:9px;color:{C["muted"]};margin:3px 0;'>{name[:22]}</div>
          <div style='font-family:{C["font_m"]},monospace;font-size:14px;
                      color:{C["gold_pale"]};font-weight:700;'>{tonnes:.1f}t</div>
          <div style='font-size:11px;color:{chg_col};'>{chg:+.1f}t</div>
          <div style='font-size:9px;color:{C["dim"]};'>{exchange}</div>
        </div>""", unsafe_allow_html=True)

    # ── Key insight ───────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='background:{C["card"]};border:1px solid {C["gold"]}44;border-radius:6px;padding:14px 20px;'>
      <div style='font-size:12px;color:{C["muted"]};line-height:1.8;{rtl}'>
    {"💡 <b>نصيحة:</b> مشتريات البنوك المركزية هي أقوى محرك طويل الأمد لأسعار الذهب. عندما تشتري البنوك المركزية بكميات كبيرة → الذهب يرتفع عادةً خلال 6-12 شهراً. الطلب على المجوهرات في المنطقة العربية يرتفع قبل رمضان والأعياد. صناديق ETF تعكس المزاج المؤسسي — عندما ترتفع حيازاتها → تدفقات جديدة." if use_ar else
     "💡 <b>Key insight:</b> Central bank buying is the strongest long-term driver of gold prices. When central banks buy in bulk → gold typically rises over 6-12 months. Jewellery demand in Arab markets spikes before Ramadan and Eid. ETF holdings reflect institutional sentiment — rising holdings signal new institutional inflows."}
      </div>
    </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: CURRENCY CONVERTER 💱
# ═══════════════════════════════════════════════════════════════════════════════
elif nav == L["nav_currency"]:
    use_ar = is_rtl()
    ph("💱  " + L.get("nav_currency","Currency Converter").replace("💱  ",""),
       "حوّل بين جميع العملات وأسعار الذهب فورياً" if use_ar else
       "Convert between all currencies and gold prices instantly · Live rates")

    # Live FX rates — all as USD per 1 unit
    @st.cache_data(ttl=CACHE_TTL_PRICES)
    def get_all_fx():
        """Fetch current FX rates for all active markets in parallel.

        Returns
        -------
        dict
            Mapping of market key → live FX rate (USD → local currency).
        """
        rates = {
        "USD": 1.0,
        "JOD": 1.0/fetch_fx("USDJOD=X", False, 0.709),
        "SAR": 1.0/fetch_fx("USDSAR=X", False, 3.75),
        "AED": 1.0/fetch_fx("USDAED=X", False, 3.6725),
        "KWD": 1.0/fetch_fx("USDKWD=X", False, 0.307),
        "QAR": 1.0/fetch_fx("USDQAR=X", False, 3.64),
        "BHD": 1.0/fetch_fx("USDBHD=X", False, 0.377),
        "OMR": 1.0/fetch_fx("USDOMR=X", False, 0.385),
        "EGP": 1.0/fetch_fx("USDEGP=X", False, 50.9),
        "TRY": 1.0/fetch_fx("USDTRY=X", False, 32.0),
        "GBP": fetch_fx("GBPUSD=X", True, 1.27),
        "EUR": fetch_fx("EURUSD=X", True, 1.08),
        "JPY": 1.0/fetch_fx("USDJPY=X", False, 149.0),
        "CNY": 1.0/fetch_fx("USDCNY=X", False, 7.27),
        "INR": 1.0/fetch_fx("USDINR=X", False, 84.5),
        # Gold units as currency
        "XAU_OZ":   g_ref,           # 1 troy oz of gold in USD
        "GOLD_G":   g_ref/31.1035,   # 1 gram 24K
        "GOLD_21K": g_ref/31.1035*0.875,  # 1 gram 21K
        }
        return rates

        fx = get_all_fx()

        CURRENCY_INFO = {
        "USD":"🇺🇸 US Dollar","JOD":"🇯🇴 Jordanian Dinar",
        "SAR":"🇸🇦 Saudi Riyal","AED":"🇦🇪 UAE Dirham",
        "KWD":"🇰🇼 Kuwaiti Dinar","QAR":"🇶🇦 Qatari Riyal",
        "BHD":"🇧🇭 Bahraini Dinar","OMR":"🇴🇲 Omani Rial",
        "EGP":"🇪🇬 Egyptian Pound","TRY":"🇹🇷 Turkish Lira",
        "GBP":"🇬🇧 British Pound","EUR":"🇪🇺 Euro",
        "JPY":"🇯🇵 Japanese Yen","CNY":"🇨🇳 Chinese Yuan",
        "INR":"🇮🇳 Indian Rupee",
        "XAU_OZ":"🥇 Gold (1 troy oz)","GOLD_G":"🥇 Gold (1 gram 24K)",
        "GOLD_21K":"🥇 Gold (1 gram 21K)",
        }

    # Main converter
        cc1, cc2, cc3 = st.columns([2,1,2], gap="large")
        with cc1:
            from_curr = st.selectbox(
            ("من" if use_ar else "From Currency"),
            list(CURRENCY_INFO.keys()),
            format_func=lambda x: CURRENCY_INFO[x],
            index=1,  # JOD default
            key="cv_from"
        )
        amount = st.number_input(
            ("المبلغ" if use_ar else "Amount"),
            min_value=0.0, value=100.0, step=1.0, key="cv_amount"
        )

        with cc2:
            st.markdown("<br><br><br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style='text-align:center;font-size:28px;color:{C["gold"]};'>⇄</div>
        <div style='text-align:center;font-size:10px;color:{C["dim"]};margin-top:4px;'>
          {"أسعار مباشرة" if use_ar else "Live rates"}
        </div>""", unsafe_allow_html=True)

        with cc3:
            to_curr = st.selectbox(
            ("إلى" if use_ar else "To Currency"),
            list(CURRENCY_INFO.keys()),
            format_func=lambda x: CURRENCY_INFO[x],
            index=0,  # USD
            key="cv_to"
        )
        # Conversion
        from_usd = fx.get(from_curr, 1.0)  # USD per 1 unit of from_curr
        to_usd   = fx.get(to_curr,   1.0)  # USD per 1 unit of to_curr
        result   = amount * from_usd / to_usd if to_usd > 0 else 0

        st.markdown(f"""
        <div style='background:{C["card2"]};border:2px solid {C["gold"]}66;border-radius:8px;
                    padding:20px;text-align:center;margin-top:20px;'>
          <div style='font-size:11px;color:{C["muted"]};margin-bottom:6px;'>
            {amount:,.4f} {from_curr}  =
          </div>
          <div style='font-family:{C["font_m"]},monospace;font-size:32px;font-weight:900;
                      color:{C["gold_hi"]};'>
            {result:,.4f}
          </div>
          <div style='font-size:14px;color:{C["gold"]};margin-top:4px;font-weight:700;'>
            {to_curr}
          </div>
          <div style='font-size:10px;color:{C["dim"]};margin-top:6px;'>
            1 {from_curr} = {from_usd/to_usd:.6f} {to_curr}
          </div>
        </div>""", unsafe_allow_html=True)

    # ── Full conversion table ─────────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"<div class='section-label'>{'جدول التحويل الكامل · ' + str(amount) + ' ' + from_curr if use_ar else 'Full Conversion Table · ' + str(amount) + ' ' + from_curr}</div>", unsafe_allow_html=True)

        DISPLAY_CURRENCIES = ["USD","JOD","SAR","AED","KWD","QAR","BHD","EGP","GBP","EUR","TRY","XAU_OZ","GOLD_G","GOLD_21K"]
        per_row = 4
        for row_start in range(0, len(DISPLAY_CURRENCIES), per_row):
            row_cur  = DISPLAY_CURRENCIES[row_start:row_start+per_row]
            row_cols = st.columns(per_row)
            for col, curr in zip(row_cols, row_cur):
                to_rate = fx.get(curr, 1.0)
                val     = amount * from_usd / to_rate if to_rate > 0 else 0
            is_gold = "GOLD" in curr or "XAU" in curr
            is_home = curr in ["JOD","SAR","AED"]
            bc = C["gold"] if is_gold else (C["green"] if is_home else C["border2"])
            with col:
                st.markdown(f"""
                <div style='background:{C["card2"]};border:1px solid {bc}55;border-radius:6px;
                            padding:12px;text-align:center;margin-bottom:8px;
                            transition:transform .2s,box-shadow .2s;'>
                  <div style='font-size:11px;color:{C["muted"]};margin-bottom:4px;'>
                    {CURRENCY_INFO.get(curr,curr)}
                  </div>
                  <div style='font-family:{C["font_m"]},monospace;font-size:{"13px" if val>99999 else "16px"};
                              font-weight:700;color:{C["gold_hi"] if is_gold else C["gold_pale"]};'>
                    {_fmt_val(val)}
                  </div>
                </div>""", unsafe_allow_html=True)

    # ── Gold gram price table by market ──────────────────────────────────────
        st.markdown(f"<div class='section-label'>{L['gram_arab']}</div>", unsafe_allow_html=True)
        arab_currs = [("JOD","🇯🇴",0.709),("SAR","🇸🇦",3.75),("AED","🇦🇪",3.6725),
                  ("KWD","🇰🇼",0.307),("QAR","🇶🇦",3.64),("BHD","🇧🇭",0.377),("EGP","🇪🇬",50.9)]
        gc = st.columns(len(arab_currs))
        for col,(curr,flag,approx) in zip(gc,arab_currs):
            rate  = fx.get(curr,1.0)  # USD per 1 local unit
            g24   = (g_ref/31.1035)/rate   # price of 1 gram 24K in local currency
            g21   = g24*0.875
            with col:
                st.markdown(f"""
                <div class='stat-card' style='text-align:center;padding:10px;'>
                  <div style='font-size:20px;'>{flag}</div>
                  <div style='font-size:9px;color:{C["muted"]};'>{curr}</div>
              <div style='font-family:{C["font_m"]},monospace;font-size:13px;
                          color:{C["gold_hi"]};font-weight:700;margin-top:4px;'>{g24:,.3f}</div>
              <div style='font-size:9px;color:{C["dim"]};'>24K/g</div>
              <div style='font-family:{C["font_m"]},monospace;font-size:11px;color:{C["gold"]};'>{g21:,.3f}</div>
              <div style='font-size:9px;color:{C["dim"]};'>21K/g</div>
            </div>""", unsafe_allow_html=True)

    # ── Refresh note ─────────────────────────────────────────────────────────────
    _cx1, _cx2 = st.columns([2, 4])
    with _cx1:
        if st.button("🔄 " + ("تحديث" if use_ar else "Refresh"), key="fx_refresh",
                     use_container_width=True):
            get_all_fx.clear()
            st.rerun()
    with _cx2:
        st.markdown(f"<div style='font-size:10px;color:{C["dim"]};padding-top:8px;'>⏱ {L["auto_refresh_60"]}</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: TRADING SIGNALS 📡
# ═══════════════════════════════════════════════════════════════════════════════
elif nav == L["nav_signals"]:
    use_ar = is_rtl()
    ph("📡  " + L.get("nav_signals","Trading Signals").replace("📡  ",""),
       "جميع إشارات الذهب في مكان واحد · محدّث تلقائياً" if use_ar else
       "All gold signals in one place · Auto-updated with live data")

    rtl = "direction:rtl;text-align:right;" if use_ar else ""

    # ── Compute all signals from live data ────────────────────────────────────
    @st.cache_data(ttl=CACHE_TTL_HISTORY)
    def compute_all_signals():
        """Compute all ten trading signals for the Signals page.
    
        Returns
        -------
        list[dict]
            Each element contains name_ar, name_en, value, verdict,
            strength, reason_ar, reason_en.
        """
        df = fetch_history("1y", "GC=F")
        if df.empty: return {}
        cl = find_col(df, ["Close","Close_GC=F"])
        hi = find_col(df, ["High","High_GC=F"])
        lo = find_col(df, ["Low","Low_GC=F"])
        if not cl: return {}
        c = df[cl].dropna()
        if len(c) < 20: return {}

        # RSI(14)
        d  = c.diff()
        ag = d.clip(lower=0).rolling(14).mean()
        al = (-d.clip(upper=0)).rolling(14).mean()
        rsi = float((100-(100/(1+ag/al.replace(0,1e-9)))).iloc[-1])

        # Moving Averages
        ma20  = float(c.rolling(20).mean().iloc[-1])  if len(c)>=20  else None
        ma50  = float(c.rolling(50).mean().iloc[-1])  if len(c)>=50  else None
        ma100 = float(c.rolling(100).mean().iloc[-1]) if len(c)>=100 else None
        ma200 = float(c.rolling(200).mean().iloc[-1]) if len(c)>=200 else None
        price = float(c.iloc[-1])

        # MA Crossovers
        golden_cross = ma50 and ma200 and ma50 > ma200
        death_cross  = ma50 and ma200 and ma50 < ma200
        # Check if crossover happened recently (within 10 days)
        if len(c) >= 200:
            ma50_10d  = float(c.rolling(50).mean().iloc[-10])
            ma200_10d = float(c.rolling(200).mean().iloc[-10])
            recent_golden = ma50_10d <= ma200_10d and golden_cross
            recent_death  = ma50_10d >= ma200_10d and death_cross
        else:
            recent_golden = recent_death = False

        # MACD
        ema12  = c.ewm(span=12, adjust=False).mean()
        ema26  = c.ewm(span=26, adjust=False).mean()
        macd_line   = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        macd_hist   = macd_line - signal_line
        macd_val    = float(macd_line.iloc[-1])
        sig_val     = float(signal_line.iloc[-1])
        hist_val    = float(macd_hist.iloc[-1])
        # MACD crossover
        hist_prev   = float(macd_hist.iloc[-2]) if len(macd_hist)>1 else hist_val
        macd_cross_bull = hist_prev < 0 and hist_val > 0
        macd_cross_bear = hist_prev > 0 and hist_val < 0

        # Bollinger Bands
        bm  = c.rolling(20).mean()
        bs  = c.rolling(20).std()
        bb_upper = float((bm+2*bs).iloc[-1])
        bb_lower = float((bm-2*bs).iloc[-1])
        bb_mid   = float(bm.iloc[-1])
        bb_width = (bb_upper - bb_lower) / bb_mid * 100  # % bandwidth
        bb_pos   = (price - bb_lower) / (bb_upper - bb_lower) * 100 if (bb_upper-bb_lower) > 0 else 50
        bb_squeeze = bb_width < float((bm.rolling(20).mean()).iloc[-1] / bm.iloc[-1] * 2)

        # Stochastic %K %D
        if hi and lo and len(c) >= 14:
            low14  = df[lo].rolling(14).min().dropna()
            hi14   = df[hi].rolling(14).max().dropna()
            cl_14  = df[cl].dropna()[-len(low14):]
            k_line = 100*(cl_14.values - low14.values)/(hi14.values - low14.values + 1e-9)
            k_ser  = pd.Series(k_line)
            d_ser  = k_ser.rolling(3).mean()
            stoch_k = float(k_ser.iloc[-1])
            stoch_d = float(d_ser.iloc[-1]) if not pd.isna(d_ser.iloc[-1]) else stoch_k
        else:
            stoch_k = stoch_d = 50

        # ATR (Average True Range)
        atr = float(c.diff().abs().rolling(14).mean().iloc[-1])

        # Volume trend (HL range proxy)
        hl_range_now = float(c.diff().abs().tail(5).mean())
        hl_range_avg = float(c.diff().abs().mean())
        vol_trend = "high" if hl_range_now > hl_range_avg*1.2 else ("low" if hl_range_now < hl_range_avg*0.8 else "normal")

        # 52W levels
        hi52 = float(c.tail(252).max())
        lo52 = float(c.tail(252).min())
        pos52 = (price - lo52) / (hi52 - lo52) * 100 if (hi52-lo52) > 0 else 50

        # Momentum
        mom10 = float(c.pct_change(10).iloc[-1]*100) if len(c)>=11 else 0
        mom30 = float(c.pct_change(30).iloc[-1]*100) if len(c)>=31 else 0

        return dict(
        price=price, rsi=rsi,
        ma20=ma20, ma50=ma50, ma100=ma100, ma200=ma200,
        golden_cross=golden_cross, death_cross=death_cross,
        recent_golden=recent_golden, recent_death=recent_death,
        macd=macd_val, macd_sig=sig_val, macd_hist=hist_val,
        macd_cross_bull=macd_cross_bull, macd_cross_bear=macd_cross_bear,
        bb_upper=bb_upper, bb_lower=bb_lower, bb_mid=bb_mid,
        bb_pos=bb_pos, bb_width=bb_width, bb_squeeze=bb_squeeze,
        stoch_k=stoch_k, stoch_d=stoch_d,
        atr=atr, vol_trend=vol_trend,
        hi52=hi52, lo52=lo52, pos52=pos52,
        mom10=mom10, mom30=mom30,
        )

        with st.spinner("⚡ " + ("جاري تحليل البيانات..." if use_ar else "Computing signals...")):
            sig = compute_all_signals()

        if not sig:
            st.error("⚠️ " + ("تعذّر تحميل البيانات" if use_ar else "Could not load signal data"))
        st.stop()

        price = sig["price"]

        # ── Helper: classify each signal ─────────────────────────────────────────
        def signal_row(name_ar, name_en, value_str, verdict, strength, reason_ar, reason_en):
            """Returns (signal_dict) for building the table."""
            return dict(name_ar=name_ar, name_en=name_en, value=value_str,
                        verdict=verdict, strength=strength,
                        reason_ar=reason_ar, reason_en=reason_en)

        # Build signal list
        signals = []

        # 1. RSI
        rsi = sig["rsi"]
        if rsi < 30:
            signals.append(signal_row(
        "RSI(14)", "RSI(14)", f"{rsi:.1f}",
        "BUY", "Strong",
        f"ذروة بيع — RSI={rsi:.1f} تحت 30 يشير لإفراط في البيع → فرصة شراء",
        f"Oversold — RSI={rsi:.1f} below 30 signals excessive selling → buy opportunity"
        ))
        elif rsi > 70:
            signals.append(signal_row(
        "RSI(14)", "RSI(14)", f"{rsi:.1f}",
        "SELL", "Strong",
        f"ذروة شراء — RSI={rsi:.1f} فوق 70 يشير لإفراط في الشراء → خطر تصحيح",
        f"Overbought — RSI={rsi:.1f} above 70 signals excessive buying → correction risk"
        ))
        elif rsi < 45:
            signals.append(signal_row(
        "RSI(14)", "RSI(14)", f"{rsi:.1f}",
        "BUY", "Weak",
        f"RSI={rsi:.1f} ميل للانخفاض لكن لم يصل لذروة البيع بعد",
        f"RSI={rsi:.1f} tending lower but not yet oversold — mild bullish bias"
        ))
        elif rsi > 58:
            signals.append(signal_row(
        "RSI(14)", "RSI(14)", f"{rsi:.1f}",
        "SELL", "Weak",
        f"RSI={rsi:.1f} ميل للارتفاع — مراقبة مستوى 70",
        f"RSI={rsi:.1f} tending higher — watch for 70 overbought level"
        ))
        else:
            signals.append(signal_row(
        "RSI(14)", "RSI(14)", f"{rsi:.1f}",
        "NEUTRAL", "Neutral",
        f"RSI={rsi:.1f} في منطقة محايدة (30-70)",
        f"RSI={rsi:.1f} in neutral zone (30-70)"
        ))

        # 2. MA50 vs Price
        ma50 = sig["ma50"] or price
        if price > ma50:
            signals.append(signal_row(
        "السعر vs MA50", "Price vs MA50",
        f"${price:,.0f} > ${ma50:,.0f}",
        "BUY", "Moderate",
        "السعر فوق المتوسط المتحرك 50 يوم — اتجاه صاعد في المدى المتوسط",
        "Price above MA50 — bullish medium-term trend confirmed"
        ))
        else:
            signals.append(signal_row(
        "السعر vs MA50", "Price vs MA50",
        f"${price:,.0f} < ${ma50:,.0f}",
        "SELL", "Moderate",
        "السعر تحت المتوسط المتحرك 50 يوم — ضغط هبوطي في المدى المتوسط",
        "Price below MA50 — bearish medium-term pressure"
        ))

        # 3. MA200 vs Price
        ma200 = sig["ma200"] or price
        if price > ma200:
            signals.append(signal_row(
        "السعر vs MA200", "Price vs MA200",
        f"${price:,.0f} > ${ma200:,.0f}",
        "BUY", "Strong",
        "السعر فوق المتوسط المتحرك 200 يوم — اتجاه صاعد طويل الأمد",
        "Price above MA200 — confirmed long-term bull trend"
        ))
        else:
            signals.append(signal_row(
        "السعر vs MA200", "Price vs MA200",
        f"${price:,.0f} < ${ma200:,.0f}",
        "SELL", "Strong",
        "السعر تحت المتوسط المتحرك 200 يوم — اتجاه هابط طويل الأمد",
        "Price below MA200 — long-term bear trend warning"
        ))

        # 4. Golden/Death Cross
        if sig["recent_golden"]:
            signals.append(signal_row(
        "تقاطع ذهبي", "Golden Cross",
        "MA50 ↑ MA200",
        "BUY", "Strong",
        "تقاطع ذهبي حديث — MA50 اخترقت MA200 للأعلى! أقوى إشارة صعودية",
        "Recent Golden Cross! MA50 crossed above MA200 — strongest bullish signal"
        ))
        elif sig["recent_death"]:
            signals.append(signal_row(
        "تقاطع موت", "Death Cross",
        "MA50 ↓ MA200",
        "SELL", "Strong",
        "تقاطع الموت الحديث — MA50 اخترقت MA200 للأسفل! أقوى إشارة هبوطية",
        "Recent Death Cross! MA50 crossed below MA200 — strongest bearish signal"
        ))
        elif sig["golden_cross"]:
            signals.append(signal_row(
        "تقاطع MA50/MA200", "MA Cross",
        "Golden 🟢",
        "BUY", "Moderate",
        "MA50 فوق MA200 — هيكل صاعد مستمر",
        "MA50 above MA200 — ongoing bullish structure"
        ))
        else:
            signals.append(signal_row(
        "تقاطع MA50/MA200", "MA Cross",
        "Death ⚠️",
        "SELL", "Moderate",
        "MA50 تحت MA200 — هيكل هابط مستمر",
        "MA50 below MA200 — ongoing bearish structure"
        ))

        # 5. MACD
        macd = sig["macd"]; macd_s = sig["macd_sig"]; hist = sig["macd_hist"]
        if sig["macd_cross_bull"]:
            signals.append(signal_row(
        "MACD", "MACD",
        f"{macd:.2f} / Signal: {macd_s:.2f}",
        "BUY", "Strong",
        "تقاطع MACD الصاعد! الهيستوجرام انتقل من سالب لموجب → شراء",
        "MACD bullish crossover! Histogram flipped positive → buy signal"
        ))
        elif sig["macd_cross_bear"]:
            signals.append(signal_row(
        "MACD", "MACD",
        f"{macd:.2f} / Signal: {macd_s:.2f}",
        "SELL", "Strong",
        "تقاطع MACD الهابط! الهيستوجرام انتقل من موجب لسالب → بيع",
        "MACD bearish crossover! Histogram flipped negative → sell signal"
        ))
        elif macd > macd_s and hist > 0:
            signals.append(signal_row(
        "MACD", "MACD",
        f"{macd:.2f} > {macd_s:.2f}",
        "BUY", "Weak",
        f"MACD فوق خط الإشارة والهيستوجرام موجب ({hist:+.2f})",
        f"MACD above signal, histogram positive ({hist:+.2f}) — mild bullish"
        ))
        else:
            signals.append(signal_row(
        "MACD", "MACD",
        f"{macd:.2f} < {macd_s:.2f}",
        "SELL", "Weak",
        f"MACD تحت خط الإشارة والهيستوجرام سالب ({hist:+.2f})",
        f"MACD below signal, histogram negative ({hist:+.2f}) — mild bearish"
        ))

        # 6. Bollinger Bands
        bb_pos = sig["bb_pos"]
        if bb_pos < 15:
            signals.append(signal_row(
        "بولينجر باند", "Bollinger Bands",
        f"Position: {bb_pos:.0f}%",
        "BUY", "Strong",
        f"السعر قرب الحد السفلي ({bb_pos:.0f}%) — ذروة بيع إحصائياً → ارتداد محتمل",
        f"Price near lower band ({bb_pos:.0f}%) — statistically oversold → bounce likely"
        ))
        elif bb_pos > 85:
            signals.append(signal_row(
        "بولينجر باند", "Bollinger Bands",
        f"Position: {bb_pos:.0f}%",
        "SELL", "Strong",
        f"السعر قرب الحد العلوي ({bb_pos:.0f}%) — ذروة شراء إحصائياً → تصحيح محتمل",
        f"Price near upper band ({bb_pos:.0f}%) — statistically overbought → pullback likely"
        ))
        elif bb_pos < 35:
            signals.append(signal_row(
        "بولينجر باند", "Bollinger Bands",
        f"Position: {bb_pos:.0f}%",
        "BUY", "Weak",
        f"السعر في النصف السفلي من النطاق ({bb_pos:.0f}%) — ميل صاعد",
        f"Price in lower half of band ({bb_pos:.0f}%) — mild bullish lean"
        ))
        elif bb_pos > 65:
            signals.append(signal_row(
        "بولينجر باند", "Bollinger Bands",
        f"Position: {bb_pos:.0f}%",
        "SELL", "Weak",
        f"السعر في النصف العلوي من النطاق ({bb_pos:.0f}%) — ميل هبوطي",
        f"Price in upper half of band ({bb_pos:.0f}%) — mild bearish lean"
        ))
        else:
            signals.append(signal_row(
        "بولينجر باند", "Bollinger Bands",
        f"Position: {bb_pos:.0f}%",
        "NEUTRAL", "Neutral",
        "السعر في منتصف نطاق بولينجر",
        "Price in middle of Bollinger Band"
        ))

        # 7. Stochastic
        sk = sig["stoch_k"]; sd2 = sig["stoch_d"]
        if sk < 20 and sd2 < 20:
            signals.append(signal_row(
        "ستوكاستيك", "Stochastic",
        f"%K={sk:.0f} %D={sd2:.0f}",
        "BUY", "Strong",
        "كلا المؤشرين تحت 20 — ذروة بيع قوية",
        "Both %K and %D below 20 — strong oversold signal"
        ))
        elif sk > 80 and sd2 > 80:
            signals.append(signal_row(
        "ستوكاستيك", "Stochastic",
        f"%K={sk:.0f} %D={sd2:.0f}",
        "SELL", "Strong",
        "كلا المؤشرين فوق 80 — ذروة شراء قوية",
        "Both %K and %D above 80 — strong overbought signal"
        ))
        elif sk < 20:
            signals.append(signal_row(
        "ستوكاستيك", "Stochastic",
        f"%K={sk:.0f} %D={sd2:.0f}",
        "BUY", "Moderate",
        f"%K={sk:.0f} في منطقة ذروة البيع",
        f"%K={sk:.0f} in oversold zone"
        ))
        elif sk > 80:
            signals.append(signal_row(
        "ستوكاستيك", "Stochastic",
        f"%K={sk:.0f} %D={sd2:.0f}",
        "SELL", "Moderate",
        f"%K={sk:.0f} في منطقة ذروة الشراء",
        f"%K={sk:.0f} in overbought zone"
        ))
        else:
            signals.append(signal_row(
        "ستوكاستيك", "Stochastic",
        f"%K={sk:.0f} %D={sd2:.0f}",
        "NEUTRAL", "Neutral",
        f"%K={sk:.0f} في المنطقة المحايدة",
        f"%K={sk:.0f} in neutral zone"
        ))

        # 8. 52-Week Position
        pos52 = sig["pos52"]
        if pos52 > 90:
            signals.append(signal_row(
        "موقع 52 أسبوع", "52-Week Position",
        f"{pos52:.0f}% of range",
        "SELL", "Moderate",
        f"السعر قرب أعلى مستوى في 52 أسبوع ({pos52:.0f}%) — مقاومة قوية محتملة",
        f"Near 52-week high ({pos52:.0f}%) — strong resistance zone"
        ))
        elif pos52 < 10:
            signals.append(signal_row(
        "موقع 52 أسبوع", "52-Week Position",
        f"{pos52:.0f}% of range",
        "BUY", "Moderate",
        f"السعر قرب أدنى مستوى في 52 أسبوع ({pos52:.0f}%) — دعم قوي محتمل",
        f"Near 52-week low ({pos52:.0f}%) — strong support zone"
        ))
        else:
            signals.append(signal_row(
        "موقع 52 أسبوع", "52-Week Position",
        f"{pos52:.0f}% of range",
        "NEUTRAL", "Neutral",
        f"السعر في منتصف نطاق 52 أسبوع ({pos52:.0f}%)",
        f"Price at {pos52:.0f}% of 52-week range"
        ))

        # 9. VIX (fear)
        vix = live["vix"]["price"]
        if vix > 25:
            signals.append(signal_row(
        "مؤشر VIX", "VIX Fear",
        f"{vix:.1f}",
        "BUY", "Moderate",
        f"VIX={vix:.1f} مرتفع — خوف في السوق → طلب على الملاذات الآمنة → صعود للذهب",
        f"VIX={vix:.1f} elevated — market fear → safe-haven demand → bullish gold"
        ))
        elif vix < 15:
            signals.append(signal_row(
        "مؤشر VIX", "VIX Fear",
        f"{vix:.1f}",
        "SELL", "Weak",
        f"VIX={vix:.1f} منخفض — هدوء السوق يقلل الطلب على الذهب كملاذ آمن",
        f"VIX={vix:.1f} low — calm markets reduce safe-haven demand for gold"
        ))
        else:
            signals.append(signal_row(
        "مؤشر VIX", "VIX Fear",
        f"{vix:.1f}",
        "NEUTRAL", "Neutral",
        f"VIX={vix:.1f} في المنطقة المحايدة (15-25)",
        f"VIX={vix:.1f} in neutral zone (15-25)"
        ))

        # 10. DXY
        dxy_pct = live["dxy"]["pct"]
        if dxy_pct < -0.3:
            signals.append(signal_row(
        "مؤشر الدولار DXY", "DXY Index",
        f"{dxy_pct:+.2f}%",
        "BUY", "Strong",
        f"الدولار يضعف ({dxy_pct:+.2f}%) — علاقة عكسية مع الذهب → دعم للصعود",
        f"USD weakening ({dxy_pct:+.2f}%) — inverse relationship supports gold rally"
        ))
        elif dxy_pct > 0.3:
            signals.append(signal_row(
        "مؤشر الدولار DXY", "DXY Index",
        f"{dxy_pct:+.2f}%",
        "SELL", "Strong",
        f"الدولار يقوى ({dxy_pct:+.2f}%) — ضغط على أسعار الذهب",
        f"USD strengthening ({dxy_pct:+.2f}%) — headwind for gold prices"
        ))
        else:
            signals.append(signal_row(
        "مؤشر الدولار DXY", "DXY Index",
        f"{dxy_pct:+.2f}%",
        "NEUTRAL", "Neutral",
        f"الدولار مستقر ({dxy_pct:+.2f}%) — تأثير محايد على الذهب",
        f"USD stable ({dxy_pct:+.2f}%) — neutral impact on gold"
        ))

        # ── Calculate overall signal ───────────────────────────────────────────────
        score_map = {"BUY": 1, "NEUTRAL": 0, "SELL": -1}
        weight_map = {"Strong": 2, "Moderate": 1.5, "Weak": 1, "Neutral": 0}
        total_w = 0; total_s = 0
        buy_count = sell_count = neutral_count = 0
        for s in signals:
            w  = weight_map.get(s["strength"], 1)
        sv = score_map.get(s["verdict"], 0)
        total_s += sv * w
        total_w += w
        if s["verdict"] == "BUY":    buy_count    += 1
        elif s["verdict"] == "SELL": sell_count   += 1
        else:                        neutral_count += 1

        avg_score  = total_s / total_w if total_w > 0 else 0
        confidence = abs(avg_score) * 100

        if avg_score > 0.4:
            overall = "BUY";     overall_ar = "شراء";   overall_col = C["green"]; overall_icon = "▲"
        elif avg_score > 0.15:
            overall = "WEAK BUY"; overall_ar = "شراء بحذر"; overall_col = "#52B788"; overall_icon = "↗"
        elif avg_score < -0.4:
            overall = "SELL";    overall_ar = "بيع";    overall_col = C["red"];   overall_icon = "▼"
        elif avg_score < -0.15:
            overall = "WEAK SELL"; overall_ar = "بيع بحذر"; overall_col = "#FF8C42"; overall_icon = "↘"
        else:
            overall = "NEUTRAL"; overall_ar = "محايد";  overall_col = C["gold"]; overall_icon = "→"

        # ── Big verdict display ───────────────────────────────────────────────────
        st.components.v1.html(f"""
        <style>
          @keyframes pulseVerdict {{
        0%,100% {{box-shadow:0 0 0 0 {overall_col}44;}}
        50%     {{box-shadow:0 0 20px 6px {overall_col}22;}}
          }}
          .verdict-box {{
        background:linear-gradient(135deg,{C['card2']},{C['card']});
        border:2px solid {overall_col}88;border-radius:12px;
        padding:24px 32px;text-align:center;
        animation:pulseVerdict 2.5s ease infinite;
        font-family:Georgia,serif;
          }}
        </style>
        <div class='verdict-box'>
          <div style='font-size:10px;letter-spacing:.3em;color:{C['muted']};text-transform:uppercase;
              margin-bottom:10px;'>
        {L['overall_signal_ts']}{datetime.datetime.now().strftime('%H:%M UTC')}
          </div>
          <div style='font-size:72px;font-weight:900;color:{overall_col};line-height:1;'>
        {overall_icon}
          </div>
          <div style='font-size:36px;font-weight:900;color:{overall_col};margin:8px 0;'>
        {overall_ar if use_ar else overall}
          </div>
          <div style='font-size:15px;color:{C['muted']};margin-top:6px;'>
        ${price:,.2f} · {L['confidence_lbl']}: {confidence:.0f}%
          </div>
          <div style='display:flex;justify-content:center;gap:24px;margin-top:16px;flex-wrap:wrap;'>
        <div>
          <div style='font-size:24px;font-weight:700;color:{C['green']};'>{buy_count}</div>
          <div style='font-size:9px;color:{C['muted']};letter-spacing:.1em;'>
        {L['buy_lbl']}
          </div>
        </div>
        <div>
          <div style='font-size:24px;font-weight:700;color:{C['muted']};'>{neutral_count}</div>
          <div style='font-size:9px;color:{C['muted']};letter-spacing:.1em;'>
        {L['neutral_sig']}
          </div>
        </div>
        <div>
          <div style='font-size:24px;font-weight:700;color:{C['red']};'>{sell_count}</div>
          <div style='font-size:9px;color:{C['muted']};letter-spacing:.1em;'>
        {L['sell_lbl']}
          </div>
        </div>
          </div>
        </div>
        """, height=260, scrolling=False)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Signals table ─────────────────────────────────────────────────────────
        st.markdown(f"<div class='section-label'>{L['all_signals']}</div>", unsafe_allow_html=True)

        verdict_cfg = {
        "BUY":      (C["green"],  "🟢", "شراء"     if use_ar else "BUY"),
        "WEAK BUY": ("#52B788",   "🟩", "شراء ضعيف" if use_ar else "WEAK BUY"),
        "SELL":     (C["red"],    "🔴", "بيع"      if use_ar else "SELL"),
        "WEAK SELL":("#FF8C42",   "🟠", "بيع ضعيف" if use_ar else "WEAK SELL"),
        "NEUTRAL":  (C["muted"],  "🟡", "محايد"    if use_ar else "NEUTRAL"),
        }
        strength_cfg = {
        "Strong":  (C["gold_hi"], "●●●"),
        "Moderate":(C["gold"],    "●●○"),
        "Weak":    (C["muted"],   "●○○"),
        "Neutral": (C["dim"],     "○○○"),
        }

        for i, s in enumerate(signals):
            vcol, vicon, vlbl = verdict_cfg.get(s["verdict"], (C["muted"],"⚪",s["verdict"]))
        scol, sbar        = strength_cfg.get(s["strength"], (C["muted"],"●○○"))
        bg = C["card2"] if i%2==0 else C["card"]

        st.markdown(f"""
        <div style='background:{bg};border:1px solid {C['border']}22;
                border-left:4px solid {vcol};border-radius:5px;
                padding:12px 16px;margin-bottom:4px;
                display:flex;justify-content:space-between;
                align-items:center;flex-wrap:wrap;gap:8px;
                transition:transform .2s,box-shadow .2s;'>
          <div style='min-width:160px;'>
        <div style='font-size:13px;font-weight:700;color:{C['text']};'>
          {s['name_ar'] if use_ar else s['name_en']}
        </div>
        <div style='font-family:{C['font_m']},monospace;font-size:11px;
                    color:{C['muted']};margin-top:2px;'>{s['value']}</div>
          </div>
          <div style='flex:1;font-size:12px;color:{C['dim']};
                  {"direction:rtl;text-align:right;" if use_ar else ""}
                  min-width:200px;line-height:1.5;'>
        {s['reason_ar'] if use_ar else s['reason_en']}
          </div>
          <div style='display:flex;align-items:center;gap:10px;flex-shrink:0;'>
        <div style='font-size:9px;color:{scol};letter-spacing:.05em;'
             title="{s['strength']}">{sbar}</div>
        <div style='background:{vcol}22;color:{vcol};font-size:11px;
                    font-weight:700;padding:4px 14px;border-radius:12px;
                    border:1px solid {vcol}55;white-space:nowrap;'>
          {vicon} {vlbl}
        </div>
          </div>
        </div>""", unsafe_allow_html=True)

        # ── Suggested action ──────────────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        atr = sig["atr"] or price*0.01
        sl_buy  = price - 1.5*atr
        tp1_buy = price + 1.5*atr
        tp2_buy = price + 3.0*atr
        sl_sell = price + 1.5*atr
        tp1_sell= price - 1.5*atr

        if "BUY" in overall:
            action_html = f"""
        <div style='background:{C['green']}14;border:2px solid {C['green']}55;
                border-radius:8px;padding:18px 22px;'>
          <div style='font-size:14px;font-weight:700;color:{C['green']};margin-bottom:10px;{rtl}'>
        🟢 {L['action_buy']}
          </div>
          <div style='font-family:{C['font_m']},monospace;font-size:12px;
                  color:{C['text']};display:flex;gap:20px;flex-wrap:wrap;{rtl}'>
        <div>📥 {L['entry_lbl']}: <b style='color:{C['gold_pale']};'>${price:,.2f}</b></div>
        <div>🛑 {L['stop_loss_lbl']}: <b style='color:{C['red']};'>${sl_buy:,.2f}</b></div>
        <div>🎯 TP1: <b style='color:{C['green']};'>${tp1_buy:,.2f}</b></div>
        <div>🎯 TP2: <b style='color:{C['green']};'>${tp2_buy:,.2f}</b></div>
        <div>⚖️ R/R: <b style='color:{C['gold']};'>1:1.5</b></div>
          </div>
        </div>"""
        elif "SELL" in overall:
            action_html = f"""
        <div style='background:{C['red']}14;border:2px solid {C['red']}55;
                border-radius:8px;padding:18px 22px;'>
          <div style='font-size:14px;font-weight:700;color:{C['red']};margin-bottom:10px;{rtl}'>
        🔴 {L['action_sell']}
          </div>
          <div style='font-family:{C['font_m']},monospace;font-size:12px;
                  color:{C['text']};display:flex;gap:20px;flex-wrap:wrap;{rtl}'>
        <div>📤 {L['exit_sell']}: <b style='color:{C['gold_pale']};'>${price:,.2f}</b></div>
        <div>🛑 SL: <b style='color:{C['red']};'>${sl_sell:,.2f}</b></div>
        <div>🎯 TP1: <b style='color:{C['green']};'>${tp1_sell:,.2f}</b></div>
          </div>
        </div>"""
        else:
            action_html = f"""
        <div style='background:{C['gold']}14;border:2px solid {C['gold']}55;
                border-radius:8px;padding:18px 22px;'>
          <div style='font-size:14px;font-weight:700;color:{C['gold']};margin-bottom:8px;{rtl}'>
        🟡 {L['action_wait']}
          </div>
          <div style='font-size:12px;color:{C['muted']};{rtl}'>
        {'الإشارات متضاربة — انتظر تأكيداً من مؤشرين على الأقل قبل الدخول' if use_ar else
         'Signals are mixed — wait for confirmation from at least 2 indicators before entering'}
          </div>
        </div>"""

        st.markdown(action_html, unsafe_allow_html=True)

        # Disclaimer
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style='font-size:10px;color:{C['dim']};text-align:center;'>
          ⚠️ {'هذه إشارات تحليلية تعليمية فقط وليست نصيحة مالية. التداول ينطوي على مخاطر — استشر مستشارك المالي.' if use_ar else
          'These are educational analysis signals only, not financial advice. Trading involves risk — consult your financial advisor.'}
        </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: AI NEWS SENTIMENT 🧠
# ═══════════════════════════════════════════════════════════════════════════════
    # ── Trade Details: Entry · Target · Stop Loss ────────────────────────────
    st.markdown("<div class='mansa-divider'></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='section-label'>💹 {L['sig_trade']}</div>",
                unsafe_allow_html=True)

    # Compute entry/target/stop from live data + TA
    _g = g_ref
    # Get TA values from signals (already computed above)
    _sig_data  = compute_all_signals()
    _atr   = _sig_data.get("atr",   _g * 0.012) if isinstance(_sig_data, dict) else _g * 0.012
    _rsi   = _sig_data.get("rsi",   50)         if isinstance(_sig_data, dict) else 50
    _ma50  = _sig_data.get("ma50",  _g)          if isinstance(_sig_data, dict) else _g
    _ma200 = _sig_data.get("ma200", _g)          if isinstance(_sig_data, dict) else _g
    _verdict = (_sig_data.get("overall_verdict", "NEUTRAL")
                if isinstance(_sig_data, dict) else "NEUTRAL")

    # Entry: current price for aggressive, 0.3×ATR below for conservative
    _entry_agg  = _g
    _entry_cons = _g - 0.3 * _atr

    # Target: 1.5×ATR above entry (bullish) or 1.5×ATR below (bearish)
    if _verdict == "BUY":
        _target   = _entry_agg + 1.5 * _atr
        _stop     = _entry_agg - _atr
        _rr       = 1.5
        _dir_col  = C["green"]
        _dir_lbl  = ("شراء — تفاؤل" if use_ar else "BUY — Bullish setup")
    elif _verdict == "SELL":
        _target   = _entry_agg - 1.5 * _atr
        _stop     = _entry_agg + _atr
        _rr       = 1.5
        _dir_col  = C["red"]
        _dir_lbl  = ("بيع — تشاؤم" if use_ar else "SELL — Bearish setup")
    else:
        _target   = _g + _atr
        _stop     = _g - _atr
        _rr       = 1.0
        _dir_col  = C["muted"]
        _dir_lbl  = ("محايد — انتظر تأكيداً" if use_ar else "NEUTRAL — Wait for confirmation")

    _risk_usd   = abs(_entry_agg - _stop)
    _reward_usd = abs(_target - _entry_agg)

    # ── Display trade box ─────────────────────────────────────────────────────
    st.markdown(f"""
    <div style='background:{C["card2"]};border:1px solid {_dir_col}44;
                border-top:3px solid {_dir_col};border-radius:8px;
                padding:20px 24px;{"direction:rtl;text-align:right;" if use_ar else ""}'>
      <div style='font-size:14px;font-weight:700;color:{_dir_col};
                  letter-spacing:.05em;margin-bottom:16px;'>
        📊 {_dir_lbl}
      </div>
      <div style='display:grid;grid-template-columns:repeat(4,1fr);gap:10px;'>
        <div style='background:{C["card"]};border:1px solid {C["border"]};
                    border-radius:6px;padding:12px 14px;'>
          <div style='font-size:9px;color:{C["muted"]};letter-spacing:.15em;'>
            {"💰 " + L["sig_entry"]}</div>
          <div style='font-family:{C["font_m"]},monospace;font-size:18px;
                      font-weight:900;color:{C["gold_pale"]};margin-top:4px;'>
            ${_entry_agg:,.2f}</div>
          <div style='font-size:9px;color:{C["dim"]};margin-top:3px;'>
            {"محافظ: " if use_ar else "Conservative: "}${_entry_cons:,.2f}</div>
        </div>
        <div style='background:{C["card"]};border:1px solid {C["green"]}44;
                    border-radius:6px;padding:12px 14px;'>
          <div style='font-size:9px;color:{C["muted"]};letter-spacing:.15em;'>
            {"🎯 " + L["sig_target"]}</div>
          <div style='font-family:{C["font_m"]},monospace;font-size:18px;
                      font-weight:900;color:{C["green"]};margin-top:4px;'>
            ${_target:,.2f}</div>
          <div style='font-size:9px;color:{C["dim"]};margin-top:3px;'>
            +${_reward_usd:,.2f} (+{_reward_usd/_g*100:.2f}%)</div>
        </div>
        <div style='background:{C["card"]};border:1px solid {C["red"]}44;
                    border-radius:6px;padding:12px 14px;'>
          <div style='font-size:9px;color:{C["muted"]};letter-spacing:.15em;'>
            {"🛡️ " + L["sig_stop"]}</div>
          <div style='font-family:{C["font_m"]},monospace;font-size:18px;
                      font-weight:900;color:{C["red"]};margin-top:4px;'>
            ${_stop:,.2f}</div>
          <div style='font-size:9px;color:{C["dim"]};margin-top:3px;'>
            -${_risk_usd:,.2f} (-{_risk_usd/_g*100:.2f}%)</div>
        </div>
        <div style='background:{C["card"]};border:1px solid {C["gold"]}44;
                    border-radius:6px;padding:12px 14px;'>
          <div style='font-size:9px;color:{C["muted"]};letter-spacing:.15em;'>
            {"⚖️ " + L["sig_rr"]}</div>
          <div style='font-family:{C["font_m"]},monospace;font-size:18px;
                      font-weight:900;color:{C["gold_hi"]};margin-top:4px;'>
            1 : {_rr:.1f}</div>
          <div style='font-size:9px;color:{C["dim"]};margin-top:3px;'>
            {"ATR: " + str(round(_atr, 2))}</div>
        </div>
      </div>
      <div style='margin-top:12px;font-size:10px;color:{C["dim"]};
                  font-style:italic;{"text-align:right;" if use_ar else ""}'>
        ⚠️ {L["disclaimer"]}
      </div>
    </div>""", unsafe_allow_html=True)


elif nav == L["nav_game"]:

    use_ar = is_rtl()
    ph("🎮  " + L.get("nav_game","Mansa Game").replace("🎮  ",""),
       "🎮 العب وتحدَّ نفسك لجمع أكبر عدد من قطع الذهب!" if use_ar else "🎮 Play and challenge yourself to collect as many gold coins as you can!")

    # Score tracking
    if "game_score"     not in st.session_state: st.session_state["game_score"]     = 0
    if "game_highscore" not in st.session_state: st.session_state["game_highscore"] = 0

    # ── Score display — live JS feed from localStorage ──────────────────────────
    st.components.v1.html(f"""
    <style>
      .score-row {{
        display:flex; gap:12px; margin-bottom:8px;
      }}
      .score-box {{
        flex:1; background:{C["card"]}; border:1px solid {C["border"]};
        border-radius:4px; padding:14px 16px; text-align:center;
        position:relative; overflow:hidden;
      }}
      .score-box::after {{
        content:""; position:absolute; bottom:0; left:0; right:0; height:1px;
        background:linear-gradient(90deg,transparent,{C["gold"]}33,transparent);
      }}
      .score-lbl {{ font-family:'{C["font_h"]}',serif; font-size:10px;
                    letter-spacing:.2em; color:{C["muted"]}; text-transform:uppercase; }}
      .score-val {{ font-family:'{C["font_m"]}',monospace; font-size:32px;
                    font-weight:900; margin-top:4px; transition:all .3s; }}
      .session-val {{ color:{C["gold_hi"]}; }}
      .high-val     {{ color:{C["gold"]}; }}
    </style>
    <div class="score-row">
      <div class="score-box">
        <div class="score-lbl">{L["session_score"]}</div>
        <div class="score-val session-val" id="disp-session">0 🪙</div>
      </div>
      <div class="score-box">
        <div class="score-lbl">{L["high_score_lbl"]}</div>
        <div class="score-val high-val" id="disp-high">0 🏆</div>
      </div>
    </div>
    <script>
    // Load persisted high score from localStorage on page load
    (function() {{
      var hs = parseInt(localStorage.getItem('mansa_highscore') || '0');
      var el = document.getElementById('disp-high');
      if (el) el.textContent = hs + ' 🏆';
    }})();

    // Listen for score updates posted by the game iframe
    window.addEventListener('message', function(e) {{
      var d = e.data;
      if (!d || d.type !== 'mansa_score') return;
      var s  = d.score  || 0;
      var hs = d.hs     || s;

      var se = document.getElementById('disp-session');
      var he = document.getElementById('disp-high');
      if (se) se.textContent = s  + ' 🪙';
      if (he) he.textContent = hs + ' 🏆';

      // Animate new high score
      if (d.newHigh && he) {{
        he.style.color = '{C["gold_hi"]}';
        he.style.transform = 'scale(1.15)';
        setTimeout(function() {{
          he.style.transform = 'scale(1)';
          he.style.color = '{C["gold"]}';
        }}, 600);
      }}
    }});
    </script>
    """, height=110, scrolling=False)
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Game canvas ───────────────────────────────────────────────────────────
    theme_colors = {
    "bg":       C["bg"],
    "gold":     C["gold"],
    "gold_hi":  C["gold_hi"],
    "text":     C["text"],
    "green":    C["green"],
    "red":      C["red"],
    "card":     C["card"],
    "muted":    C["muted"],
    }

    game_html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<style>
  body {{ margin:0; background:{theme_colors['bg']}; display:flex; flex-direction:column;
      align-items:center; font-family:'Cairo',sans-serif; overflow:hidden; }}
  canvas {{ border:2px solid {theme_colors['gold']}55; border-radius:8px;
        box-shadow: 0 0 20px {theme_colors['gold']}33; }}
  #ui {{ color:{theme_colors['gold']}; font-size:14px; margin:8px 0 4px;
     letter-spacing:.1em; display:flex; gap:24px; align-items:center; }}
  #msg {{ color:{theme_colors['gold_hi']}; font-size:13px; min-height:20px; margin:4px 0; font-style:italic; }}
  button {{ background:linear-gradient(135deg,{theme_colors['card']},{theme_colors['gold']}66);
        color:{theme_colors['gold_hi']}; border:1px solid {theme_colors['gold']}88;
        padding:8px 22px; border-radius:4px; cursor:pointer; font-size:13px;
        font-family:'Cairo',sans-serif; letter-spacing:.1em; }}
  button:hover {{ opacity:.85; }}
</style>
</head>
<body>
<div id="ui">
  <span>🪙 Score: <b id="score">0</b></span>
  <span>❤️ Lives: <b id="lives">3</b></span>
  <span>📏 Speed: <b id="speed">1x</b></span>
  <span>🏆 Best: <b id="best">0</b></span>
</div>
<div id="msg">Press SPACE or tap to start ☽ مانسا</div>
<canvas id="c" width="800" height="260"></canvas>
<br>
<button id="startBtn" onclick="startGame()">▶ {L['start_game']}</button>

<script>
const canvas = document.getElementById('c');
const ctx    = canvas.getContext('2d');
const W = canvas.width, H = canvas.height;

// ── Named colour palette — all game colours in one place ─────────────────
const GAME_PALETTE = {{
  robe:       '#F5F0E8',        // Mansa's robe (ivory)
  grid:       '#C8BFA0',        // Ground grid lines
  body_dark:  '#3A2010',        // Character body shadow
  body_mid:   '#5C3317',        // Character body mid-tone
  body_light: '#7A4525',        // Character body highlight
  eyes:       '#1A0800',        // Character eyes
  platform:   '#3A3020',        // Platform fill
  plat_edge:  '#6A5830',        // Platform top edge
  obstacle:   '#3A2800',        // Obstacle base
  bg_bottom:  '#0D0820',        // Sky gradient bottom stop
  ground_bg:  '#1A0F05',        // Ground strip background
  overlay_bg: 'rgba(0,0,0,0.65)', // Game-over overlay
}};

// ── Game state ────────────────────────────────────────────────────────────
let state = 'idle'; // idle | running | dead
let score = 0, lives = 3, frame = 0, gameSpeed = 4;
let animId = null;

// ── Ground ────────────────────────────────────────────────────────────────
const GROUND = H - 50;

// ── Initialise best score display from localStorage ───────────────────────
(function() {{
  var el = document.getElementById('best');
  if (el) el.textContent = _sessionHigh;
}})();

// ── Mansa character ───────────────────────────────────────────────────────
const mansa = {{
  x: 80, y: GROUND, w: 38, h: 58,
  vy: 0, jumping: false, legPhase: 0,
  jump() {{
    if (!this.jumping) {{
      this.vy = -16;
      this.jumping = true;
    }}
  }},
  update() {{
    this.vy += 0.8;
    this.y += this.vy;
    if (this.y >= GROUND) {{
      this.y = GROUND; this.vy = 0; this.jumping = false;
    }}
    if (!this.jumping) this.legPhase += 0.25;
  }},
  draw() {{
    const x = this.x, y = this.y - this.h;
    // Robe (white/ivory)
    ctx.fillStyle = GAME_PALETTE.robe;
    ctx.beginPath();
    ctx.moveTo(x+4, y+18);
    ctx.lineTo(x-4, y+this.h);
    ctx.lineTo(x+this.w+4, y+this.h);
    ctx.lineTo(x+this.w-4, y+18);
    ctx.closePath();
    ctx.fill();
    // Robe shadow detail
    ctx.strokeStyle = GAME_PALETTE.grid; ctx.lineWidth=1;
    ctx.stroke();
    // Legs peeking out
    const legSwing = Math.sin(this.legPhase)*6;
    ctx.fillStyle = GAME_PALETTE.body_dark;
    // Left leg
    ctx.fillRect(x+8, y+this.h-14, 7, 14+legSwing);
    // Right leg
    ctx.fillRect(x+this.w-15, y+this.h-14, 7, 14-legSwing);
    // Head (dark skin)
    ctx.fillStyle = GAME_PALETTE.body_mid;
    ctx.beginPath();
    ctx.arc(x+this.w/2, y+10, 12, 0, Math.PI*2);
    ctx.fill();
    // Face highlight
    ctx.fillStyle = GAME_PALETTE.body_light;
    ctx.beginPath();
    ctx.arc(x+this.w/2-2, y+8, 6, 0, Math.PI*2);
    ctx.fill();
    // Eyes
    ctx.fillStyle='white';
    ctx.fillRect(x+this.w/2-6, y+6, 4, 4);
    ctx.fillRect(x+this.w/2+2, y+6, 4, 4);
    ctx.fillStyle = GAME_PALETTE.eyes;
    ctx.fillRect(x+this.w/2-5, y+7, 2, 2);
    ctx.fillRect(x+this.w/2+3, y+7, 2, 2);
    // Turban (gold + purple)
    ctx.fillStyle='{theme_colors['gold']}';
    ctx.beginPath();
    ctx.ellipse(x+this.w/2, y+2, 14, 8, 0, Math.PI, 2*Math.PI);
    ctx.fill();
    ctx.fillStyle='#7B2FBE';
    ctx.beginPath();
    ctx.arc(x+this.w/2, y-1, 10, Math.PI, 2*Math.PI);
    ctx.fill();
    // Crescent on turban
    ctx.fillStyle='{theme_colors['gold_hi']}';
    ctx.font='10px serif';
    ctx.fillText('☽', x+this.w/2-5, y+1);
  }}
}};

// ── Obstacles (rocks / dark stones) ──────────────────────────────────────
let obstacles = [];
function spawnObstacle() {{
  const h = 25 + Math.random()*25;
  obstacles.push({{ x: W+20, y: GROUND-h+5, w:22, h }});
}}

// ── Gold coins ────────────────────────────────────────────────────────────
let coins = [];
function spawnCoin() {{
  const yPos = GROUND - 30 - Math.random()*80;
  coins.push({{ x: W+20, y: yPos, r: 10, collected: false, flash: 0 }});
}}

// ── Particles ─────────────────────────────────────────────────────────────
let particles = [];
function burst(x, y) {{
  for(let i=0;i<10;i++) {{
    particles.push({{
      x, y,
      vx: (Math.random()-0.5)*6,
      vy: (Math.random()-0.7)*6,
      life: 30,
      color: Math.random()>0.5 ? '{theme_colors['gold_hi']}' : '{theme_colors['gold']}'
    }});
  }}
}}

// ── Background decorations ────────────────────────────────────────────────
const stars = Array.from({{length:40}}, ()=>
  ({{ x:Math.random()*W, y:Math.random()*(GROUND-60), r:Math.random()*1.5, speed:Math.random()*0.3+0.1 }}));

function drawBackground() {{
  // Sky gradient
  const grad = ctx.createLinearGradient(0,0,0,GROUND);
  grad.addColorStop(0, '{theme_colors['bg']}');
  grad.addColorStop(1, GAME_PALETTE.bg_bottom);
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, W, H);
  // Scrolling stars
  ctx.fillStyle = "rgba(245,200,66,0.4)";
  stars.forEach(s => {{
    s.x -= s.speed * gameSpeed * 0.3;
    if (s.x < 0) s.x = W;
    ctx.beginPath();
    ctx.arc(s.x, s.y, s.r, 0, Math.PI*2);
    ctx.fill();
  }});
  // Ground line
  ctx.fillStyle = '{theme_colors['gold']}33';
  ctx.fillRect(0, GROUND+5, W, 3);
  ctx.fillStyle = GAME_PALETTE.ground_bg;
  ctx.fillRect(0, GROUND+8, W, H-GROUND-8);
  // Ground texture dots
  ctx.fillStyle = '{theme_colors['gold']}22';
  for(let gx=frame%40;gx<W;gx+=40)
    ctx.fillRect(gx, GROUND+6, 20, 2);
}}

function drawObstacles() {{
  obstacles.forEach(o => {{
    // Stone shape
    ctx.fillStyle = GAME_PALETTE.platform;
    ctx.beginPath();
    ctx.roundRect(o.x, o.y, o.w, o.h, 4);
    ctx.fill();
    ctx.strokeStyle='{theme_colors['gold']}44'; ctx.lineWidth=1;
    ctx.stroke();
    // Glint
    ctx.fillStyle = GAME_PALETTE.plat_edge;
    ctx.fillRect(o.x+4, o.y+4, o.w-8, 4);
  }});
}}

function drawCoins() {{
  coins.forEach(coin => {{
    if(coin.collected) return;
    const pulse = Math.sin(frame*0.1)*2;
    // Outer glow
    ctx.shadowBlur=12; ctx.shadowColor='{theme_colors['gold']}';
    ctx.fillStyle='{theme_colors['gold_hi']}';
    ctx.beginPath();
    ctx.arc(coin.x, coin.y, coin.r+pulse*0.3, 0, Math.PI*2);
    ctx.fill();
    ctx.shadowBlur=0;
    // Inner
    ctx.fillStyle='{theme_colors['gold']}';
    ctx.beginPath();
    ctx.arc(coin.x, coin.y, coin.r-2+pulse*0.2, 0, Math.PI*2);
    ctx.fill();
    // Symbol
    ctx.fillStyle = GAME_PALETTE.obstacle;
    ctx.font='bold 9px serif';
    ctx.textAlign='center'; ctx.textBaseline='middle';
    ctx.fillText('$', coin.x, coin.y);
    ctx.textAlign='left'; ctx.textBaseline='alphabetic';
  }});
}}

function drawParticles() {{
  particles.forEach(p => {{
    ctx.globalAlpha = p.life/30;
    ctx.fillStyle = p.color;
    ctx.beginPath();
    ctx.arc(p.x, p.y, 3, 0, Math.PI*2);
    ctx.fill();
    p.x+=p.vx; p.y+=p.vy; p.vy+=0.2; p.life--;
  }});
  ctx.globalAlpha=1;
  particles = particles.filter(p=>p.life>0);
}}

// ── Collision ─────────────────────────────────────────────────────────────
function rectsOverlap(ax,ay,aw,ah, bx,by,bw,bh) {{
  return ax<bx+bw && ax+aw>bx && ay<by+bh && ay+ah>by;
}}

// ── Score tracking — localStorage + live parent display ─────────────────
var _sessionHigh = parseInt(localStorage.getItem('mansa_highscore') || '0');

function reportScore(s) {{
  try {{
    var isNew = (s > _sessionHigh && s > 0);
    if (isNew) {{
      _sessionHigh = s;
      localStorage.setItem('mansa_highscore', s);
    }}
    // Update "Best:" display inside canvas UI
    var bel = document.getElementById('best');
    if (bel) bel.textContent = _sessionHigh;
    // Post to parent frame → updates score cards above canvas
    window.parent.postMessage({{
      type:    'mansa_score',
      score:   s,
      hs:      _sessionHigh,
      newHigh: isNew
    }}, '*');
  }} catch(e) {{}}
}}

// ── Main loop ─────────────────────────────────────────────────────────────
let obstacleTick=0, coinTick=0, lastScore=0;
function loop() {{
  frame++;
  gameSpeed = 4 + Math.floor(score/20)*0.5;
  document.getElementById('speed').textContent = (gameSpeed/4).toFixed(1)+'x';

  // Spawn
  obstacleTick++;
  if(obstacleTick > Math.max(60, 110-score)) {{
    obstacleTick=0;
    if(Math.random()<0.7) spawnObstacle();
  }}
  coinTick++;
  if(coinTick > 45) {{
    coinTick=0;
    if(Math.random()<0.6) spawnCoin();
  }}

  // Move
  obstacles.forEach(o=>{{ o.x -= gameSpeed; }});
  coins.forEach(c=>{{ c.x -= gameSpeed*0.9; }});
  obstacles = obstacles.filter(o=>o.x>-50);
  coins     = coins.filter(c=>c.x>-30);
  mansa.update();

  // Coin collection
  coins.forEach(coin=>{{
    if(!coin.collected &&
       rectsOverlap(mansa.x, mansa.y-mansa.h, mansa.w, mansa.h,
               coin.x-coin.r, coin.y-coin.r, coin.r*2, coin.r*2)) {{
      coin.collected=true;
      score++;
      burst(coin.x, coin.y);
      document.getElementById('score').textContent=score;
      reportScore(score);
      // Milestone message
      if(score%10===0) {{
    document.getElementById('msg').textContent =
      '🪙 {L['amazing_10']} · Score: '+score;
      }} else {{
    document.getElementById('msg').textContent = '🪙 +1 · Total: '+score;
      }}
    }}
  }});

  // Obstacle collision
  const mx=mansa.x+4, my=mansa.y-mansa.h+8, mw=mansa.w-8, mh=mansa.h-8;
  for(let o of obstacles) {{
    if(rectsOverlap(mx,my,mw,mh, o.x+3,o.y,o.w-6,o.h)) {{
      lives--;
      document.getElementById('lives').textContent=lives;
      obstacles = obstacles.filter(ob=>ob!==o);
      burst(mansa.x+mansa.w/2, mansa.y-mansa.h/2);
      if(lives<=0) {{ die(); return; }}
      break;
    }}
  }}

  // Draw
  drawBackground();
  drawObstacles();
  drawCoins();
  mansa.draw();
  drawParticles();
  // Score overlay
  ctx.fillStyle='{theme_colors['gold']}66';
  ctx.font='bold 14px Cairo,serif';
  ctx.fillText('🪙 '+score, W-80, 24);

  animId = requestAnimationFrame(loop);
}}

function die() {{
  state='dead';
  cancelAnimationFrame(animId);
  // Update high score display via postMessage
  reportScore(score);
  drawBackground();
  mansa.draw();
  ctx.fillStyle = GAME_PALETTE.overlay_bg;
  ctx.fillRect(0,0,W,H);
  ctx.fillStyle='{theme_colors['gold_hi']}';
  ctx.font='bold 32px Cairo,serif';
  ctx.textAlign='center';
  ctx.fillText('{L['game_over']}', W/2, H/2-30);
  ctx.font='18px Cairo,serif';
  ctx.fillStyle='{theme_colors['text']}';
  ctx.fillText('{L['gold_collected']}: '+score+' 🪙', W/2, H/2+10);
  ctx.font='14px Cairo,serif';
  ctx.fillStyle='{theme_colors['gold']}';
  ctx.fillText('{L['high_score_lbl']}: '+_sessionHigh+' 🏆', W/2, H/2+35);
  ctx.textAlign='left';
  document.getElementById('startBtn').textContent='▶ {L['play_again']}';
}}

function startGame() {{
  cancelAnimationFrame(animId);
  state='running';
  score=0; lives=3; frame=0; gameSpeed=4;
  obstacles=[]; coins=[]; particles=[];
  mansa.y=GROUND; mansa.vy=0; mansa.jumping=false;
  document.getElementById('score').textContent=0;
  document.getElementById('lives').textContent=3;
  document.getElementById('speed').textContent='1x';
  document.getElementById('msg').textContent='';
  document.getElementById('best').textContent=_sessionHigh;
  document.getElementById('startBtn').textContent='🔄 {L['restart_lbl']}';
  loop();
}}

// Controls
document.addEventListener('keydown', e=>{{
  if(e.code==='Space'||e.code==='ArrowUp') {{
    e.preventDefault();
    if(state==='running') mansa.jump();
    else if(state==='idle'||state==='dead') startGame();
  }}
}});
canvas.addEventListener('click', ()=>{{
  if(state==='running') mansa.jump();
  else startGame();
}});
canvas.addEventListener('touchstart', e=>{{
  e.preventDefault();
  if(state==='running') mansa.jump();
  else startGame();
}}, {{passive:false}});
</script>
</body>
</html>"""

    st.components.v1.html(game_html, height=380, scrolling=False)

    # ── Instructions ──────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style='background:{C['card']};border:1px solid {C['gold']}44;border-radius:6px;
            padding:10px 16px;margin-top:8px;text-align:center;'>
      <div style='font-size:12px;color:{C['muted']};'>
    {L['game_instructions']}
      </div>
    </div>""", unsafe_allow_html=True)


    # ── Reset button — scores auto-tracked by JS/localStorage ────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    _, btn_col, _ = st.columns([3, 2, 3])
    with btn_col:
        if st.button("🗑️ " + ("إعادة تعيين الأرقام القياسية" if use_ar else "Reset All Scores"),
                     key="game_reset", use_container_width=True):
            st.session_state["game_score"]     = 0
            st.session_state["game_highscore"] = 0
            st.rerun()
    st.markdown(f"""
    <div style='text-align:center;font-size:10px;color:{C["dim"]};margin-top:6px;'>
      {"🔄 النقاط تُحدَّث تلقائياً · تُحفظ في المتصفح" if use_ar else
       "🔄 Scores update automatically · saved in your browser"}
    </div>""", unsafe_allow_html=True)

elif nav == L["nav_settings"]:
    use_ar = is_rtl()
    ph(L["settings"])
    col_l, col_r = st.columns(2, gap="large")

    with col_l:
        # Theme
        st.markdown("<div class='settings-card'>",unsafe_allow_html=True)
    st.markdown(f"<div class='settings-title'>🎨 {L['design_theme']}</div>",unsafe_allow_html=True)
    for tn,tc in THEMES.items():
        active_tag = " ← active" if tn==st.session_state["theme"] else ""
        st.markdown(f"""
        <div style='background:{tc['card2']};border:1px solid {tc['gold']}{"88" if tn==st.session_state['theme'] else "33"};
                        border-radius:3px;padding:9px 12px;margin-bottom:5px;'>
          <span style='font-size:16px;'>{tc['brand']}</span>
          <span style='font-family:{tc['font_h']},serif;font-size:10px;color:{tc['gold_pale']};margin-left:8px;'>
            {tn}{active_tag}
          </span>
        </div>""",unsafe_allow_html=True)
    t_sel = st.radio(L["design_theme"], list(THEMES.keys()),
        index=list(THEMES.keys()).index(st.session_state["theme"]),label_visibility="collapsed")
    if t_sel != st.session_state["theme"]:
        st.session_state["theme"]=t_sel; st.rerun()
    st.markdown("</div>",unsafe_allow_html=True)

    # Weight
    st.markdown("<div class='settings-card'>",unsafe_allow_html=True)
    st.markdown(f"<div class='settings-title'>⚖ {L['weight_unit']}</div>",unsafe_allow_html=True)
    u_sel=st.radio(L["weight_unit"],list(UNITS.keys()),
        index=list(UNITS.keys()).index(st.session_state["unit"]),label_visibility="collapsed")
    st.session_state["unit"]=u_sel
    st.markdown(f"""<div style='padding:8px 12px;background:{C['card']};border:1px solid {C['border']};border-radius:3px;margin-top:8px;'>
      <div class='stat-label'>24K/oz → {UNITS[u_sel]['symbol']}</div>
      <div style='font-family:{C['font_m']},monospace;font-size:16px;color:{C['gold_hi']};'>
        ${g_ref*UNITS[u_sel]['factor']:,.5f}
      </div></div>""",unsafe_allow_html=True)
    st.markdown("</div>",unsafe_allow_html=True)

    # Purity
    st.markdown("<div class='settings-card'>",unsafe_allow_html=True)
    st.markdown(f"<div class='settings-title'>💎 {L['purity']}</div>",unsafe_allow_html=True)
    p_sel=st.radio(L["purity"],list(PURITIES.keys()),
        index=list(PURITIES.keys()).index(st.session_state["purity"]),label_visibility="collapsed")
    st.session_state["purity"]=p_sel
    pc=PURITIES[p_sel]
    st.markdown(f"""<div style='padding:8px 12px;background:{C['card']};border:1px solid {C['border']};border-radius:3px;margin-top:8px;'>
      <div class='stat-label'>{pc['label']} ({pc['fine']}‰)</div>
      <div style='font-family:{C['font_m']},monospace;font-size:16px;color:{C['gold_hi']};'>
        ${g_ref*pc['mult']:,.2f} / oz t
      </div></div>""",unsafe_allow_html=True)
    st.markdown("</div>",unsafe_allow_html=True)

    # Display
    st.markdown("<div class='settings-card'>",unsafe_allow_html=True)
    st.markdown(f"<div class='settings-title'>🖥 {L['display_prefs']}</div>",unsafe_allow_html=True)
    st.session_state["show_purity_table"]=st.checkbox(L["show_purity_table"],value=st.session_state.get("show_purity_table",True))
    st.session_state["auto_refresh"]=st.checkbox(L["auto_refresh"],value=st.session_state.get("auto_refresh",False))
    per_opts=["1mo","3mo","6mo","1y","2y","5y"]
    st.session_state["period"]=st.selectbox(L["default_period"],per_opts,
        index=per_opts.index(st.session_state.get("period","1y")))
    st.markdown("</div>",unsafe_allow_html=True)

    # Stocks
    st.markdown("<div class='settings-card'>",unsafe_allow_html=True)
    st.markdown(f"<div class='settings-title'>📊 {L['active_stocks']}</div>",unsafe_allow_html=True)
    ns=list(st.session_state.get("active_stocks",[]))
    for sn in STOCK_OPTIONS:
        v=st.checkbox(f"{sn} ({STOCK_OPTIONS[sn]})",value=(sn in ns),key=f"stk_{sn}")
        if v and sn not in ns: ns.append(sn)
        if not v and sn in ns: ns.remove(sn)
    st.session_state["active_stocks"]=ns
    st.markdown("</div>",unsafe_allow_html=True)

    with col_r:
    # Primary market
        st.markdown("<div class='settings-card'>",unsafe_allow_html=True)
    st.markdown(f"<div class='settings-title'>🏠 {L['primary_market']}</div>",unsafe_allow_html=True)
    pm_sel=st.selectbox(L["primary_market"],list(MARKETS.keys()),
        index=list(MARKETS.keys()).index(st.session_state["primary_mkt"]),label_visibility="collapsed")
    st.session_state["primary_mkt"]=pm_sel
    pm2=MARKETS[pm_sel]; pm2p,_=mkt_price(g_ref,pm2,st.session_state["purity"])
    st.markdown(f"""<div style='padding:8px 12px;background:{C['card']};border:1px solid {C['gold']}55;border-radius:3px;margin-top:8px;'>
      <div class='stat-label'>{pm2['flag']} {pm_sel}</div>
      <div style='font-family:{C['font_m']},monospace;font-size:16px;color:{C['gold_hi']};'>
        {pm2p:,.3f} {pm2['currency']} / {pm2['unit_label']}
      </div></div>""",unsafe_allow_html=True)
    st.markdown("</div>",unsafe_allow_html=True)

    # Active markets
    st.markdown("<div class='settings-card'>",unsafe_allow_html=True)
    st.markdown(f"<div class='settings-title'>✅ {L['active_markets']}</div>",unsafe_allow_html=True)
    arab_flags={"🇯🇴","🇸🇦","🇦🇪","🇪🇬","🇰🇼","🇶🇦","🇧🇭","🇴🇲","🇱🇧","🇮🇶","🇹🇷"}
    arab_keys=[k for k in MARKETS if MARKETS[k]["flag"] in arab_flags]
    global_keys=[k for k in MARKETS if k not in arab_keys]
    na=list(st.session_state["active_mkts"])
    st.markdown(f"<div style='font-size:9px;letter-spacing:.2em;color:{C['gold']};margin-bottom:5px;'>{L['arab_markets'].upper()}</div>",unsafe_allow_html=True)
    for mk in arab_keys:
        mi=MARKETS[mk]; v=st.checkbox(f"{mi['flag']} {mk}",value=(mk in na),key=f"chk_{mk}")
        if v and mk not in na: na.append(mk)
        if not v and mk in na: na.remove(mk)
    st.markdown(f"<div style='font-size:9px;letter-spacing:.2em;color:{C['blue']};margin:8px 0 5px;'>{L['intl_markets'].upper()}</div>",unsafe_allow_html=True)
    for mk in global_keys:
        mi=MARKETS[mk]; v=st.checkbox(f"{mi['flag']} {mk}",value=(mk in na),key=f"chk_{mk}")
        if v and mk not in na: na.append(mk)
        if not v and mk in na: na.remove(mk)
    st.session_state["active_mkts"]=na
    st.markdown("</div>",unsafe_allow_html=True)

    # Conversion table
    st.markdown(f"<div class='section-label'>{L['conversion_ref']} (24K USD)</div>",unsafe_allow_html=True)
    rows=[]
    for un,uc in UNITS.items():
        row={("الوحدة" if use_ar else "Unit"):un,("الرمز" if use_ar else "Symbol"):uc["symbol"]}
    for pn,pc2 in PURITIES.items():
        row[pc2["label"]]=f"${g_ref*uc['factor']*pc2['mult']:,.5f}"
    rows.append(row)
    st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)

    # ── Feature Information Cards ─────────────────────────────────────────────
    st.markdown("<br>",unsafe_allow_html=True)
    use_ar = is_rtl()
    st.markdown(f"<div class='section-label'>📚 {L['feat_info']}</div>",unsafe_allow_html=True)

    FEATURE_INFO = [
    {
        "name": "Gold Price / سعر الذهب",
        "icon": "🥇",
        "value": f"${g_ref:,.2f}",
        "unit": "USD/oz",
        "signal_fn": lambda: ("BUY" if live["gold"]["pct"] > 0.3 else ("SELL" if live["gold"]["pct"] < -0.3 else "HOLD"),
                                   live["gold"]["pct"]),
        "history_en": "Gold has been valued for over 5,000 years. Ancient Egyptians, Romans and Islamic civilisations used it as currency and a store of wealth. The gold standard dominated global finance until 1971.",
        "history_ar": "قُدِّر الذهب منذ أكثر من 5000 عام. استخدمه المصريون والرومان والحضارات الإسلامية عملةً ومخزنًا للثروة. سيطر معيار الذهب على المالية العالمية حتى عام 1971.",
        "relation_en": "Gold is the base asset of this platform. All other indicators are measured relative to gold to assess overvaluation or undervaluation.",
        "relation_ar": "الذهب هو الأصل الأساسي لهذه المنصة. تُقاس جميع المؤشرات الأخرى بالنسبة إليه لتقييم ما إذا كان مُبالغًا في تقييمه أو مُقلَّلًا منه.",
    },
    {
        "name": "S&P 500 / مؤشر S&P 500",
        "icon": "📈",
        "value": f"{live['spx']['price']:,.0f}",
        "unit": "pts",
        "signal_fn": lambda: ("SELL" if live["spx"]["pct"] > 1 else ("BUY" if live["spx"]["pct"] < -1 else "HOLD"),
                                   live["spx"]["pct"]),
        "history_en": "The S&P 500 tracks 500 large US companies since 1957. It is the world's most followed stock index and a proxy for global investor risk appetite.",
        "history_ar": "يتتبع مؤشر S&P 500 أداء 500 شركة أمريكية كبرى منذ عام 1957. وهو المؤشر الأكثر متابعةً في العالم وبارومتر شهية المستثمرين للمخاطرة.",
        "relation_en": "When S&P rises strongly, investors prefer stocks over gold (risk-on). When S&P falls sharply, gold benefits as a safe haven (risk-off). Inverse correlation ~60%.",
        "relation_ar": "عندما يرتفع S&P بقوة، يفضّل المستثمرون الأسهم على الذهب. وعند انخفاضه الحاد، يستفيد الذهب كملاذ آمن. الارتباط العكسي نحو 60%.",
    },
    {
        "name": "CPI / مؤشر أسعار المستهلك",
        "icon": "🧾",
        "value": "314.0",
        "unit": "index",
        "signal_fn": lambda: ("BUY", 0.4),
        "history_en": "The Consumer Price Index measures inflation since 1913. Created by the US Bureau of Labor Statistics, it tracks the price of a basket of everyday goods and services.",
        "history_ar": "يقيس مؤشر أسعار المستهلك التضخم منذ عام 1913. أنشأه مكتب إحصاءات العمل الأمريكي لتتبع أسعار سلة من السلع والخدمات اليومية.",
        "relation_en": "Gold is the classic inflation hedge. When CPI rises above 3%, gold historically outperforms most asset classes. Rising CPI = bullish for gold.",
        "relation_ar": "الذهب هو التحوط الكلاسيكي ضد التضخم. عندما يتجاوز CPI 3٪، يتفوق الذهب تاريخيًا على معظم فئات الأصول. ارتفاع CPI = إشارة صعودية للذهب.",
    },
    {
        "name": "EFFR / معدل الفائدة الفيدرالي",
        "icon": "🏦",
        "value": "4.33%",
        "unit": "%",
        "signal_fn": lambda: ("SELL", -0.5),
        "history_en": "The Effective Federal Funds Rate is set by the US Federal Reserve since 1954. It is the overnight lending rate between US banks and the anchor of global interest rates.",
        "history_ar": "حدّد الاحتياطي الفيدرالي الأمريكي معدل الفائدة الفيدرالي الفعّال منذ عام 1954. وهو سعر الإقراض الليلي بين البنوك الأمريكية ومرساة أسعار الفائدة العالمية.",
        "relation_en": "Rising interest rates increase the opportunity cost of holding gold (which pays no yield). High EFFR = bearish for gold. Historically, gold rallies when the Fed pivots to cuts.",
        "relation_ar": "رفع أسعار الفائدة يزيد من تكلفة الفرصة البديلة لحيازة الذهب. EFFR مرتفع = ضغط هبوطي على الذهب. تاريخيًا، يرتفع الذهب عند تحوّل الفيدرالي نحو خفض الفائدة.",
    },
    {
        "name": "USD Index (DXY) / مؤشر الدولار",
        "icon": "💵",
        "value": f"{live['dxy']['price']:,.2f}",
        "unit": "index",
        "signal_fn": lambda: ("BUY" if live["dxy"]["pct"] < -0.3 else ("SELL" if live["dxy"]["pct"] > 0.3 else "HOLD"),
                                   -live["dxy"]["pct"]),
        "history_en": "The DXY index was created in 1973 after the Bretton Woods collapse. It measures the US dollar against a basket of 6 currencies (EUR, JPY, GBP, CAD, SEK, CHF).",
        "history_ar": "أُنشئ مؤشر DXY عام 1973 إثر انهيار اتفاقية بريتون وودز. يقيس قيمة الدولار الأمريكي مقابل سلة من ست عملات رئيسية.",
        "relation_en": "Gold is priced in USD globally — so a weaker dollar makes gold cheaper for foreign buyers, driving demand up. DXY and Gold have ~80% inverse correlation.",
        "relation_ar": "يُسعَّر الذهب بالدولار عالميًا، لذا ضعف الدولار يجعله أرخص للمشترين الأجانب مما يرفع الطلب. الارتباط العكسي بين DXY والذهب نحو 80%.",
    },
    {
        "name": "VIX / مؤشر الخوف",
        "icon": "😨",
        "value": f"{live['vix']['price']:,.2f}",
        "unit": "index",
        "signal_fn": lambda: ("BUY" if live["vix"]["price"] > 20 else "HOLD",
                                   (live["vix"]["price"]-15)/15*100),
        "history_en": "The CBOE Volatility Index was launched in 1993. Often called the 'Fear Index', it measures expected 30-day volatility of the S&P 500 using options prices.",
        "history_ar": "أُطلق مؤشر التقلب CBOE عام 1993. يُعرف بـ'مؤشر الخوف' ويقيس التقلب المتوقع للسوق خلال 30 يومًا بناءً على أسعار الخيارات.",
        "relation_en": "When VIX spikes above 25-30, fear dominates markets and investors flee to gold as a safe haven. VIX > 30 historically marks excellent gold entry points.",
        "relation_ar": "عندما يتجاوز VIX 25-30، يسود الخوف الأسواق ويلجأ المستثمرون إلى الذهب كملاذ آمن. VIX فوق 30 يمثّل تاريخيًا نقاط دخول ممتازة للذهب.",
    },
    {
        "name": "Silver / الفضة",
        "icon": "🪙",
        "value": f"${live['silver']['price']:,.2f}",
        "unit": "USD/oz",
        "signal_fn": lambda: ("BUY" if live["silver"]["pct"] > 0.3 else ("SELL" if live["silver"]["pct"] < -0.3 else "HOLD"),
                                   live["silver"]["pct"]),
        "history_en": "Silver has been used as currency for over 4,000 years, preceding gold in some ancient economies. The Gold-Silver ratio has been tracked since ancient Rome.",
        "history_ar": "استُخدمت الفضة عملةً منذ أكثر من 4000 عام، وسبقت الذهب في بعض الاقتصادات القديمة. تُتابَع نسبة الذهب إلى الفضة منذ روما القديمة.",
        "relation_en": "Gold/Silver ratio above 80 means gold is relatively expensive vs silver. When both metals rise together, it signals broad precious metals demand — very bullish for gold.",
        "relation_ar": "نسبة ذهب/فضة فوق 80 تعني أن الذهب باهظ نسبيًا. عندما يرتفع المعدنان معًا، يُشير ذلك إلى طلب واسع على المعادن الثمينة — إشارة صعودية قوية جدًا للذهب.",
    },
    {
        "name": "Oil / النفط الخام",
        "icon": "🛢️",
        "value": f"${live['oil']['price']:,.2f}",
        "unit": "USD/bbl",
        "signal_fn": lambda: ("BUY" if live["oil"]["pct"] > 1 else ("HOLD" if live["oil"]["pct"] > -1 else "SELL"),
                                   live["oil"]["pct"]*0.5),
        "history_en": "Modern crude oil markets began in 1859 in Pennsylvania. Oil became the world's most traded commodity after WWII and is priced in USD, giving it deep links to gold.",
        "history_ar": "بدأت أسواق النفط الخام الحديثة عام 1859 في بنسلفانيا. أصبح النفط أكثر السلع تداولًا بعد الحرب العالمية الثانية ويُسعَّر بالدولار مما يربطه بالذهب.",
        "relation_en": "Rising oil signals inflation risk, which is bullish for gold. Both commodities rise during geopolitical crises. Gold/Oil ratio above 25 suggests gold is expensive vs oil.",
        "relation_ar": "ارتفاع النفط يُشير إلى مخاطر التضخم وهو إشارة صعودية للذهب. ترتفع السلعتان معًا في أوقات الأزمات الجيوسياسية. نسبة ذهب/نفط فوق 25 تعني أن الذهب مُكلف نسبيًا.",
    },
    ]

    for i in range(0, len(FEATURE_INFO), 4):
        row_feats = FEATURE_INFO[i:i+4]
    f_cols = st.columns(4)
    for fc, feat in zip(f_cols, row_feats):
        signal, pct = feat["signal_fn"]()
        sig_col = C["green"] if signal=="BUY" else (C["red"] if signal=="SELL" else C["muted"])
        sig_icon= "▲" if signal=="BUY" else ("▼" if signal=="SELL" else "◆")
        hist_txt = feat["history_ar"] if use_ar else feat["history_en"]
        rel_txt  = feat["relation_ar"] if use_ar else feat["relation_en"]
        with fc:
            with st.expander(f"{feat['icon']} {feat['name'].split('/')[0].strip()}"):
                st.markdown(f"""
                <div style='text-align:center;padding:8px 0;'>
                  <div style='font-family:{C['font_m']},monospace;font-size:22px;color:{C['gold_hi']};'>{feat['value']}</div>
                  <div style='font-size:10px;color:{C['muted']};'>{feat['unit']}</div>
                  <div style='font-family:{C['font_h']},serif;font-size:13px;font-weight:700;
                                  color:{sig_col};margin-top:6px;'>{sig_icon} {signal}</div>
                </div>
                <hr style='border-color:{C['border']};margin:6px 0;'>
                <div style='font-size:12px;color:{C['muted']};font-weight:700;margin-bottom:4px;'>
                  {L['history_lbl']}
                </div>
                <div style='font-size:12px;color:{C['text']};line-height:1.6;margin-bottom:8px;'>
                  {hist_txt}
                </div>
                <div style='font-size:12px;color:{C['muted']};font-weight:700;margin-bottom:4px;'>
                  {L['relation_gold']}
                </div>
                <div style='font-size:12px;color:{C['text']};line-height:1.6;'>
                  {rel_txt}
                </div>""", unsafe_allow_html=True)
    st.markdown("<br>",unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: DEMO TRADING 🎯
# ═══════════════════════════════════════════════════════════════════════════════
elif nav == L["nav_demo"]:
    use_ar = is_rtl()
    ph("🎯  " + L["demo_title"], L["demo_sub"])

    # ── Ensure demo state keys exist ─────────────────────────────────────────
    for _dk, _dv in [("demo_balance", 10000.0), ("demo_holdings_g", 0.0),
                      ("demo_trades", []), ("demo_total_bought", 0.0)]:
        if _dk not in st.session_state:
            st.session_state[_dk] = _dv

    spot    = g_ref                           # live USD/oz
    gram_px = spot / 31.1035                  # USD per gram
    bal     = st.session_state["demo_balance"]
    held_g  = st.session_state["demo_holdings_g"]
    held_v  = held_g * gram_px
    total_invested = st.session_state["demo_total_bought"]
    pnl     = held_v - total_invested
    pnl_pct = (pnl / total_invested * 100) if total_invested > 0 else 0.0
    equity  = bal + held_v
    pnl_col = C["green"] if pnl >= 0 else C["red"]

    # ── Header stats ─────────────────────────────────────────────────────────
    d1, d2, d3, d4 = st.columns(4)
    stats_demo = [
    (d1, "💵 " + L["demo_balance"],      f"${bal:,.2f}",        "USD"),
    (d2, "🥇 " + L["demo_holdings"],     f"{held_g:.4f}g",      f"≈ ${held_v:,.2f}"),
    (d3, "📊 " + L["demo_pnl"],          f"${pnl:+,.2f}",       f"{pnl_pct:+.2f}%"),
    (d4, "💼 Equity",                    f"${equity:,.2f}",     "Total"),
    ]
    for col, lbl, val, sub in stats_demo:
        vc = pnl_col if "P&L" in lbl or L["demo_pnl"] in lbl else C["gold_pale"]
    with col:
        st.markdown(f"""
        <div class='stat-card'>
          <div class='stat-label'>{lbl}</div>
          <div class='stat-value' style='color:{vc};'>{val}</div>
          <div style='font-size:11px;color:{C["dim"]};margin-top:2px;'>{sub}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.info(L["demo_note"])
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Buy / Sell ────────────────────────────────────────────────────────────
    tr1, tr2 = st.columns(2, gap="large")
    with tr1:
        st.markdown(f"<div class='section-label'>🟢 {L['demo_buy']}</div>",
                unsafe_allow_html=True)
    buy_qty = st.number_input(
        L["demo_qty"], min_value=0.1, max_value=10000.0,
        value=10.0, step=1.0, key="demo_buy_qty", format="%.2f"
    )
    buy_cost = buy_qty * gram_px
    st.markdown(f"""
    <div style='background:{C["card2"]};border:1px solid {C["green"]}44;
                border-radius:6px;padding:12px 16px;margin:8px 0;'>
      <div style='font-size:12px;color:{C["muted"]};'>{L["demo_qty"]}</div>
      <div style='font-family:{C["font_m"]},monospace;font-size:20px;
                  color:{C["green"]};font-weight:700;'>{buy_qty:.2f}g</div>
      <div style='font-size:12px;color:{C["dim"]};margin-top:4px;'>
        Cost: ${buy_cost:,.2f} · Balance after: ${bal-buy_cost:,.2f}
      </div>
    </div>""", unsafe_allow_html=True)
    if st.button(f"🟢 {L['demo_buy']}", type="primary",
                 use_container_width=True, key="demo_do_buy"):
        if buy_cost > bal:
            st.error("❌ " + ("رصيد غير كافٍ" if use_ar else "Insufficient balance"))
        else:
            st.session_state["demo_balance"] -= buy_cost
            st.session_state["demo_holdings_g"] += buy_qty
            st.session_state["demo_total_bought"] += buy_cost
            st.session_state["demo_trades"].append({
                "id": len(st.session_state["demo_trades"]),
                "type": "BUY", "qty_g": buy_qty,
                "price_usd": gram_px, "ts": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                "pnl": 0.0,
            })
            _sb_save()
            st.success(f"✅ {L['demo_buy']}: {buy_qty:.2f}g @ ${gram_px:,.4f}/g")
            st.rerun()

    with tr2:
        st.markdown(f"<div class='section-label'>🔴 {L['demo_sell']}</div>",
                unsafe_allow_html=True)
    max_sell = max(0.01, held_g)
    sell_qty = st.number_input(
        L["demo_qty"], min_value=0.0, max_value=float(max_sell),
        value=min(10.0, float(max_sell)), step=1.0,
        key="demo_sell_qty", format="%.2f"
    )
    sell_val  = sell_qty * gram_px
    cost_basis_per_g = (total_invested / held_g) if held_g > 0 else gram_px
    sell_pnl  = (gram_px - cost_basis_per_g) * sell_qty
    sell_col  = C["green"] if sell_pnl >= 0 else C["red"]
    st.markdown(f"""
    <div style='background:{C["card2"]};border:1px solid {C["red"]}44;
                border-radius:6px;padding:12px 16px;margin:8px 0;'>
      <div style='font-size:12px;color:{C["muted"]};'>{L["demo_qty"]}</div>
      <div style='font-family:{C["font_m"]},monospace;font-size:20px;
                  color:{C["red"]};font-weight:700;'>{sell_qty:.2f}g</div>
      <div style='font-size:12px;color:{sell_col};margin-top:4px;'>
        Value: ${sell_val:,.2f} · P&L: ${sell_pnl:+,.2f}
      </div>
    </div>""", unsafe_allow_html=True)
    sell_disabled = held_g <= 0
    if st.button(f"🔴 {L['demo_sell']}", type="secondary",
                 use_container_width=True, key="demo_do_sell",
                 disabled=sell_disabled):
        if sell_qty > held_g:
            st.error("❌ " + ("لا يوجد رصيد ذهب كافٍ" if use_ar else "Not enough gold"))
        else:
            realized_cost = cost_basis_per_g * sell_qty
            st.session_state["demo_balance"] += sell_val
            st.session_state["demo_holdings_g"] -= sell_qty
            st.session_state["demo_total_bought"] -= realized_cost
            st.session_state["demo_trades"].append({
                "id": len(st.session_state["demo_trades"]),
                "type": "SELL", "qty_g": sell_qty,
                "price_usd": gram_px, "ts": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                "pnl": sell_pnl,
            })
            _sb_save()
            st.success(f"✅ {L['demo_sell']}: {sell_qty:.2f}g · P&L ${sell_pnl:+,.2f}")
            st.rerun()

    # ── Trade log ─────────────────────────────────────────────────────────────
    trades_d = list(reversed(st.session_state.get("demo_trades", [])[-20:]))
    if trades_d:
        st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"<div class='section-label'>{L['demo_trades']}</div>",
                unsafe_allow_html=True)
    for tr in trades_d:
        tc = C["green"] if tr["type"] == "BUY" else C["red"]
        icon = "▲" if tr["type"] == "BUY" else "▼"
        pnl_s = f" · P&L ${tr['pnl']:+,.2f}" if tr["type"] == "SELL" else ""
        st.markdown(f"""
        <div style='background:{C["card"]};border:1px solid {tc}33;
                        border-left:3px solid {tc};border-radius:4px;
                        padding:8px 14px;margin-bottom:4px;
                        font-family:{C["font_m"]},monospace;font-size:12px;
                        display:flex;justify-content:space-between;'>
          <span style='color:{tc};font-weight:700;'>{icon} {tr["type"]} {tr["qty_g"]:.2f}g</span>
          <span style='color:{C["muted"]};'>@ ${tr["price_usd"]:,.4f}/g{pnl_s}</span>
          <span style='color:{C["dim"]};'>{tr["ts"]}</span>
        </div>""", unsafe_allow_html=True)

    # ── Reset ─────────────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button(f"🔄 {L['demo_reset']}", key="demo_reset_btn"):
        st.session_state["demo_balance"]      = 10000.0
    st.session_state["demo_holdings_g"]   = 0.0
    st.session_state["demo_trades"]       = []
    st.session_state["demo_total_bought"] = 0.0
    _sb_save()
    st.success(L["demo_cleared"])
    st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: GOLD SAVINGS PLAN 🪙
# ═══════════════════════════════════════════════════════════════════════════════
elif nav == L["nav_savings"]:
    use_ar = is_rtl()
    ph("🪙  " + L["savings_title"], L["savings_sub"])
    st.info(L["savings_note"])
    st.markdown("<br>", unsafe_allow_html=True)

    if "savings_plans" not in st.session_state:
        st.session_state["savings_plans"] = []

    # ── Add new plan ──────────────────────────────────────────────────────────
    with st.expander("➕ " + L["savings_add"], expanded=not st.session_state["savings_plans"]):
        sp1, sp2, sp3, sp4 = st.columns(4)
    with sp1:
        sp_monthly = st.number_input(
            L["savings_monthly"], min_value=10.0, max_value=100000.0,
            value=100.0, step=10.0, key="sp_monthly"
        )
    with sp2:
        sp_curr = st.selectbox(
            L["savings_curr"],
            ["USD","JOD","SAR","AED","EGP","KWD","QAR","BHD","GBP","EUR","TRY"],
            key="sp_curr"
        )
    with sp3:
        sp_since = st.date_input(
            L["savings_since"], value=datetime.date.today().replace(day=1),
            key="sp_since"
        )
    with sp4:
        sp_label = st.text_input(
            "🏷️ " + ("الاسم" if use_ar else "Label"),
            placeholder=("مدخرات ذهبية" if use_ar else "My gold savings"),
            key="sp_label"
        )
    if st.button("✅ " + L["savings_add"], type="primary", key="sp_add_btn"):
        st.session_state["savings_plans"].append({
            "id":      len(st.session_state["savings_plans"]),
            "monthly": sp_monthly,
            "currency": sp_curr,
            "start":   str(sp_since),
            "label":   sp_label or ("مدخرات ذهبية" if use_ar else "Gold Savings"),
        })
        _sb_save()
        st.rerun()

    # ── FX rates for conversion ───────────────────────────────────────────────
    _FX_TO_USD = {
    "USD":1.0,"JOD":1/0.709,"SAR":1/3.75,"AED":1/3.6725,"EGP":1/50.9,
    "KWD":1/0.307,"QAR":1/3.64,"BHD":1/0.376,"GBP":1.27,"EUR":1.08,"TRY":1/38.0,
    }

    plans = st.session_state.get("savings_plans", [])
    if plans:
        for plan in plans:
            fx       = _FX_TO_USD.get(plan["currency"], 1.0)
        monthly_usd = plan["monthly"] * fx
        start_dt = datetime.date.fromisoformat(plan["start"])
        today_d  = datetime.date.today()
        months   = max(1, (today_d.year - start_dt.year)*12 + today_d.month - start_dt.month)
        total_saved_usd = monthly_usd * months
        # Average gold price over period ≈ use current as proxy (conservative)
        gram_px  = g_ref / 31.1035
        grams_acc = total_saved_usd / gram_px
        current_val = grams_acc * gram_px
        gain_vs_cash = current_val - total_saved_usd   # simplified: no interest on cash
        gain_col = C["green"] if gain_vs_cash >= 0 else C["red"]
        gain_pct = (gain_vs_cash / total_saved_usd * 100) if total_saved_usd > 0 else 0.0

        with st.container():
            st.markdown(f"<div class='section-label'>🪙 {plan['label']}</div>",
                            unsafe_allow_html=True)
            sc1,sc2,sc3,sc4,sc5 = st.columns(5)
            for scol, slbl, sval, ssub in [
                (sc1, L["savings_monthly"],  f"{plan['monthly']:,.0f} {plan['currency']}", f"${monthly_usd:,.2f}"),
                (sc2, L["savings_total"],     f"${total_saved_usd:,.2f}", f"{months} months"),
                (sc3, L["savings_gold"],      f"{grams_acc:.3f}g", f"{grams_acc/31.1035:.4f} oz"),
                (sc4, L["savings_value"],     f"${current_val:,.2f}", f"@ ${gram_px:,.2f}/g"),
                (sc5, L["savings_gain"],      f"${gain_vs_cash:+,.2f}", f"{gain_pct:+.1f}%"),
            ]:
                vc = gain_col if slbl == L["savings_gain"] else C["gold_pale"]
                with scol:
                        st.markdown(f"""
                        <div class='stat-card'>
                          <div class='stat-label'>{slbl}</div>
                          <div class='stat-value' style='color:{vc};font-size:15px;'>{sval}</div>
                          <div style='font-size:10px;color:{C["dim"]};margin-top:2px;'>{ssub}</div>
                        </div>""", unsafe_allow_html=True)

            # Growth chart: monthly cumulative gold value vs cash
            import plotly.graph_objects as _pgo2
            months_range  = list(range(1, months+1))
            cash_vals     = [monthly_usd * m for m in months_range]
            gold_vals     = [monthly_usd * m for m in months_range]  # simplified same cost basis
            fig_sv = _pgo2.Figure()
            fig_sv.add_trace(_pgo2.Scatter(
                x=months_range, y=gold_vals,
                name=("الذهب" if use_ar else "Gold"),
                line=dict(color=C["gold"], width=2),
                fill="tozeroy", fillcolor=rgba(C["gold"], 0.09)
            ))
            fig_sv.add_trace(_pgo2.Scatter(
                x=months_range, y=cash_vals,
                name=("نقد" if use_ar else "Cash"),
                line=dict(color=C["muted"], width=1.5, dash="dash")
            ))
            fig_sv.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color=C["text"], size=10), height=180,
                margin=dict(l=0,r=0,t=10,b=0),
                xaxis=dict(gridcolor=C["border2"], title=("شهر" if use_ar else "Month")),
                yaxis=dict(gridcolor=C["border2"], tickprefix="$"),
                legend=dict(orientation="h", y=-0.3),
                showlegend=True,
            )
            st.plotly_chart(fig_sv, use_container_width=True, config={"displayModeBar":False})

            if st.button("🗑️ " + L["savings_delete"], key=f"sp_del_{plan['id']}"):
                st.session_state["savings_plans"] = [
                        p for p in st.session_state["savings_plans"] if p["id"] != plan["id"]
                ]
                _sb_save()
                st.rerun()
            
    # ── Embed Widget ──────────────────────────────────────────────────────────
    st.markdown("<div class='mansa-divider'></div>", unsafe_allow_html=True)
    with st.expander("🔌 " + ("كود التضمين — أداة المطورين" if use_ar else "Embed Widget — For Developers"), expanded=False):
        w_curr_s = st.selectbox(
        L["widget_curr2"],
        ["USD","JOD","SAR","AED","EGP","KWD","GBP","EUR"],
        key="w_curr_settings"
    )
    _FX_EMB = {"USD":1.0,"JOD":0.709,"SAR":3.75,"AED":3.6725,
               "EGP":50.9,"KWD":0.307,"GBP":0.787,"EUR":0.926}
    w_p = g_ref / 31.1035 * _FX_EMB.get(w_curr_s, 1.0)
    w_pct_e = live["gold"]["pct"]
    w_col_e = C["green"] if w_pct_e >= 0 else C["red"]
    w_arr_e = "▲" if w_pct_e >= 0 else "▼"
    st.markdown(f"""
    <div style='background:{C["card"]};border:1px solid {C["gold"]}44;border-radius:6px;
                padding:14px 18px;margin:8px 0;text-align:center;'>
      <div style='font-size:9px;letter-spacing:.2em;color:{C["muted"]};'>☽ MANSA · GOLD PRICE</div>
      <div style='font-family:{C["font_m"]},monospace;font-size:26px;
                  font-weight:900;color:{C["gold_hi"]};margin:6px 0;'>
        {w_p:,.3f} {w_curr_s}</div>
      <div style='font-size:11px;color:{w_col_e};'>{w_arr_e} {abs(w_pct_e):.2f}%</div>
      <div style='font-size:8px;color:{C["muted"]};margin-top:4px;letter-spacing:.1em;'>LIVE DATA · mansa.gold</div>
    </div>""", unsafe_allow_html=True)
    fx_val = _FX_EMB.get(w_curr_s, 1.0)
    _fx_s = f"{fx_val:.4f}"
    _curr_s = w_curr_s
    embed_code = (
        "<!-- MANSA Gold Widget -->\n"
        '<div id="mg" style="font-family:Georgia;background:#050408;color:#F5C842;'
        'border:1px solid #3A2800;border-radius:8px;padding:14px 18px;'
        'display:inline-block;min-width:180px;"></div>\n'
        "<script>(function(){\n"
        "  var el=document.getElementById('mg');\n"
        "  function u(){\n"
        "    fetch('https://query1.finance.yahoo.com/v8/finance/chart/GC=F?interval=1m&range=1d')\n"
        "      .then(function(r){return r.json();})\n"
        "      .then(function(d){\n"
        f"        var p=d.chart.result[0].meta.regularMarketPrice||0;\n"
        f"        var c=d.chart.result[0].meta.previousClose||p;\n"
        f"        var local=(p/31.1035*{_fx_s}).toFixed(3);\n"
        "        var chg=((p-c)/c*100).toFixed(2);\n"
        "        var col=p>=c?'#52D98A':'#FF5555';\n"
        "        var arr=p>=c?'UP':'DN';\n"
        f"        el.innerHTML='<b>MANSA GOLD</b><br>'+local+' {_curr_s}<br>'+arr+' '+chg+'%';\n"
        "      }).catch(function(){});\n"
        "  }\n"
        "  u(); setInterval(u,60000);\n"
        "})();</script>"
    )
    st.code(embed_code, language="html")
    st.download_button(
        label="📥 " + L["widget_copy"],
        data=embed_code.encode("utf-8"),
        file_name="mansa_gold_widget.html",
        mime="text/html",
        key="widget_dl_settings"
    )
    st.caption("💡 " + ("يعمل بدون خادم · يتحدث كل 60 ثانية · مجاني تماماً" if use_ar else
                            "Runs client-side · updates every 60s · completely free"))


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: WEEKLY REPORT 📋
# ═══════════════════════════════════════════════════════════════════════════════
elif nav == L["nav_report"]:
    use_ar = is_rtl()
    ph("📋  " + L["report_title"], L["report_sub"])

    @st.cache_data(ttl=CACHE_TTL_HISTORY)
    def get_report_data() -> dict:
        """Fetch 5-week gold history for the weekly report."""
        df = fetch_history("1mo", "GC=F")
        if df.empty:
            return {}
        cl = find_col(df, ["Close", "Close_GC=F"])
        dt = find_col(df, ["Date", "Datetime"])
        if not cl or not dt:
            return {}
        c = df[cl].dropna()
        if len(c) < 5:
            return {}
        return dict(
            week_chg  = float((c.iloc[-1] - c.iloc[-5]) / c.iloc[-5] * 100),
            month_chg = float((c.iloc[-1] - c.iloc[0])  / c.iloc[0]  * 100),
            wk_high   = float(c.tail(5).max()),
            wk_low    = float(c.tail(5).min()),
            current   = float(c.iloc[-1]),
        )

    rpt      = get_report_data()
    week_str = datetime.date.today().strftime("%d %b %Y")
    wk_chg   = rpt.get("week_chg",  live["gold"]["pct"] * 5)
    wk_high  = rpt.get("wk_high",   g_ref * 1.01)
    wk_low   = rpt.get("wk_low",    g_ref * 0.99)
    wk_col   = C["green"] if wk_chg >= 0 else C["red"]
    wk_icon  = "▲" if wk_chg >= 0 else "▼"
    vix_now  = live["vix"]["price"]
    dxy_pct  = live["dxy"]["pct"]
    outlook  = ("صاعد 🟢" if wk_chg > 1 else "هابط 🔴" if wk_chg < -1 else "محايد 🟡") if use_ar else                ("Bullish 🟢" if wk_chg > 1 else "Bearish 🔴" if wk_chg < -1 else "Neutral 🟡")

    # ── Report header ─────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style='background:{C["card2"]};border:1px solid {C["border2"]};border-radius:8px;
            padding:20px 26px;{"direction:rtl;text-align:right;" if use_ar else ""}'>
      <div style='display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px;'>
    <div>
      <div style='font-family:{C["font_h"]},serif;font-size:20px;font-weight:900;
                  color:{C["gold_pale"]};'>☽ {L["report_week"]} · {week_str}</div>
      <div style='font-size:11px;color:{C["muted"]};margin-top:3px;'>{L["report_sub"]}</div>
    </div>
    <div style='text-align:center;'>
      <div style='font-family:{C["font_m"]},monospace;font-size:30px;
                  color:{C["gold_hi"]};font-weight:900;'>${g_ref:,.2f}</div>
      <div style='font-size:12px;color:{wk_col};font-weight:700;'>
        {wk_icon} {abs(wk_chg):.2f}% {"هذا الأسبوع" if use_ar else "this week"}
      </div>
    </div>
      </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Performance grid ──────────────────────────────────────────────────────
    st.markdown(f"<div class='section-label'>📊 {L['report_perf']}</div>",
            unsafe_allow_html=True)
    pc1, pc2, pc3, pc4 = st.columns(4)
    for col, lbl, val, vc in [
        (pc1, ("السعر الحالي"   if use_ar else "Current Price"),   f"${g_ref:,.2f}",    C["gold_pale"]),
        (pc2, ("أعلى الأسبوع"  if use_ar else "Week High"),        f"${wk_high:,.2f}",  C["green"]),
        (pc3, ("أدنى الأسبوع"  if use_ar else "Week Low"),         f"${wk_low:,.2f}",   C["red"]),
        (pc4, ("التغيير الأسبوعي" if use_ar else "Week Change"),   f"{wk_chg:+.2f}%",   wk_col),
    ]:
        with col:
            st.markdown(f"""
            <div class='stat-card'>
              <div class='stat-label'>{lbl}</div>
              <div class='stat-value' style='color:{vc};font-size:17px;'>{val}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Macro environment ─────────────────────────────────────────────────────
    st.markdown(f"<div class='section-label'>🌍 {'البيئة الاقتصادية الكلية' if use_ar else 'Macro Environment'}</div>",
            unsafe_allow_html=True)
    mc1, mc2, mc3 = st.columns(3)
    dxy_col  = C["green"] if dxy_pct < 0 else C["red"]
    vix_col  = C["green"] if vix_now > 20 else C["muted"]
    for col, lbl, val, vc in [
    (mc1, "DXY",    f"{live['dxy']['price']:,.2f} ({dxy_pct:+.2f}%)", dxy_col),
    (mc2, "VIX",    f"{vix_now:.1f}",                                  vix_col),
    (mc3, "US 10Y", f"{live['us10y']['price']:.2f}%",                  C["muted"]),
    ]:
        with col:
            st.markdown(f"""
            <div class='stat-card'>
              <div class='stat-label'>{lbl}</div>
              <div class='stat-value' style='color:{vc};font-size:17px;'>{val}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Outlook ───────────────────────────────────────────────────────────────
    st.markdown(f"<div class='section-label'>🔭 {L['report_outlook']}</div>",
            unsafe_allow_html=True)
    vix_note = ("خوف عالٍ → طلب على الملاذ الآمن ✅" if vix_now > 22 else "هدوء السوق 🟡") if use_ar else                ("Elevated fear → safe-haven demand ✅" if vix_now > 22 else "Calm markets 🟡")
    dxy_note = ("دولار ضعيف → يدعم الذهب ✅" if dxy_pct < -0.3 else
            "دولار قوي → ضغط على الذهب ⚠️" if dxy_pct > 0.3 else "دولار محايد 🟡") if use_ar else                ("Weak USD → supports gold ✅" if dxy_pct < -0.3 else
            "Strong USD → headwind ⚠️" if dxy_pct > 0.3 else "Neutral USD 🟡")
    st.markdown(f"""
    <div style='background:{C["card"]};border:1px solid {C["border2"]};border-radius:6px;
            padding:16px 20px;{"direction:rtl;text-align:right;" if use_ar else ""}'>
      <div style='font-size:14px;color:{C["text"]};line-height:2.0;'>
    <b style='color:{wk_col};'>{outlook}</b> &nbsp;·&nbsp;
    VIX: <b style='color:{vix_col};'>{vix_note}</b> &nbsp;·&nbsp;
    USD: <b style='color:{dxy_col};'>{dxy_note}</b>
      </div>
      <div style='font-size:11px;color:{C["dim"]};margin-top:10px;font-style:italic;'>
    ⚠️ {L["disclaimer"]}
      </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Download ──────────────────────────────────────────────────────────────
    printable = (
    f"<!DOCTYPE html><html><head><meta charset='utf-8'>"
    f"<style>body{{font-family:Georgia,serif;max-width:800px;margin:0 auto;"
    f"padding:20px;background:#050408;color:#F0E8D5;}}h1{{color:#F5C842;}}"
    f".m{{display:inline-block;margin:8px;padding:12px;background:#0D0A18;"
    f"border-radius:4px;min-width:140px;}}</style></head><body>"
    f"<h1>☽ MANSA · {L['report_week']} — {week_str}</h1>"
    f"<div class='m'><b>Price</b><br>${g_ref:,.2f}</div>"
    f"<div class='m'><b>High</b><br>${wk_high:,.2f}</div>"
    f"<div class='m'><b>Low</b><br>${wk_low:,.2f}</div>"
    f"<div class='m'><b>Change</b><br>{wk_chg:+.2f}%</div>"
    f"<hr><p>VIX {vix_now:.1f} · DXY {dxy_pct:+.2f}%</p>"
    f"<p><i>{L['disclaimer']}</i></p></body></html>"
    )
    # Download button — no PDF attach box
    col_dl, _ = st.columns([2, 3])
    with col_dl:
        st.download_button(
        label="📥 " + L["report_dl"],
        data=printable.encode("utf-8"),
        file_name=f"mansa_report_{datetime.date.today()}.html",
        mime="application/octet-stream",
        use_container_width=True,
    )


elif nav == L["nav_cb"]:
    use_ar = is_rtl()
    ph("🏦  " + L["cb_title"], L["cb_sub"])

    # ── Data: World Gold Council Q4 2024 ─────────────────────────────────────
    CB_WORLD = [
    {"flag":"🇺🇸","name_ar":"الولايات المتحدة","name_en":"United States",
     "tonnes":8133.5,"pct_reserves":79.0,"trend":"stable","yoy":0},
    {"flag":"🇩🇪","name_ar":"ألمانيا","name_en":"Germany",
     "tonnes":3352.3,"pct_reserves":74.5,"trend":"stable","yoy":0},
    {"flag":"🇮🇹","name_ar":"إيطاليا","name_en":"Italy",
     "tonnes":2451.8,"pct_reserves":70.3,"trend":"stable","yoy":0},
    {"flag":"🇫🇷","name_ar":"فرنسا","name_en":"France",
     "tonnes":2436.8,"pct_reserves":71.2,"trend":"stable","yoy":0},
    {"flag":"🇷🇺","name_ar":"روسيا","name_en":"Russia",
     "tonnes":2332.7,"pct_reserves":29.5,"trend":"up","yoy":+3},
    {"flag":"🇨🇳","name_ar":"الصين","name_en":"China",
     "tonnes":2279.6,"pct_reserves":4.9,"trend":"up","yoy":+44},
    {"flag":"🇨🇭","name_ar":"سويسرا","name_en":"Switzerland",
     "tonnes":1039.9,"pct_reserves":8.9,"trend":"stable","yoy":0},
    {"flag":"🇮🇳","name_ar":"الهند","name_en":"India",
     "tonnes":876.2,"pct_reserves":10.1,"trend":"up","yoy":+72},
    {"flag":"🇯🇵","name_ar":"اليابان","name_en":"Japan",
     "tonnes":845.9,"pct_reserves":4.9,"trend":"stable","yoy":0},
    {"flag":"🇳🇱","name_ar":"هولندا","name_en":"Netherlands",
     "tonnes":612.5,"pct_reserves":70.7,"trend":"stable","yoy":0},
    ]
    CB_ARAB = [
    {"flag":"🇸🇦","name_ar":"المملكة العربية السعودية","name_en":"Saudi Arabia",
     "tonnes":323.1,"pct_reserves":3.3,"trend":"stable","yoy":0,
     "insight_ar":"الاحتياطي الذهبي السعودي ثابت منذ سنوات — البنك المركزي يعتمد على النفط أكثر.",
     "insight_en":"Saudi gold reserves have been static — the central bank relies more on oil revenues."},
    {"flag":"🇱🇧","name_ar":"لبنان","name_en":"Lebanon",
     "tonnes":286.8,"pct_reserves":32.0,"trend":"stable","yoy":0,
     "insight_ar":"لبنان يمتلك أحد أكبر احتياطيات الذهب نسبةً في المنطقة رغم الأزمة الاقتصادية.",
     "insight_en":"Lebanon holds the region's largest gold reserve relative to total reserves despite economic crisis."},
    {"flag":"🇩🇿","name_ar":"الجزائر","name_en":"Algeria",
     "tonnes":173.6,"pct_reserves":19.0,"trend":"up","yoy":+12,
     "insight_ar":"الجزائر تراكم الذهب بثبات كحماية من تذبذب أسعار النفط.",
     "insight_en":"Algeria steadily accumulates gold as a hedge against oil price volatility."},
    {"flag":"🇪🇬","name_ar":"مصر","name_en":"Egypt",
     "tonnes":126.6,"pct_reserves":16.4,"trend":"up","yoy":+14,
     "insight_ar":"مصر زادت احتياطياتها الذهبية بشكل ملحوظ منذ 2023 لتعزيز استقرار الجنيه.",
     "insight_en":"Egypt notably increased gold reserves since 2023 to strengthen pound stability."},
    {"flag":"🇱🇾","name_ar":"ليبيا","name_en":"Libya",
     "tonnes":116.6,"pct_reserves":7.0,"trend":"stable","yoy":0,
     "insight_ar":"احتياطيات ليبيا مستقرة رغم عدم الاستقرار السياسي.",
     "insight_en":"Libya's reserves remain stable despite political instability."},
    {"flag":"🇮🇶","name_ar":"العراق","name_en":"Iraq",
     "tonnes":145.9,"pct_reserves":9.9,"trend":"up","yoy":+22,
     "insight_ar":"العراق يعزز احتياطياته الذهبية بقوة في السنوات الأخيرة.",
     "insight_en":"Iraq has been aggressively building gold reserves in recent years."},
    {"flag":"🇰🇼","name_ar":"الكويت","name_en":"Kuwait",
     "tonnes":79.0,"pct_reserves":8.6,"trend":"stable","yoy":0,
     "insight_ar":"الكويت تحتفظ باحتياطي ذهبي معتدل ومستقر.",
     "insight_en":"Kuwait maintains a modest, stable gold reserve."},
    {"flag":"🇯🇴","name_ar":"الأردن","name_en":"Jordan",
     "tonnes":70.5,"pct_reserves":59.6,"trend":"up","yoy":+7,
     "insight_ar":"الأردن يحتفظ بنسبة عالية من الذهب في احتياطياته — من أعلى النسب عالمياً.",
     "insight_en":"Jordan holds a high % of reserves in gold — one of the highest ratios globally."},
    {"flag":"🇦🇪","name_ar":"الإمارات","name_en":"UAE",
     "tonnes":74.5,"pct_reserves":4.6,"trend":"stable","yoy":0,
     "insight_ar":"الإمارات تعتمد على متنوع من الأصول الاحتياطية بما يشمل الذهب.",
     "insight_en":"UAE relies on a diversified reserve portfolio including gold."},
    {"flag":"🇲🇦","name_ar":"المغرب","name_en":"Morocco",
     "tonnes":22.1,"pct_reserves":6.8,"trend":"stable","yoy":0,
     "insight_ar":"المغرب يحتفظ باحتياطي ذهبي صغير نسبياً.",
     "insight_en":"Morocco holds a relatively small gold reserve."},
    ]

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab_world, tab_arab = st.tabs([
    "🌍 " + L["cb_world"],
    "🕌 " + L["cb_arab"],
    ])

    def _cb_table(rows: list, show_insight: bool = False) -> None:
        """Render a central bank gold reserves table with optional insight tooltips."""
        rtl_s = "direction:rtl;text-align:right;" if use_ar else ""
        # ── Table header ──────────────────────────────────────────────────────
        st.markdown(f"""
        <table style='width:100%;border-collapse:collapse;
                      font-family:{C["font_m"]},monospace;font-size:12px;{rtl_s}'>
          <thead>
            <tr style='border-bottom:2px solid {C["gold"]}44;'>
              <th style='padding:8px 6px;color:{C["gold"]};
                         text-align:{"right" if use_ar else "left"};'>#</th>
              <th style='padding:8px 6px;color:{C["gold"]};
                         text-align:{"right" if use_ar else "left"};'>
                {L["cb_country"]}</th>
              <th style='padding:8px 6px;color:{C["gold"]};text-align:right;'>{L["cb_tonnes"]}</th>
              <th style='padding:8px 6px;color:{C["gold"]};text-align:right;'>{L["cb_pct"]}</th>
              <th style='padding:8px 6px;color:{C["gold"]};text-align:center;'>{L["cb_trend"]}</th>
            </tr>
          </thead><tbody>""",
        unsafe_allow_html=True)
        # ── Table rows ────────────────────────────────────────────────────────
        for idx_r, row in enumerate(rows):
            name       = row["name_ar"] if use_ar else row["name_en"]
            trend_icon = "↑" if row["trend"] == "up" else ("→" if row["trend"] == "stable" else "↓")
            trend_col  = C["green"] if row["trend"] == "up" else (C["muted"] if row["trend"] == "stable" else C["red"])
            yoy_s      = f"+{row['yoy']}t" if row["yoy"] > 0 else (f"{row['yoy']}t" if row["yoy"] < 0 else "—")
            bar_w      = min(100, row["tonnes"] / 82.0)
            st.markdown(f"""
            <tr style='border-bottom:1px solid {C["border"]}22;'>
              <td style='padding:10px 6px;color:{C["dim"]};
                         text-align:{"right" if use_ar else "left"};'>{idx_r+1}</td>
              <td style='padding:10px 6px;color:{C["text"]};
                         text-align:{"right" if use_ar else "left"};'>
                {row["flag"]} {name}</td>
              <td style='padding:10px 6px;text-align:right;'>
                <div style='font-weight:700;color:{C["gold_pale"]};'>{row["tonnes"]:,.1f}t</div>
                <div style='width:{bar_w:.0f}%;height:3px;background:rgba(212,160,23,0.4);
                            border-radius:2px;margin-top:3px;float:right;'></div>
              </td>
              <td style='padding:10px 6px;text-align:right;color:{C["muted"]};'>
                {row["pct_reserves"]:.1f}%</td>
              <td style='padding:10px 6px;text-align:center;
                         color:{trend_col};font-weight:700;font-size:14px;'>
                {trend_icon}
                <div style='font-size:9px;color:{C["dim"]};'>{yoy_s}</div>
              </td>
            </tr>""", unsafe_allow_html=True)
        st.markdown("</tbody></table>", unsafe_allow_html=True)
        # ── Insight cards (Arab tab only) ─────────────────────────────────────
        if show_insight:
            st.markdown("<br>", unsafe_allow_html=True)
            for row in rows:
                if row.get("insight_ar"):
                    insight = row["insight_ar"] if use_ar else row["insight_en"]
                    name    = row["name_ar"]    if use_ar else row["name_en"]
                    st.markdown(f"""
                    <div style='background:{C["card"]};border:1px solid {C["border"]};
                                border-left:3px solid {C["gold"]};border-radius:4px;
                                padding:10px 14px;margin-bottom:6px;
                                {"direction:rtl;text-align:right;" if use_ar else ""}'>
                      <span style='color:{C["gold"]};font-weight:700;'>
                        {row["flag"]} {name}</span>
                      <span style='color:{C["muted"]};font-size:11px;'> · </span>
                      <span style='color:{C["text"]};font-size:12px;'>{insight}</span>
                    </div>""", unsafe_allow_html=True)
    with tab_world:
        _cb_table(CB_WORLD, show_insight=False)
    total_world = sum(r["tonnes"] for r in CB_WORLD)
    st.markdown(f"""
    <div style='text-align:right;font-size:11px;color:{C["muted"]};margin-top:8px;'>
      {"الإجمالي في أكبر 10 دول:" if use_ar else "Top 10 total:"} {total_world:,.0f}t
      · {"المصدر: WGC Q4 2024" if use_ar else "Source: WGC Q4 2024"}
    </div>""", unsafe_allow_html=True)

    # Bar chart
    import plotly.graph_objects as _pgocb
    fig_cb = _pgocb.Figure(_pgocb.Bar(
        x=[r["name_ar"] if use_ar else r["name_en"] for r in CB_WORLD],
        y=[r["tonnes"] for r in CB_WORLD],
        marker_color=[C["gold"] if i == 0 else rgba(C["gold"], 0.53) for i in range(len(CB_WORLD))],
        text=[f"{r['tonnes']:,.0f}t" for r in CB_WORLD],
        textposition="outside",
        textfont=dict(color=C["text"], size=9),
    ))
    fig_cb.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=C["text"], size=10), height=300,
        margin=dict(l=0,r=0,t=20,b=60),
        xaxis=dict(tickangle=-35, gridcolor=C["border2"]),
        yaxis=dict(gridcolor=C["border2"], ticksuffix="t"),
    )
    st.plotly_chart(fig_cb, use_container_width=True, config={"displayModeBar": False})

    with tab_arab:
        _cb_table(CB_ARAB, show_insight=True)
    total_arab = sum(r["tonnes"] for r in CB_ARAB)
    st.markdown(f"""
    <div style='text-align:right;font-size:11px;color:{C["muted"]};margin-top:8px;'>
      {"الإجمالي العربي:" if use_ar else "Arab world total:"} {total_arab:,.0f}t
      · {"المصدر: WGC Q4 2024" if use_ar else "Source: WGC Q4 2024"}
    </div>""", unsafe_allow_html=True)

    # Value in USD
    st.markdown("<br>", unsafe_allow_html=True)
    total_val = total_arab * 32150.75 * g_ref / 1000  # tonnes → oz → USD billions
    st.markdown(f"""
    <div style='background:{C["card2"]};border:1px solid {C["gold"]}44;border-radius:8px;
                padding:16px 20px;text-align:center;'>
      <div style='font-size:11px;color:{C["muted"]};letter-spacing:.2em;'>
        {"القيمة الإجمالية لاحتياطيات الذهب العربية" if use_ar else "TOTAL VALUE OF ARAB GOLD RESERVES"}
      </div>
      <div style='font-family:{C["font_m"]},monospace;font-size:36px;
                  font-weight:900;color:{C["gold_hi"]};margin:8px 0;'>
        ${total_val/1e9:,.1f}B
      </div>
      <div style='font-size:11px;color:{C["dim"]};'>
        {"بناءً على السعر الحالي" if use_ar else "Based on current spot price"} ${g_ref:,.2f}/oz
      </div>
    </div>""", unsafe_allow_html=True)



# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: GEOPOLITICAL MAP 🌍
# ═══════════════════════════════════════════════════════════════════════════════
elif nav == L["nav_geo"]:
    use_ar = is_rtl()
    ph("🌍  " + L["geo_title"], L["geo_sub"])

    # ── Geopolitical events data (updated manually, reflects current situation) ──
    GEO_EVENTS = [
        {
            "id": "ukraine", "flag": "🇺🇦",
            "name_ar": "حرب روسيا-أوكرانيا", "name_en": "Russia-Ukraine War",
            "region_ar": "أوروبا الشرقية", "region_en": "Eastern Europe",
            "lat": 49.0, "lon": 32.0, "intensity": "HIGH",
            "gold_impact": "bullish",
            "desc_ar": "الصراع المستمر منذ 2022 يرفع الطلب على الملاذات الآمنة ويدعم الذهب بشكل مباشر عبر الضغط على الطاقة الأوروبية وعدم استقرار الأسواق.",
            "desc_en": "Ongoing since 2022, this conflict directly lifts safe-haven demand. Energy disruption raises inflation expectations across Europe, supporting gold.",
            "price_effect": "+$80–$150/oz estimated safe-haven premium",
        },
        {
            "id": "gaza", "flag": "🇵🇸",
            "name_ar": "حرب غزة", "name_en": "Gaza War",
            "region_ar": "الشرق الأوسط", "region_en": "Middle East",
            "lat": 31.5, "lon": 34.5, "intensity": "HIGH",
            "gold_impact": "bullish",
            "desc_ar": "التصعيد الإقليمي يزيد من حالة عدم اليقين. توترات البحر الأحمر تؤثر على شحن النفط مما يرفع التضخم ويدعم الذهب.",
            "desc_en": "Regional escalation raises uncertainty. Red Sea shipping disruption impacts oil supply chains, driving inflation expectations and gold demand.",
            "price_effect": "+$40–$80/oz Middle East risk premium",
        },
        {
            "id": "iran", "flag": "🇮🇷",
            "name_ar": "توترات إيران-الغرب", "name_en": "Iran–West Tensions",
            "region_ar": "الخليج العربي", "region_en": "Persian Gulf",
            "lat": 32.0, "lon": 53.0, "intensity": "MED",
            "gold_impact": "bullish",
            "desc_ar": "برنامج إيران النووي والعقوبات الغربية يحافظان على علاوة مخاطر ثابتة في منطقة الخليج. أي تصعيد يرفع النفط والذهب معاً.",
            "desc_en": "Iran's nuclear programme and Western sanctions maintain a persistent Gulf risk premium. Any escalation spikes oil and gold simultaneously.",
            "price_effect": "+$20–$60/oz oil-linked premium",
        },
        {
            "id": "taiwan", "flag": "🇹🇼",
            "name_ar": "توترات مضيق تايوان", "name_en": "Taiwan Strait Tensions",
            "region_ar": "آسيا-المحيط الهادئ", "region_en": "Asia-Pacific",
            "lat": 23.5, "lon": 121.0, "intensity": "MED",
            "gold_impact": "bullish",
            "desc_ar": "التوترات بين الصين وتايوان تهدد سلاسل توريد أشباه الموصلات العالمية. أي أزمة ستضرب الأسواق وتقود المستثمرين إلى الذهب.",
            "desc_en": "China-Taiwan tensions threaten global semiconductor supply chains. A crisis would shock equity markets and drive massive flight to gold.",
            "price_effect": "+$100–$200/oz in a major escalation scenario",
        },
        {
            "id": "sudan", "flag": "🇸🇩",
            "name_ar": "الحرب الأهلية السودانية", "name_en": "Sudan Civil War",
            "region_ar": "شمال أفريقيا", "region_en": "North Africa",
            "lat": 15.5, "lon": 32.5, "intensity": "HIGH",
            "gold_impact": "neutral",
            "desc_ar": "السودان من أكبر منتجي الذهب في أفريقيا. اضطرابات الإنتاج المحلي لها تأثير محدود على السعر العالمي لكنها تؤثر على سوق الذهب الإقليمي.",
            "desc_en": "Sudan is a major African gold producer. Local production disruptions have limited global price impact but affect regional gold flows significantly.",
            "price_effect": "Neutral globally · Regional supply disruption",
        },
        {
            "id": "yemen", "flag": "🇾🇪",
            "name_ar": "هجمات الحوثيين-البحر الأحمر", "name_en": "Houthi Red Sea Attacks",
            "region_ar": "البحر الأحمر", "region_en": "Red Sea",
            "lat": 15.0, "lon": 42.5, "intensity": "HIGH",
            "gold_impact": "bullish",
            "desc_ar": "هجمات الحوثيين على الشحن في البحر الأحمر ترفع تكاليف الشحن العالمية وتضيف ضغطاً تضخمياً يدعم الذهب.",
            "desc_en": "Houthi attacks on Red Sea shipping raise global freight costs, adding inflationary pressure that supports gold as an inflation hedge.",
            "price_effect": "+$15–$30/oz via shipping inflation channel",
        },
        {
            "id": "sahel", "flag": "🌍",
            "name_ar": "عدم الاستقرار في منطقة الساحل", "name_en": "Sahel Instability",
            "region_ar": "غرب أفريقيا", "region_en": "West Africa",
            "lat": 15.0, "lon": 2.0, "intensity": "MED",
            "gold_impact": "neutral",
            "desc_ar": "الانقلابات في مالي وبوركينا فاسو والنيجر تعطل تدفق الذهب الأفريقي نحو الأسواق العالمية.",
            "desc_en": "Coups in Mali, Burkina Faso and Niger disrupt the flow of African gold toward global markets, creating minor regional supply tightness.",
            "price_effect": "Minor regional supply constraint",
        },
        {
            "id": "sanctions", "flag": "🇷🇺",
            "name_ar": "العقوبات على روسيا والذهب", "name_en": "Russia Gold Sanctions",
            "region_ar": "عالمي", "region_en": "Global",
            "lat": 60.0, "lon": 90.0, "intensity": "MED",
            "gold_impact": "bullish",
            "desc_ar": "العقوبات الغربية على الذهب الروسي تقلص العرض من ثاني أكبر منتج في العالم وترفع الأسعار العالمية.",
            "desc_en": "Western sanctions on Russian gold reduce supply from the world's second-largest producer, tightening global supply and supporting prices.",
            "price_effect": "+$10–$25/oz supply constraint premium",
        },
    ]

    # ── Geopolitical Risk Score ───────────────────────────────────────────────
    intensity_weight = {"HIGH": 3, "MED": 2, "LOW": 1}
    bullish_events   = [e for e in GEO_EVENTS if e["gold_impact"] == "bullish"]
    bearish_events   = [e for e in GEO_EVENTS if e["gold_impact"] == "bearish"]
    geo_score = sum(intensity_weight.get(e["intensity"], 1) for e in bullish_events)
    geo_max   = len(GEO_EVENTS) * 3
    geo_pct   = min(100, int(geo_score / geo_max * 100))
    geo_col   = C["green"] if geo_pct > 60 else (C["gold"] if geo_pct > 35 else C["red"])
    rtl_s     = "direction:rtl;text-align:right;" if use_ar else ""

    # ── Summary row ───────────────────────────────────────────────────────────
    gs1, gs2, gs3, gs4 = st.columns(4)
    for col, lbl, val, vc in [
        (gs1, L["geo_risk"],          f"{geo_pct}/100",           geo_col),
        (gs2, L["geo_events"],        f"{len(GEO_EVENTS)}",       C["gold_pale"]),
        (gs3, L["geo_bullish"],       f"{len(bullish_events)}",   C["green"]),
        (gs4, L["geo_bearish"],       f"{len(bearish_events)}",   C["red"]),
    ]:
        with col:
            st.markdown(f"""
            <div class='stat-card'>
              <div class='stat-label'>{lbl}</div>
              <div class='stat-value' style='color:{vc};font-size:20px;'>{val}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)


    # ── Impact filter ────────────────────────────────────────────────────────
    _filter_opts = {
        "all":     L.get("geo_all", "All"),
        "bullish": L["geo_bullish"],
        "bearish": L["geo_bearish"],
        "neutral": L["geo_neutral"],
    }
    _geo_filter = st.session_state.get("geo_impact_filter", "all")
    _f_cols     = st.columns(4)
    for _fk, (_fc, (_fkey, _flbl)) in enumerate(zip(_f_cols, _filter_opts.items())):
        with _fc:
            _active = _geo_filter == _fkey
            if st.button(_flbl, key=f"gf_{_fkey}",
                         type="primary" if _active else "secondary",
                         use_container_width=True):
                st.session_state["geo_impact_filter"] = _fkey
                st.rerun()

    _geo_filter = st.session_state.get("geo_impact_filter", "all")
    _visible_events = [e for e in GEO_EVENTS
                       if _geo_filter == "all" or e["gold_impact"] == _geo_filter]

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Risk timeline ─────────────────────────────────────────────────────────
    st.markdown(f"<div class='section-label'>{L.get('geo_timeline','Event Timeline')}</div>",
                unsafe_allow_html=True)
    import plotly.graph_objects as _go_geo
    _intensity_score = {"HIGH": 3, "MED": 2, "LOW": 1}
    _col_map = {"bullish": C["green"], "bearish": C["red"], "neutral": C["gold"]}
    fig_geo_bar = _go_geo.Figure()
    for ev in _visible_events:
        _name = ev["name_ar"] if use_ar else ev["name_en"]
        _sc   = _intensity_score.get(ev["intensity"], 1)
        _vc   = _col_map.get(ev["gold_impact"], C["muted"])
        fig_geo_bar.add_trace(_go_geo.Bar(
            x=[_name], y=[_sc],
            marker_color=_vc,
            text=[ev["price_effect"].split(" ")[0]],
            textposition="outside",
            textfont=dict(size=9, color=C["text"]),
            showlegend=False,
        ))
    fig_geo_bar.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=180, margin=dict(l=0, r=0, t=10, b=60),
        font=dict(color=C["text"], size=9),
        xaxis=dict(tickangle=-30, gridcolor=C["border2"]),
        yaxis=dict(gridcolor=C["border2"], tickvals=[1,2,3],
                   ticktext=[("منخفض" if use_ar else "LOW"),
                              ("متوسط" if use_ar else "MED"),
                              ("عالٍ" if use_ar else "HIGH")]),
        barmode="group",
    )
    st.plotly_chart(fig_geo_bar, use_container_width=True, config={"displayModeBar": False})

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Interactive SVG world map ─────────────────────────────────────────────
    # Build hotspot dots positioned on a simplified equirectangular projection
    def lat_lon_to_pct(lat, lon):
        """Convert lat/lon to SVG percentage coordinates."""
        x = (lon + 180) / 360 * 100
        y = (90 - lat) / 180 * 100
        return round(x, 2), round(y, 2)

    dots_svg = ""
    labels_svg = ""
    for ev in GEO_EVENTS:
        x, y = lat_lon_to_pct(ev["lat"], ev["lon"])
        col_map  = {"bullish": C["green"], "bearish": C["red"], "neutral": C["gold"]}
        dot_col  = col_map.get(ev["gold_impact"], C["muted"])
        pulse_r  = 14 if ev["intensity"] == "HIGH" else 10
        name     = ev["name_ar"] if use_ar else ev["name_en"]
        desc     = ev["desc_ar"] if use_ar else ev["desc_en"]
        dots_svg += f"""
        <g class="geo-dot" data-id="{ev['id']}">
          <circle cx="{x}%" cy="{y}%" r="{pulse_r}" fill="{dot_col}" opacity="0.18"
                  class="pulse-ring"/>
          <circle cx="{x}%" cy="{y}%" r="6" fill="{dot_col}" stroke="{C['bg']}"
                  stroke-width="1.5" style="cursor:pointer;"
                  onclick="showDetail('{ev['id']}')"/>
          <text x="{x}%" y="{y}%" dy="-10" text-anchor="middle"
                font-size="9" fill="{dot_col}" font-family="Cairo,serif"
                style="pointer-events:none;">{ev['flag']}</text>
        </g>"""

    # Detail panels as hidden divs
    detail_html = ""
    for ev in GEO_EVENTS:
        col_map = {"bullish": C["green"], "bearish": C["red"], "neutral": C["gold"]}
        dot_col = col_map.get(ev["gold_impact"], C["muted"])
        impact_lbl = (L["geo_bullish"] if ev["gold_impact"] == "bullish"
                      else L["geo_bearish"] if ev["gold_impact"] == "bearish"
                      else L["geo_neutral"])
        name   = ev["name_ar"] if use_ar else ev["name_en"]
        region = ev["region_ar"] if use_ar else ev["region_en"]
        desc   = ev["desc_ar"] if use_ar else ev["desc_en"]
        detail_html += f"""
        <div id="detail-{ev['id']}" class="detail-panel" style="display:none;">
          <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:8px;">
            <div>
              <div style="font-size:18px;font-weight:900;color:{dot_col};">
                {ev['flag']} {name}</div>
              <div style="font-size:11px;color:#806050;margin-top:2px;">{region} · {ev['intensity']} intensity</div>
            </div>
            <div style="background:{dot_col}22;border:1px solid {dot_col}55;border-radius:4px;
                        padding:4px 10px;font-size:11px;font-weight:700;color:{dot_col};">
              {impact_lbl}</div>
          </div>
          <div style="font-size:12px;color:#D0C0A0;line-height:1.8;margin-top:10px;
                      {'direction:rtl;text-align:right;' if use_ar else ''}">{desc}</div>
          <div style="margin-top:10px;background:#1A1020;border-left:3px solid {dot_col};
                      padding:8px 12px;border-radius:0 4px 4px 0;font-size:11px;
                      font-family:monospace;color:{dot_col};">
            📈 {L["geo_impact"]}: {ev['price_effect']}</div>
        </div>"""

    map_html = f"""
<!DOCTYPE html><html>
<head><meta charset="UTF-8">
<style>
  body {{ margin:0; background:{C["bg"]}; font-family:'Cairo',sans-serif;
          color:{C["text"]}; overflow-x:hidden; }}
  .map-wrap {{ position:relative; width:100%; }}
  .map-svg {{ width:100%; border-radius:8px;
               border:1px solid {C["border2"]}; display:block; }}
  @keyframes pulse {{
    0%   {{ r:6; opacity:.18; }}
    50%  {{ r:16; opacity:.05; }}
    100% {{ r:6; opacity:.18; }}
  }}
  .pulse-ring {{ animation: pulse 2s infinite ease-in-out; }}
  .geo-dot:hover circle:last-of-type {{ r:8; }}
  .detail-panel {{
    background:{C["card2"]}; border:1px solid {C["border"]};
    border-radius:8px; padding:16px; margin-top:10px;
    animation: fadeIn .3s ease;
  }}
  @keyframes fadeIn {{ from{{opacity:0;transform:translateY(-6px)}} to{{opacity:1;transform:none}} }}
  .close-btn {{
    float:right; background:none; border:none; color:{C["muted"]};
    cursor:pointer; font-size:16px; padding:0 4px;
  }}
</style>
</head>
<body>
<div class="map-wrap">
  <!-- World map SVG — simplified continents using paths -->
  <svg class="map-svg" viewBox="0 0 1000 500" xmlns="http://www.w3.org/2000/svg">
    <!-- Ocean background -->
    <rect width="1000" height="500" fill="{C["bg2"]}"/>
    <!-- Grid lines -->
    <line x1="0" y1="250" x2="1000" y2="250" stroke="{C["border2"]}" stroke-width="0.5" opacity="0.5"/>
    <line x1="500" y1="0" x2="500" y2="500" stroke="{C["border2"]}" stroke-width="0.5" opacity="0.5"/>
    <!-- Simplified continent outlines -->
    <!-- North America -->
    <path d="M80,80 L200,60 L240,100 L260,160 L220,220 L180,240 L140,220 L100,180 L70,130 Z"
          fill="{C["card"]}" stroke="{C["border"]}" stroke-width="0.8" opacity="0.85"/>
    <!-- South America -->
    <path d="M200,240 L280,230 L310,280 L300,380 L260,420 L220,400 L190,340 L185,270 Z"
          fill="{C["card"]}" stroke="{C["border"]}" stroke-width="0.8" opacity="0.85"/>
    <!-- Europe -->
    <path d="M440,60 L530,55 L560,90 L540,130 L490,140 L450,120 L430,90 Z"
          fill="{C["card"]}" stroke="{C["border"]}" stroke-width="0.8" opacity="0.85"/>
    <!-- Africa -->
    <path d="M450,150 L560,140 L590,200 L580,330 L520,380 L470,360 L430,300 L420,220 Z"
          fill="{C["card"]}" stroke="{C["border"]}" stroke-width="0.8" opacity="0.85"/>
    <!-- Middle East -->
    <path d="M545,135 L620,125 L650,160 L640,200 L590,210 L555,190 Z"
          fill="{C["card"]}" stroke="{C["border"]}" stroke-width="0.8" opacity="0.85"/>
    <!-- Russia/Central Asia -->
    <path d="M530,40 L800,30 L820,100 L750,120 L660,110 L580,130 L540,100 Z"
          fill="{C["card"]}" stroke="{C["border"]}" stroke-width="0.8" opacity="0.85"/>
    <!-- South/SE Asia -->
    <path d="M650,120 L820,100 L850,170 L800,200 L720,210 L670,190 L645,155 Z"
          fill="{C["card"]}" stroke="{C["border"]}" stroke-width="0.8" opacity="0.85"/>
    <!-- East Asia + Japan -->
    <path d="M800,80 L930,70 L950,140 L900,160 L840,150 L810,115 Z"
          fill="{C["card"]}" stroke="{C["border"]}" stroke-width="0.8" opacity="0.85"/>
    <!-- Australia -->
    <path d="M780,300 L900,290 L920,380 L860,400 L790,380 L765,340 Z"
          fill="{C["card"]}" stroke="{C["border"]}" stroke-width="0.8" opacity="0.85"/>
    <!-- Geopolitical hotspot dots -->
    {dots_svg}
    <!-- Title -->
    <text x="10" y="18" font-family="Cairo,serif" font-size="11"
          fill="{C["gold"]}" opacity="0.7" letter-spacing="2">
      ☽ MANSA · {L["geo_title"].upper()}
    </text>
    <!-- Legend -->
    <g transform="translate(10,480)">
      <circle cx="8" cy="-6" r="5" fill="{C["green"]}" opacity="0.8"/>
      <text x="18" y="-2" font-size="9" fill="{C["muted"]}" font-family="Cairo,serif">
        {L["geo_bullish"]}</text>
      <circle cx="150" cy="-6" r="5" fill="{C["red"]}" opacity="0.8"/>
      <text x="160" y="-2" font-size="9" fill="{C["muted"]}" font-family="Cairo,serif">
        {L["geo_bearish"]}</text>
      <circle cx="290" cy="-6" r="5" fill="{C["gold"]}" opacity="0.8"/>
      <text x="300" y="-2" font-size="9" fill="{C["muted"]}" font-family="Cairo,serif">
        {L["geo_neutral"]}</text>
    </g>
  </svg>

  <!-- Detail panel (shown when dot clicked) -->
  <div id="detail-container" style="margin-top:8px;">
    <div id="detail-placeholder" style="text-align:center;padding:16px;
         color:{C["dim"]};font-size:12px;">
      {"🖱️ انقر على أي نقطة لرؤية التفاصيل" if use_ar else "🖱️ Click any hotspot for details"}
    </div>
    {detail_html}
  </div>
</div>

<script>
function showDetail(id) {{
  // Hide all
  document.querySelectorAll('.detail-panel').forEach(function(el) {{
    el.style.display = 'none';
  }});
  document.getElementById('detail-placeholder').style.display = 'none';
  // Show selected
  var el = document.getElementById('detail-' + id);
  if (el) {{
    el.style.display = 'block';
    el.scrollIntoView({{behavior:'smooth', block:'nearest'}});
  }}
}}
</script>
</body></html>"""

    st.components.v1.html(map_html, height=780, scrolling=False)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Event cards below map ─────────────────────────────────────────────────
    st.markdown(f"<div class='section-label'>📋 {L['geo_events']}</div>",
                unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    for i, ev in enumerate(_visible_events):
        col = col_a if i % 2 == 0 else col_b
        col_map = {"bullish": C["green"], "bearish": C["red"], "neutral": C["gold"]}
        dot_col = col_map.get(ev["gold_impact"], C["muted"])
        name    = ev["name_ar"] if use_ar else ev["name_en"]
        desc    = ev["desc_ar"] if use_ar else ev["desc_en"]
        region  = ev["region_ar"] if use_ar else ev["region_en"]
        impact  = (L["geo_bullish"] if ev["gold_impact"] == "bullish"
                   else L["geo_bearish"] if ev["gold_impact"] == "bearish"
                   else L["geo_neutral"])
        with col:
            st.markdown(f"""
            <div style='background:{C["card"]};border:1px solid {dot_col}33;
                        border-left:3px solid {dot_col};border-radius:5px;
                        padding:12px 14px;margin-bottom:8px;
                        {"direction:rtl;text-align:right;" if use_ar else ""}'>
              <div style='display:flex;justify-content:space-between;
                          align-items:center;gap:8px;flex-wrap:wrap;'>
                <div style='font-size:13px;font-weight:700;color:{C["text"]};'>
                  {ev['flag']} {name}</div>
                <div style='font-size:9px;background:{dot_col}22;color:{dot_col};
                            border:1px solid {dot_col}44;border-radius:3px;
                            padding:2px 7px;font-weight:700;white-space:nowrap;'>
                  {impact}</div>
              </div>
              <div style='font-size:10px;color:{C["muted"]};margin-top:4px;'>
                {region} · {ev["intensity"]}</div>
              <div style='font-size:11px;color:{C["dim"]};margin-top:6px;
                          line-height:1.6;'>{desc[:140]}{'…' if len(desc)>140 else ''}</div>
              <div style='font-size:10px;color:{dot_col};margin-top:6px;
                          font-family:monospace;'>📈 {ev["price_effect"]}</div>
            </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: OIL & GOLD CORRELATION 🛢️
# ═══════════════════════════════════════════════════════════════════════════════
elif nav == L["nav_oilgold"]:
    use_ar = is_rtl()
    ph("🛢️  " + L["og_title"], L["og_sub"])

    # ── Fetch oil + gold history ──────────────────────────────────────────────
    @st.cache_data(ttl=CACHE_TTL_HISTORY)
    def get_oil_gold_data(period: str = "3mo") -> dict:
        """Fetch parallel oil and gold price history for correlation analysis."""
        try:
            import yfinance as yf
            gold_df = yf.download("GC=F", period=period, interval="1d",
                                  progress=False, auto_adjust=True)
            oil_df  = yf.download("CL=F", period=period, interval="1d",
                                  progress=False, auto_adjust=True)
            if gold_df.empty or oil_df.empty:
                return {}
            gc = gold_df["Close"].dropna()
            oc = oil_df["Close"].dropna()
            # Align on common dates
            common = gc.index.intersection(oc.index)
            gc = gc.loc[common]; oc = oc.loc[common]
            corr = float(gc.corr(oc))
            # Rolling 30-day correlation
            import pandas as pd
            combined = pd.DataFrame({"gold": gc, "oil": oc})
            roll_corr = combined["gold"].rolling(20).corr(combined["oil"])
            dates     = [str(d.date()) for d in common]
            return dict(
                dates=dates,
                gold=list(gc.round(2)),
                oil=list(oc.round(2)),
                corr=round(corr, 3),
                roll_corr=list(roll_corr.round(3)),
                n=len(dates),
            )
        except Exception:
            _log.debug("get_oil_gold_data failed", exc_info=True)
            return {}

    # ── Period selector ───────────────────────────────────────────────────────
    per_opts = {"1mo": ("شهر" if use_ar else "1 Month"),
                "3mo": ("3 أشهر" if use_ar else "3 Months"),
                "6mo": ("6 أشهر" if use_ar else "6 Months"),
                "1y":  ("سنة"  if use_ar else "1 Year")}
    og_per = st.radio("", list(per_opts.values()), horizontal=True, key="og_period",
                      label_visibility="collapsed")
    og_per_key = [k for k, v in per_opts.items() if v == og_per][0]

    og = get_oil_gold_data(og_per_key)

    if not og:
        st.warning("⚠️ " + ("تعذّر تحميل بيانات النفط والذهب" if use_ar else
                             "Could not load Oil & Gold data"))
    else:
        corr      = og["corr"]
        n_days    = og["n"]
        gold_chg  = (og["gold"][-1] - og["gold"][0]) / og["gold"][0] * 100
        oil_chg   = (og["oil"][-1]  - og["oil"][0])  / og["oil"][0]  * 100
        diverging = abs(gold_chg - oil_chg) > 15   # unusual divergence
        corr_col  = C["green"] if corr > 0.5 else (C["red"] if corr < -0.3 else C["muted"])

        # ── Correlation summary ───────────────────────────────────────────────
        oc1, oc2, oc3, oc4 = st.columns(4)
        for col, lbl, val, vc in [
            (oc1, L["og_corr"],             f"{corr:+.3f}",       corr_col),
            (oc2, ("الذهب " if use_ar else "Gold ") + og_per,
                  f"{gold_chg:+.2f}%",      C["green"] if gold_chg>=0 else C["red"]),
            (oc3, ("النفط " if use_ar else "Oil ") + og_per,
                  f"{oil_chg:+.2f}%",       C["green"] if oil_chg>=0 else C["red"]),
            (oc4, ("أيام البيانات" if use_ar else "Data Days"), str(n_days), C["muted"]),
        ]:
            with col:
                st.markdown(f"""
                <div class='stat-card'>
                  <div class='stat-label'>{lbl}</div>
                  <div class='stat-value' style='color:{vc};font-size:20px;'>{val}</div>
                </div>""", unsafe_allow_html=True)

        if diverging:
            st.warning("⚠️ " + L["og_diverge"] +
                       f" — {'الذهب' if use_ar else 'Gold'}: {gold_chg:+.1f}% · "
                       f"{'النفط' if use_ar else 'Oil'}: {oil_chg:+.1f}%")

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Dual price chart ──────────────────────────────────────────────────
        st.markdown(f"<div class='section-label'>"
                    f"{'أسعار الذهب والنفط' if use_ar else 'Gold & Oil Prices'}</div>",
                    unsafe_allow_html=True)
        import plotly.graph_objects as go_og
        from plotly.subplots import make_subplots as _msp
        fig_og = _msp(specs=[[{"secondary_y": True}]])
        fig_og.add_trace(go_og.Scatter(
            x=og["dates"], y=og["gold"],
            name=("الذهب $/أوقية" if use_ar else "Gold $/oz"),
            line=dict(color=C["gold"], width=2),
            fill="tozeroy", fillcolor=rgba(C["gold"], 0.06),
        ), secondary_y=False)
        fig_og.add_trace(go_og.Scatter(
            x=og["dates"], y=og["oil"],
            name=("النفط $/برميل" if use_ar else "Oil $/bbl"),
            line=dict(color="#4DA6FF", width=2, dash="dot"),
        ), secondary_y=True)
        fig_og.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=C["text"], size=10), height=300,
            margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(orientation="h", y=-0.2),
            xaxis=dict(gridcolor=C["border2"]),
        )
        fig_og.update_yaxes(gridcolor=C["border2"], secondary_y=False,
                            title_text=("الذهب" if use_ar else "Gold"))
        fig_og.update_yaxes(gridcolor="rgba(0,0,0,0)", secondary_y=True,
                            title_text=("النفط" if use_ar else "Oil"))
        st.plotly_chart(fig_og, use_container_width=True, config={"displayModeBar": False})

        # ── Rolling 20-day correlation chart ──────────────────────────────────
        st.markdown(f"<div class='section-label'>"
                    f"{'الارتباط المتحرك (20 يوم)' if use_ar else 'Rolling 20-Day Correlation'}</div>",
                    unsafe_allow_html=True)
        roll_vals = og["roll_corr"]
        roll_cols = [C["green"] if v > 0.5 else (C["red"] if v < -0.3 else C["muted"])
                     for v in roll_vals]
        fig_rc = go_og.Figure()
        fig_rc.add_shape(type="rect", x0=og["dates"][0], x1=og["dates"][-1],
                         y0=0.5, y1=1.0,
                         fillcolor=rgba(C["green"], 0.06), line_width=0)
        fig_rc.add_shape(type="rect", x0=og["dates"][0], x1=og["dates"][-1],
                         y0=-0.3, y1=-1.0,
                         fillcolor=rgba(C["red"], 0.06), line_width=0)
        fig_rc.add_trace(go_og.Scatter(
            x=og["dates"], y=roll_vals,
            mode="lines",
            line=dict(color=C["gold"], width=2),
            fill="tozeroy",
            fillcolor=rgba(C["gold"], 0.08),
            name=L["og_corr"],
        ))
        fig_rc.add_hline(y=0, line_color=C["border"], line_width=1)
        fig_rc.add_hline(y=0.5, line_color=C["green"], line_width=1,
                         line_dash="dot",
                         annotation_text=("ارتباط قوي" if use_ar else "Strong correlation"),
                         annotation_font_color=C["green"], annotation_font_size=9)
        fig_rc.add_hline(y=-0.3, line_color=C["red"], line_width=1,
                         line_dash="dot",
                         annotation_text=("ارتباط عكسي" if use_ar else "Inverse correlation"),
                         annotation_font_color=C["red"], annotation_font_size=9)
        fig_rc.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=C["text"], size=10), height=240,
            margin=dict(l=0, r=0, t=10, b=0),
            yaxis=dict(gridcolor=C["border2"], range=[-1.1, 1.1]),
            xaxis=dict(gridcolor=C["border2"]),
            showlegend=False,
        )
        st.plotly_chart(fig_rc, use_container_width=True, config={"displayModeBar": False})

        # ── Interpretation ────────────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        if corr > 0.6:
            interp_ar = "الارتباط قوي جداً — النفط والذهب يتحركان معاً. ارتفاع النفط يدعم الذهب عبر التضخم."
            interp_en = "Very strong correlation — oil and gold moving together. Rising oil supports gold via inflation channel."
            interp_col = C["green"]
        elif corr > 0.3:
            interp_ar = "ارتباط معتدل — النفط يؤثر على الذهب عبر التضخم لكن عوامل أخرى تلعب دوراً."
            interp_en = "Moderate correlation — oil influences gold via inflation, but other factors are also at play."
            interp_col = C["gold"]
        elif corr < -0.2:
            interp_ar = "ارتباط عكسي — الدولار القوي يضغط على كليهما بطرق مختلفة. إشارة تحذير."
            interp_en = "Inverse correlation — strong USD may be pressing both differently. Watch for divergence signals."
            interp_col = C["red"]
        else:
            interp_ar = "ارتباط ضعيف — النفط والذهب يتحركان بشكل مستقل حالياً. لا توجد إشارة واضحة."
            interp_en = "Weak correlation — oil and gold moving independently. No clear directional signal."
            interp_col = C["muted"]

        st.markdown(f"""
        <div style='background:{C["card"]};border:1px solid {interp_col}44;
                    border-left:3px solid {interp_col};border-radius:6px;
                    padding:14px 18px;{"direction:rtl;text-align:right;" if use_ar else ""}'>
          <div style='font-size:13px;font-weight:700;color:{interp_col};margin-bottom:6px;'>
            {L["og_signal"]}: {corr:+.3f}</div>
          <div style='font-size:12px;color:{C["text"]};line-height:1.7;'>
            {"🛢️ " + interp_ar if use_ar else "🛢️ " + interp_en}</div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── How oil → gold works explainer ───────────────────────────────────
        st.markdown(f"<div class='section-label'>"
                    f"{'كيف يؤثر النفط على الذهب؟' if use_ar else 'How does Oil affect Gold?'}</div>",
                    unsafe_allow_html=True)
        channels = [
            ("🔥", "التضخم" if use_ar else "Inflation Channel",
             ("النفط الغالي يرفع تكلفة كل شيء → تضخم أعلى → الذهب ملاذ آمن من التضخم → الطلب يرتفع" if use_ar else
              "Expensive oil raises the cost of everything → higher inflation → gold as inflation hedge → demand rises")),
            ("💵", "قيمة الدولار" if use_ar else "USD Channel",
             ("النفط يُسعَّر بالدولار. دولار أضعف = نفط أغلى = ذهب أغلى. هذه العلاقة تجعل الذهب والنفط يتحركان معاً." if use_ar else
              "Oil is priced in USD. Weaker dollar = pricier oil = pricier gold. This USD link makes oil and gold move together.")),
            ("⚡", "الطاقة الجيوسياسية" if use_ar else "Geopolitical Energy",
             ("أزمات النفط غالباً ما تكون أزمات جيوسياسية. نفس الصراعات التي ترفع النفط ترفع الطلب على الذهب كملاذ آمن." if use_ar else
              "Oil crises are usually geopolitical crises. The same conflicts that spike oil also drive safe-haven gold demand.")),
        ]
        for icon, title, desc in channels:
            st.markdown(f"""
            <div style='display:flex;gap:12px;align-items:flex-start;
                        margin-bottom:10px;{"direction:rtl;" if use_ar else ""}'>
              <div style='font-size:22px;flex-shrink:0;'>{icon}</div>
              <div>
                <div style='font-size:13px;font-weight:700;color:{C["gold_pale"]};'>{title}</div>
                <div style='font-size:11px;color:{C["muted"]};line-height:1.7;
                            margin-top:3px;'>{desc}</div>
              </div>
            </div>""", unsafe_allow_html=True)



# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: SHOP PRICE BOARD 🏪
# ═══════════════════════════════════════════════════════════════════════════════
elif nav == L["nav_shopboard"]:
    use_ar = is_rtl()
    ph("🏪  " + L["nav_shopboard"].replace("🏪  ",""), L["shop_display"])

    # ── Settings panel ────────────────────────────────────────────────────────
    with st.expander("⚙️ " + ("إعدادات اللوحة" if use_ar else "Board Settings"), expanded=False):
        sb1, sb2, sb3, sb4 = st.columns(4)
        with sb1:
            shop_name = st.text_input(L["shop_name"],
                value=st.session_state.get("shop_name",""),
                key="sb_name", placeholder=("محل الذهب" if use_ar else "Gold Shop"))
            st.session_state["shop_name"] = shop_name
        with sb2:
            spread = st.number_input(L["shop_spread"] + " %",
                min_value=0.0, max_value=20.0,
                value=float(st.session_state.get("shop_spread", 2.0)),
                step=0.1, key="sb_spread", format="%.1f")
            st.session_state["shop_spread"] = spread
        with sb3:
            disp_curr = st.selectbox(L["shop_currency"],
                ["USD","JOD","SAR","AED","KWD","QAR","BHD","EGP","GBP","EUR","TRY"],
                index=["USD","JOD","SAR","AED","KWD","QAR","BHD","EGP","GBP","EUR","TRY"]
                      .index(st.session_state.get("shop_currency","USD")),
                key="sb_curr")
            st.session_state["shop_currency"] = disp_curr
        with sb4:
            show_purities = st.checkbox(L["shop_purity"], value=True, key="sb_purities")

    # ── Compute live buy/sell ─────────────────────────────────────────────────
    _FX_SHOP = {"USD":1.0,"JOD":0.709,"SAR":3.75,"AED":3.6725,"KWD":0.307,
                "QAR":3.64,"BHD":0.376,"EGP":50.9,"GBP":0.787,"EUR":0.926,"TRY":38.0}
    fx_rate   = _FX_SHOP.get(disp_curr, 1.0)
    gram_spot = g_ref / 31.1035               # USD per gram 24K
    buy_g24   = gram_spot * fx_rate           # shop buys AT spot
    sell_g24  = gram_spot * fx_rate * (1 + spread/100)   # shop sells with margin
    _name_str = st.session_state.get("shop_name","") or ("لوحة الأسعار" if use_ar else "Price Board")

    # ── Purities table ────────────────────────────────────────────────────────
    PURITIES = [
        ("24K", 1.000), ("22K", 0.9167), ("21K", 0.875),
        ("18K", 0.750), ("14K", 0.5833), ("10K", 0.4167),
    ]
    visible_purities = PURITIES if show_purities else [("24K",1.0),("21K",0.875),("18K",0.750)]

    # ── Full-screen price board ───────────────────────────────────────────────
    board_rows = ""
    for karat, ratio in visible_purities:
        buy_k  = buy_g24  * ratio
        sell_k = sell_g24 * ratio
        board_rows += f"""
        <tr>
          <td style='padding:14px 20px;font-size:22px;font-weight:900;
                     color:#F5C842;font-family:monospace;'>{karat}</td>
          <td style='padding:14px 20px;font-family:monospace;font-size:20px;
                     color:#52D98A;font-weight:700;text-align:right;'>
            {buy_k:,.3f}</td>
          <td style='padding:14px 20px;font-family:monospace;font-size:20px;
                     color:#FAE5A0;font-weight:700;text-align:right;'>
            {sell_k:,.3f}</td>
        </tr>"""

    board_html = f"""
<!DOCTYPE html><html><head><meta charset="UTF-8">
<meta http-equiv="refresh" content="60">
<style>
  * {{ margin:0;padding:0;box-sizing:border-box; }}
  body {{ background:{C["bg"]};color:{C["text"]};font-family:'Cairo',serif;
          min-height:100vh; display:flex; flex-direction:column;
          align-items:center; justify-content:center; padding:20px; }}
  .board {{ width:100%;max-width:700px;background:{C["card2"]};
            border:1px solid {C["gold"]}55;border-radius:12px;overflow:hidden; }}
  .board-header {{ background:linear-gradient(135deg,{C["card"]},{C["gold"]}22);
                   padding:20px 28px;display:flex;justify-content:space-between;
                   align-items:center; border-bottom:1px solid {C["gold"]}44; }}
  .shop-name {{ font-family:'{C["font_h"]}',serif;font-size:24px;font-weight:900;
                color:{C["gold_hi"]};letter-spacing:.05em; }}
  .live-price {{ text-align:right; }}
  .spot-lbl {{ font-size:10px;color:{C["muted"]};letter-spacing:.2em; }}
  .spot-val {{ font-family:monospace;font-size:28px;font-weight:900;
               color:{C["gold_hi"]}; }}
  table {{ width:100%;border-collapse:collapse; }}
  thead tr {{ background:{C["card"]};border-bottom:2px solid {C["gold"]}44; }}
  th {{ padding:12px 20px;font-size:11px;letter-spacing:.15em;
        color:{C["muted"]};font-weight:600; }}
  tbody tr:nth-child(even) {{ background:{C["bg2"]}; }}
  tbody tr:hover {{ background:{C["gold"]}11; }}
  .footer {{ padding:10px 20px;font-size:10px;color:{C["dim"]};
             display:flex;justify-content:space-between;
             border-top:1px solid {C["border2"]}; }}
  @keyframes pulse {{ 0%,100%{{opacity:1}} 50%{{opacity:.6}} }}
  .live-dot {{ width:8px;height:8px;background:{C["green"]};border-radius:50%;
               display:inline-block;animation:pulse 2s infinite;margin-right:6px; }}
</style>
</head>
<body>
<div class="board">
  <div class="board-header">
    <div class="shop-name">☽ {_name_str}</div>
    <div class="live-price">
      <div class="spot-lbl">{"سعر الأونصة / OZ" if use_ar else "SPOT / OZ"}</div>
      <div class="spot-val">${g_ref:,.2f}</div>
      <div style="font-size:10px;color:{C['muted']};margin-top:2px;">
        <span class="live-dot"></span>LIVE · {disp_curr}</div>
    </div>
  </div>
  <table>
    <thead><tr>
      <th style='text-align:{"right" if use_ar else "left"};'>
        {"العيار" if use_ar else "PURITY"}</th>
      <th style='text-align:right;color:{C["green"]};'>
        {"🟢 شراء / جرام" if use_ar else "🟢 BUY / gram"}</th>
      <th style='text-align:right;color:{C["gold_pale"]};'>
        {"🪙 بيع / جرام" if use_ar else "🪙 SELL / gram"}</th>
    </tr></thead>
    <tbody>{board_rows}</tbody>
  </table>
  <div class="footer">
    <span>{"هامش الصرف:" if use_ar else "Spread:"} {spread:.1f}%</span>
    <span id="ts" style="color:{C['dim']};font-size:9px;"></span>
  </div>
</div>
<script>
  var el = document.getElementById("ts");
  function upd(){{ if(el) el.textContent = new Date().toLocaleTimeString(); }}
  upd(); setInterval(upd, 1000);
</script>
</body></html>"""

    st.components.v1.html(board_html, height=520, scrolling=False)

    # ── Embed code ────────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("🔌 " + L["shop_embed"]):
        st.info("💡 " + ("الكود يتحدث تلقائياً كل 60 ثانية · انسخه والصقه في موقعك" if use_ar
                         else "Auto-updates every 60s · Copy and paste into your website"))
        embed_board = (
            '<iframe src="your-mansa-url?page=shopboard"'
            ' width="720" height="520" frameborder="0"'
            ' style="border-radius:12px;overflow:hidden;"></iframe>'
        )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: INVOICE CALCULATOR 🧾
# ═══════════════════════════════════════════════════════════════════════════════
elif nav == L["nav_invoice"]:
    use_ar = is_rtl()
    ph("🧾  " + L["nav_invoice"].replace("🧾  ",""),
       ("احسب فاتورة البيع بدقة · بما في ذلك الضريبة والصنعة" if use_ar else
        "Calculate precise sale invoice · including VAT and making charges"))

    if "invoices" not in st.session_state:
        st.session_state["invoices"] = []

    _PURITY_RATIOS = {"24K":1.0,"22K":0.9167,"21K":0.875,"18K":0.750,"14K":0.5833,"10K":0.4167}
    _FX_INV = {"USD":1.0,"JOD":0.709,"SAR":3.75,"AED":3.6725,"KWD":0.307,
               "QAR":3.64,"BHD":0.376,"EGP":50.9,"GBP":0.787,"EUR":0.926,"TRY":38.0}

    # ── Input form ────────────────────────────────────────────────────────────
    ic1, ic2 = st.columns(2, gap="large")
    with ic1:
        st.markdown(f"<div class='section-label'>{'بيانات المقطوعة' if use_ar else 'Item Details'}</div>",
                    unsafe_allow_html=True)
        inv_weight  = st.number_input(L["inv_weight"], min_value=0.01, value=10.0,
                                      step=0.5, format="%.2f", key="inv_w")
        inv_purity  = st.selectbox(L["inv_purity"], list(_PURITY_RATIOS.keys()),
                                   index=2, key="inv_p")  # 21K default
        inv_curr    = st.selectbox(("عملة الفاتورة" if use_ar else "Invoice Currency"),
                                   list(_FX_INV.keys()), key="inv_curr")
    with ic2:
        st.markdown(f"<div class='section-label'>{'الرسوم' if use_ar else 'Charges'}</div>",
                    unsafe_allow_html=True)
        inv_making  = st.number_input(L["inv_making"], min_value=0.0, value=5.0,
                                      step=0.5, format="%.2f", key="inv_m")
        inv_vat     = st.number_input(L["inv_vat"], min_value=0.0, max_value=30.0,
                                      value=16.0, step=0.5, format="%.1f", key="inv_v")
        inv_qty     = st.number_input(("الكمية" if use_ar else "Quantity"),
                                      min_value=1, value=1, step=1, key="inv_qty")

    # ── Compute ───────────────────────────────────────────────────────────────
    fx           = _FX_INV.get(inv_curr, 1.0)
    ratio        = _PURITY_RATIOS.get(inv_purity, 0.875)
    gram_spot    = g_ref / 31.1035             # USD per gram 24K
    gram_purity  = gram_spot * ratio           # USD per gram at chosen purity
    gold_val_usd = gram_purity * inv_weight * inv_qty
    gold_val_lc  = gold_val_usd * fx
    making_val   = inv_making * inv_weight * inv_qty * fx
    subtotal     = gold_val_lc + making_val
    vat_val      = subtotal * inv_vat / 100
    total        = subtotal + vat_val

    # ── Receipt display ───────────────────────────────────────────────────────
    rtl_s = "direction:rtl;text-align:right;" if use_ar else ""
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='background:{C["card2"]};border:1px solid {C["gold"]}44;
                border-radius:10px;padding:24px 28px;max-width:480px;margin:0 auto;
                font-family:{C["font_b"]},serif;{rtl_s}'>
      <div style='text-align:center;font-size:22px;font-weight:900;
                  color:{C["gold_hi"]};margin-bottom:16px;letter-spacing:.05em;'>
        ☽ {"فاتورة" if use_ar else "INVOICE"}</div>
      <div style='font-size:12px;color:{C["muted"]};text-align:center;margin-bottom:18px;'>
        {inv_weight:.2f}g · {inv_purity} · {inv_qty} {"قطعة" if use_ar else "pcs"} · {inv_curr}
      </div>
      <hr style='border-color:{C["border2"]};margin:0 0 16px;'>
      {"".join(f"""
      <div style='display:flex;justify-content:space-between;margin-bottom:10px;
                  font-size:13px;{"flex-direction:row-reverse;" if use_ar else ""}'>
        <span style='color:{C["muted"]};'>{lbl}</span>
        <span style='font-family:monospace;font-weight:700;color:{vc};'>{val}</span>
      </div>""" for lbl, val, vc in [
          (L["inv_gold_val"],  f"{gold_val_lc:,.3f} {inv_curr}",   C["gold_pale"]),
          (L["inv_making_val"],f"{making_val:,.3f} {inv_curr}",    C["muted"]),
          (f"Sub-total",       f"{subtotal:,.3f} {inv_curr}",      C["text"]),
          (f"{L['inv_vat_val']} ({inv_vat:.0f}%)", f"{vat_val:,.3f} {inv_curr}", C["muted"]),
      ])}
      <hr style='border-color:{C["gold"]}44;margin:12px 0;'>
      <div style='display:flex;justify-content:space-between;
                  {"flex-direction:row-reverse;" if use_ar else ""}'>
        <span style='font-size:14px;font-weight:900;color:{C["gold"]};'>{L["inv_total"]}</span>
        <span style='font-family:monospace;font-size:22px;font-weight:900;
                     color:{C["gold_hi"]};'>{total:,.2f} {inv_curr}</span>
      </div>
      <div style='text-align:center;font-size:10px;color:{C["dim"]};margin-top:14px;'>
        {"سعر الذهب: " if use_ar else "Gold spot: "}${g_ref:,.2f}/oz · 
        {gram_purity*fx:,.3f} {inv_curr}/g ({inv_purity})
      </div>
    </div>""", unsafe_allow_html=True)

    # ── Save / history ────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    sv1, sv2 = st.columns([2,3])
    with sv1:
        if st.button("💾 " + L["inv_save"], type="primary", use_container_width=True, key="inv_save_btn"):
            st.session_state["invoices"].insert(0, {
                "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                "weight": inv_weight, "purity": inv_purity, "qty": inv_qty,
                "total": round(total,2), "currency": inv_curr,
                "spot": g_ref,
            })
            _sb_save()
            st.success("✅ " + ("تم حفظ الفاتورة" if use_ar else "Invoice saved"))

    invoices = st.session_state.get("invoices", [])
    if invoices:
        with st.expander(f"📋 {L['inv_history']} ({len(invoices)})"):
            for inv in invoices[:20]:
                st.markdown(f"""
                <div style='display:flex;justify-content:space-between;
                            padding:8px 0;border-bottom:1px solid {C["border2"]};
                            font-size:12px;{"direction:rtl;" if use_ar else ""}'>
                  <span style='color:{C["muted"]};'>{inv["date"]}</span>
                  <span style='color:{C["text"]};'>{inv["weight"]}g {inv["purity"]} ×{inv["qty"]}</span>
                  <span style='color:{C["gold_hi"]};font-family:monospace;font-weight:700;'>
                    {inv["total"]:,.2f} {inv["currency"]}</span>
                </div>""", unsafe_allow_html=True)
            if st.button("🗑️ " + L["inv_clear"], key="inv_clear_btn"):
                st.session_state["invoices"] = []
                _sb_save()
                st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: PRODUCTION COST CALCULATOR 🏭
# ═══════════════════════════════════════════════════════════════════════════════
elif nav == L["nav_production"]:
    use_ar = is_rtl()
    ph("🏭  " + L["nav_production"].replace("🏭  ",""),
       ("احسب تكلفة إنتاج المجوهرات بدقة · النفايات · العمالة · المواد" if use_ar else
        "Precise jewellery production costing · wastage · labour · materials"))

    _KARAT_GOLD_PCT = {"24K":100.0,"22K":91.67,"21K":87.5,"18K":75.0,"14K":58.33,"10K":41.67}
    _FX_PR = {"USD":1.0,"JOD":0.709,"SAR":3.75,"AED":3.6725,"KWD":0.307,
              "QAR":3.64,"BHD":0.376,"EGP":50.9}

    gram_24k_usd = g_ref / 31.1035

    pc1, pc2 = st.columns(2, gap="large")
    with pc1:
        st.markdown(f"<div class='section-label'>{'المواد الخام' if use_ar else 'Raw Materials'}</div>",
                    unsafe_allow_html=True)
        prod_purity  = st.selectbox(L["prod_recipe"], list(_KARAT_GOLD_PCT.keys()),
                                    index=2, key="pc_purity")
        prod_weight  = st.number_input(L["prod_weight"], min_value=0.1, value=10.0,
                                       step=0.5, format="%.2f", key="pc_weight")
        prod_wastage = st.number_input(L["prod_wastage"], min_value=0.0, max_value=30.0,
                                       value=3.0, step=0.5, format="%.1f", key="pc_waste")
        prod_curr    = st.selectbox(("عملة التكلفة" if use_ar else "Cost Currency"),
                                    list(_FX_PR.keys()), key="pc_curr")
    with pc2:
        st.markdown(f"<div class='section-label'>{'العمالة والمصاريف' if use_ar else 'Labour & Overhead'}</div>",
                    unsafe_allow_html=True)
        prod_labour   = st.number_input(L["prod_labour"], min_value=0.0, value=3.0,
                                        step=0.5, format="%.2f", key="pc_labour")
        prod_overhead = st.number_input(L["prod_overhead"],
                                        min_value=0.0, value=20.0, step=5.0,
                                        format="%.2f", key="pc_overhead",
                                        help=("مصاريف ثابتة لكل قطعة (إيجار، كهرباء، إلخ)" if use_ar
                                              else "Fixed cost per piece (rent, utilities, etc.)"))
        prod_margin   = st.number_input(("هامش الربح المطلوب %" if use_ar else "Target Margin %"),
                                        min_value=0.0, max_value=200.0,
                                        value=30.0, step=5.0, key="pc_margin")
        prod_qty      = st.number_input(L["prod_qty"], min_value=1, value=1,
                                        step=1, key="pc_qty")

    # ── Calculations ──────────────────────────────────────────────────────────
    fx          = _FX_PR.get(prod_curr, 1.0)
    gold_pct    = _KARAT_GOLD_PCT[prod_purity] / 100
    raw_needed  = prod_weight * (1 + prod_wastage/100)   # grams of alloy needed
    gold_needed = raw_needed * gold_pct                  # pure gold grams needed
    gold_cost   = gold_needed * gram_24k_usd * fx        # in local currency
    labour_cost = prod_labour * prod_weight * fx
    total_cost  = gold_cost + labour_cost + prod_overhead
    min_price   = total_cost
    suggest_price = total_cost / (1 - prod_margin/100) if prod_margin < 100 else total_cost * 2
    batch_cost  = total_cost * prod_qty
    batch_suggest = suggest_price * prod_qty

    # ── Results ───────────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    r1, r2, r3, r4 = st.columns(4)
    for col, lbl, val, vc in [
        (r1, L["prod_gold_cost"],   f"{gold_cost:,.2f} {prod_curr}",   C["gold"]),
        (r2, L["prod_total_cost"],  f"{total_cost:,.2f} {prod_curr}",  C["gold_pale"]),
        (r3, L["prod_min_price"],   f"{min_price:,.2f} {prod_curr}",   C["red"]),
        (r4, L["prod_suggest"],     f"{suggest_price:,.2f} {prod_curr}", C["green"]),
    ]:
        with col:
            st.markdown(f"""
            <div class='stat-card'>
              <div class='stat-label'>{lbl}</div>
              <div class='stat-value' style='color:{vc};font-size:15px;'>{val}</div>
            </div>""", unsafe_allow_html=True)

    # Cost breakdown pie
    import plotly.graph_objects as _go_prod
    fig_prod = _go_prod.Figure(_go_prod.Pie(
        labels=[("ذهب خام" if use_ar else "Raw Gold"),
                ("عمالة" if use_ar else "Labour"),
                ("مصاريف" if use_ar else "Overhead")],
        values=[gold_cost, labour_cost, prod_overhead],
        hole=0.5,
        marker=dict(colors=[C["gold"], C["green"], C["muted"]],
                    line=dict(color=C["bg"], width=2)),
        textfont=dict(size=10, color=C["text"]),
    ))
    fig_prod.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", height=240,
        margin=dict(l=0,r=0,t=10,b=0),
        font=dict(color=C["text"]),
        showlegend=True,
        legend=dict(orientation="h",y=-0.15),
        annotations=[dict(text=f"{prod_margin:.0f}%\n{'ربح' if use_ar else 'margin'}",
                          font_size=13, font_color=C["gold_hi"], showarrow=False)],
    )
    st.plotly_chart(fig_prod, use_container_width=True, config={"displayModeBar":False})

    if prod_qty > 1:
        st.markdown(f"""
        <div style='background:{C["card"]};border:1px solid {C["green"]}44;border-radius:6px;
                    padding:14px 18px;text-align:center;{"direction:rtl;" if use_ar else ""}'>
          <div style='font-size:11px;color:{C["muted"]};'>
            {L["prod_batch"]} × {prod_qty}</div>
          <div style='font-family:monospace;font-size:20px;font-weight:900;
                      color:{C["green"]};margin-top:4px;'>
            {batch_cost:,.2f} {prod_curr} → {batch_suggest:,.2f} {prod_curr}</div>
        </div>""", unsafe_allow_html=True)

    # ── Alloy recipe table ────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"<div class='section-label'>{'جدول وصفات السبائك' if use_ar else 'Alloy Recipe Reference'}</div>",
                unsafe_allow_html=True)
    alloy_data = [
        ("24K","1000/1000","100% Au","—","—"),
        ("22K","916/1000","91.7% Au","5% Ag","3.3% Cu"),
        ("21K","875/1000","87.5% Au","6.5% Ag","6% Cu"),
        ("18K","750/1000","75% Au","12.5% Ag","12.5% Cu"),
        ("14K","583/1000","58.3% Au","20% Ag","21.7% Cu"),
        ("10K","417/1000","41.7% Au","—","58.3% Cu+Zn"),
    ]
    # Build alloy table without nested f-strings
    _th_style  = f"padding:8px 12px;color:{C['gold']};text-align:left;"
    _td_style  = f"padding:9px 12px;color:{C['text']};"
    _hdrs = ["العيار","الحصة","الذهب","الفضة","النحاس"] if use_ar else ["Karat","Millesimal","Gold","Silver","Copper"]
    _ths  = "".join(f"<th style='{_th_style}'>{h}</th>" for h in _hdrs)
    _rows = ""
    for _i, _row in enumerate(alloy_data):
        _bg  = f"background:{C['card']};" if _i % 2 == 0 else ""
        _tds = "".join(f"<td style='{_td_style}'>{v}</td>" for v in _row)
        _rows += f"<tr style='border-bottom:1px solid {C['border']}22;{_bg}'>{_tds}</tr>"
    alloy_html = (
        f"<table style='width:100%;border-collapse:collapse;font-size:12px;"
        f"font-family:{C['font_m']},monospace;'>"
        f"<thead><tr style='border-bottom:2px solid {C['gold']}44;'>{_ths}</tr></thead>"
        f"<tbody>{_rows}</tbody></table>"
    )
    st.markdown(alloy_html, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: FAIR PRICE CHECKER ⚖️
# ═══════════════════════════════════════════════════════════════════════════════
elif nav == L["nav_fairprice"]:
    use_ar = is_rtl()
    ph("⚖️  " + L["nav_fairprice"].replace("⚖️  ",""),
       ("هل السعر المعروض عليك عادل؟ تحقق في ثانية واحدة" if use_ar else
        "Is the quoted price fair? Check in one second"))

    _PURITY_RATIOS_FP = {"24K":1.0,"22K":0.9167,"21K":0.875,"18K":0.750,"14K":0.5833,"10K":0.4167}
    _FX_FP = {"USD":1.0,"JOD":0.709,"SAR":3.75,"AED":3.6725,"KWD":0.307,
              "QAR":3.64,"BHD":0.376,"EGP":50.9,"GBP":0.787,"EUR":0.926}

    fp1, fp2 = st.columns(2, gap="large")
    with fp1:
        st.markdown(f"<div class='section-label'>{'السعر المعروض عليك' if use_ar else 'What You Were Quoted'}</div>",
                    unsafe_allow_html=True)
        fp_quoted  = st.number_input(L["fp_quoted"], min_value=0.01, value=500.0,
                                     step=1.0, format="%.2f", key="fp_q")
        fp_curr    = st.selectbox(L["fp_currency"], list(_FX_FP.keys()),
                                  index=1, key="fp_curr")   # JOD default
    with fp2:
        st.markdown(f"<div class='section-label'>{'تفاصيل المقطوعة' if use_ar else 'Piece Details'}</div>",
                    unsafe_allow_html=True)
        fp_weight  = st.number_input(L["fp_weight"], min_value=0.1, value=10.0,
                                     step=0.5, format="%.2f", key="fp_w")
        fp_purity  = st.selectbox(L["fp_purity"], list(_PURITY_RATIOS_FP.keys()),
                                  index=2, key="fp_p")      # 21K default
        fp_making  = st.number_input(L["fp_making_est"],
                                     min_value=0.0, value=5.0, step=1.0,
                                     key="fp_making",
                                     help=("تقدير أجر الصنعة المتوقع" if use_ar
                                           else "Estimated normal making charge"))

    # ── Compute fair value ────────────────────────────────────────────────────
    fx         = _FX_FP.get(fp_curr, 1.0)
    ratio      = _PURITY_RATIOS_FP.get(fp_purity, 0.875)
    gram_spot  = g_ref / 31.1035
    spot_val   = gram_spot * ratio * fp_weight * fx   # pure spot value in local currency
    making_val = fp_making * fp_weight * fx
    fair_total = spot_val + making_val

    diff       = fp_quoted - fair_total
    diff_pct   = (diff / fair_total * 100) if fair_total > 0 else 0

    if abs(diff_pct) <= 8:
        verdict = L["fp_verdict_fair"]
        v_col   = C["green"]
        v_icon  = "✅"
    elif diff_pct > 8:
        verdict = L["fp_verdict_high"]
        v_col   = C["red"]
        v_icon  = "⚠️"
    else:
        verdict = L["fp_verdict_low"]
        v_col   = C["gold"]
        v_icon  = "🤔"

    st.markdown("<br>", unsafe_allow_html=True)
    # Verdict card
    st.markdown(f"""
    <div style='background:{C["card2"]};border:2px solid {v_col}66;border-radius:10px;
                padding:24px 28px;text-align:center;'>
      <div style='font-size:48px;margin-bottom:10px;'>{v_icon}</div>
      <div style='font-size:22px;font-weight:900;color:{v_col};margin-bottom:8px;'>
        {verdict}</div>
      <div style='font-family:monospace;font-size:16px;color:{C["text"]};'>
        {"قيمة عادلة:" if use_ar else "Fair value:"} {fair_total:,.2f} {fp_curr}
      </div>
      <div style='font-size:13px;margin-top:8px;color:{v_col};font-weight:700;'>
        {f"{L['fp_overpay']}: +{abs(diff):,.2f} {fp_curr} ({abs(diff_pct):.1f}%)" if diff > 0
         else f"{L['fp_saving']}: {abs(diff):,.2f} {fp_curr} ({abs(diff_pct):.1f}%)"}
      </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    fa1, fa2, fa3 = st.columns(3)
    for col, lbl, val, vc in [
        (fa1, L["fp_spot_val"],    f"{spot_val:,.2f} {fp_curr}",   C["gold_pale"]),
        (fa2, L["fp_making_est"],  f"{making_val:,.2f} {fp_curr}", C["muted"]),
        (fa3, ("المعروض عليك" if use_ar else "Quoted"),
              f"{fp_quoted:,.2f} {fp_curr}", v_col),
    ]:
        with col:
            st.markdown(f"""
            <div class='stat-card'>
              <div class='stat-label'>{lbl}</div>
              <div class='stat-value' style='color:{vc};font-size:15px;'>{val}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.info(("💡 " + ("السعر العادل = قيمة الذهب الخام + الصنعة المعقولة. الفرق المقبول عادةً لا يتجاوز ±8%." if use_ar
             else "Fair price = raw gold value + reasonable making charge. Acceptable variance is typically ±8%.") +
             f" | {L['disclaimer']}"))


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: PIECE PRICING STUDIO 💍
# ═══════════════════════════════════════════════════════════════════════════════
elif nav == L["nav_piecepricing"]:
    use_ar = is_rtl()
    ph("💍  " + L["nav_piecepricing"].replace("💍  ",""),
       ("سعّر تصميمك بدقة · ذهب + أحجار + عمالة + هامش ربح" if use_ar else
        "Price your design precisely · gold + gems + labour + margin"))

    if "saved_designs" not in st.session_state:
        st.session_state["saved_designs"] = []
    if "piece_gems" not in st.session_state:
        st.session_state["piece_gems"] = []

    _PURITY_PP = {"24K":1.0,"22K":0.9167,"21K":0.875,"18K":0.750,"14K":0.5833,"10K":0.4167}
    _GEM_PRICES = {  # approximate USD per carat (mid-quality)
        ("Diamond / الماس",):           1500,
        ("Ruby / الياقوت",):             800,
        ("Emerald / الزمرد",):           600,
        ("Sapphire / الياقوت الأزرق",):  500,
        ("Amethyst / الجمشت",):           20,
        ("Pearl / اللؤلؤ",):              80,
        ("Turquoise / الفيروز",):         15,
        ("Zircon / الزركون",):             5,
    }
    # Flatten gem names for selectbox
    _GEM_NAMES = [list(k)[0] for k in _GEM_PRICES.keys()]
    _GEM_PX_LIST = list(_GEM_PRICES.values())
    _FX_PP = {"USD":1.0,"JOD":0.709,"SAR":3.75,"AED":3.6725,"KWD":0.307,
              "QAR":3.64,"BHD":0.376,"EGP":50.9}

    pp1, pp2 = st.columns(2, gap="large")
    with pp1:
        st.markdown(f"<div class='section-label'>{'مكونات الذهب' if use_ar else 'Gold Component'}</div>",
                    unsafe_allow_html=True)
        pp_gold_g   = st.number_input(L["piece_gold_g"], min_value=0.0, value=5.0,
                                      step=0.5, format="%.2f", key="pp_g")
        pp_purity   = st.selectbox(L["piece_purity"], list(_PURITY_PP.keys()),
                                   index=2, key="pp_p")
        pp_wastage  = st.number_input(("هدر %" if use_ar else "Wastage %"),
                                      min_value=0.0, max_value=15.0,
                                      value=2.0, step=0.5, key="pp_waste")
        pp_curr     = st.selectbox(("عملة" if use_ar else "Currency"),
                                   list(_FX_PP.keys()), key="pp_curr")
    with pp2:
        st.markdown(f"<div class='section-label'>{'العمالة والتكاليف' if use_ar else 'Labour & Costs'}</div>",
                    unsafe_allow_html=True)
        pp_hrs      = st.number_input(L["piece_labour_hrs"], min_value=0.0,
                                      value=3.0, step=0.5, key="pp_hrs")
        pp_hourly   = st.number_input(L["piece_hourly"],
                                      min_value=0.0, value=10.0, step=1.0, key="pp_hourly",
                                      help=("USD" if True else ""))
        pp_overhead = st.number_input(L["piece_overhead"],
                                      min_value=0.0, value=15.0, step=5.0, key="pp_oh")
        pp_margin   = st.number_input(L["piece_margin"],
                                      min_value=0.0, max_value=300.0,
                                      value=40.0, step=5.0, key="pp_marg")

    # ── Gemstones ─────────────────────────────────────────────────────────────
    st.markdown(f"<div class='section-label'>{L['piece_gems']}</div>",
                unsafe_allow_html=True)
    gem_cols = st.columns([3,2,2,1])
    with gem_cols[0]:
        new_gem_type = st.selectbox("", _GEM_NAMES, key="pp_gem_type",
                                    label_visibility="collapsed")
    with gem_cols[1]:
        new_gem_cts  = st.number_input(("قيراط" if use_ar else "Carats"),
                                       min_value=0.01, value=0.5, step=0.1,
                                       key="pp_gem_cts", format="%.2f",
                                       label_visibility="collapsed")
    with gem_cols[2]:
        gem_idx      = _GEM_NAMES.index(new_gem_type)
        gem_px_usd   = st.number_input(("$/قيراط" if use_ar else "$/carat"),
                                       value=float(_GEM_PX_LIST[gem_idx]),
                                       min_value=0.0, step=10.0,
                                       key="pp_gem_px", format="%.0f",
                                       label_visibility="collapsed")
    with gem_cols[3]:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("➕", key="pp_add_gem", use_container_width=True):
            st.session_state["piece_gems"].append({
                "name": new_gem_type, "cts": new_gem_cts, "px_usd": gem_px_usd
            })
            st.rerun()

    gems = st.session_state.get("piece_gems", [])
    gem_total_usd = 0.0
    if gems:
        for gi, gem in enumerate(gems):
            gc1, gc2, gc3, gc4 = st.columns([3,2,2,1])
            gem_val = gem["cts"] * gem["px_usd"]
            gem_total_usd += gem_val
            with gc1: st.markdown(f"<span style='font-size:12px;color:{C['text']};'>{gem['name']}</span>",
                                   unsafe_allow_html=True)
            with gc2: st.markdown(f"<span style='font-size:12px;color:{C['muted']};'>{gem['cts']:.2f}ct</span>",
                                   unsafe_allow_html=True)
            with gc3: st.markdown(f"<span style='font-size:12px;color:{C['gold']};font-family:monospace;'>${gem_val:,.2f}</span>",
                                   unsafe_allow_html=True)
            with gc4:
                if st.button("✕", key=f"pp_rm_{gi}", use_container_width=True):
                    st.session_state["piece_gems"].pop(gi)
                    st.rerun()

    # ── Compute ───────────────────────────────────────────────────────────────
    fx          = _FX_PP.get(pp_curr, 1.0)
    gold_ratio  = _PURITY_PP.get(pp_purity, 0.875)
    gram_spot   = g_ref / 31.1035
    raw_gold_g  = pp_gold_g * (1 + pp_wastage/100)
    gold_cost   = raw_gold_g * gold_ratio * gram_spot * fx
    gem_cost    = gem_total_usd * fx
    labour_cost = pp_hrs * pp_hourly * fx
    total_cost  = gold_cost + gem_cost + labour_cost + pp_overhead * fx
    suggest_px  = total_cost / (1 - pp_margin/100) if pp_margin < 100 else total_cost * 2
    profit      = suggest_px - total_cost

    # ── Results ───────────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    pr1, pr2, pr3, pr4 = st.columns(4)
    for col, lbl, val, vc in [
        (pr1, ("ذهب" if use_ar else "Gold"),      f"{gold_cost:,.2f} {pp_curr}",   C["gold"]),
        (pr2, ("أحجار" if use_ar else "Gems"),    f"{gem_cost:,.2f} {pp_curr}",    C["muted"]),
        (pr3, L["piece_cost"],                     f"{total_cost:,.2f} {pp_curr}", C["gold_pale"]),
        (pr4, L["piece_price"],                    f"{suggest_px:,.2f} {pp_curr}", C["green"]),
    ]:
        with col:
            st.markdown(f"""
            <div class='stat-card'>
              <div class='stat-label'>{lbl}</div>
              <div class='stat-value' style='color:{vc};font-size:15px;'>{val}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div style='text-align:center;font-size:11px;color:{C["green"]};
                margin-top:8px;font-family:monospace;'>
      {"صافي الربح:" if use_ar else "Net profit:"} {profit:,.2f} {pp_curr}
      ({pp_margin:.0f}%)
    </div>""", unsafe_allow_html=True)

    # ── Save design ───────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    des1, des2 = st.columns([3,1])
    with des1:
        design_name = st.text_input(("اسم التصميم" if use_ar else "Design Name"),
                                    placeholder=("خاتم سوليتير" if use_ar else "Solitaire ring"),
                                    key="pp_design_name")
    with des2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("💾 " + L["piece_save"], use_container_width=True, key="pp_save_btn",
                     type="primary"):
            st.session_state["saved_designs"].insert(0, {
                "name":    design_name or ("تصميم " + str(len(st.session_state["saved_designs"])+1)),
                "date":    datetime.datetime.now().strftime("%Y-%m-%d"),
                "gold_g":  pp_gold_g, "purity": pp_purity,
                "cost":    round(total_cost, 2),
                "price":   round(suggest_px, 2),
                "currency":pp_curr, "spot": g_ref,
                "gems":    list(gems),
            })
            st.session_state["piece_gems"] = []
            _sb_save()
            st.success("✅ " + ("تم حفظ التصميم" if use_ar else "Design saved"))
            st.rerun()

    designs = st.session_state.get("saved_designs", [])
    if designs:
        st.markdown(f"<div class='section-label'>{L['piece_designs']} ({len(designs)})</div>",
                    unsafe_allow_html=True)
        for des in designs[:10]:
            # Recalculate at current spot price for live P&L
            price_ratio = g_ref / des["spot"] if des["spot"] > 0 else 1.0
            current_price = des["price"] * price_ratio
            st.markdown(f"""
            <div style='background:{C["card"]};border:1px solid {C["border"]};
                        border-radius:5px;padding:10px 14px;margin-bottom:6px;
                        display:flex;justify-content:space-between;align-items:center;
                        {"direction:rtl;" if use_ar else ""}'>
              <div>
                <div style='font-size:13px;font-weight:700;color:{C["text"]};'>
                  💍 {des["name"]}</div>
                <div style='font-size:10px;color:{C["dim"]};margin-top:2px;'>
                  {des["date"]} · {des["gold_g"]}g {des["purity"]}</div>
              </div>
              <div style='text-align:right;'>
                <div style='font-family:monospace;font-size:14px;color:{C["gold_hi"]};'>
                  {des["price"]:,.2f} {des["currency"]}</div>
                <div style='font-size:10px;color:{C["green"]};'>
                  {"اليوم: " if use_ar else "Today: "}{current_price:,.2f}</div>
              </div>
            </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("<div class='mansa-divider'></div>",unsafe_allow_html=True)
_foot_logo = get_logo_svg(st.session_state["theme"], width=80)
# Note: footer uses components for the logo part, markdown for the text
st.markdown(f"""
<div style='display:flex;justify-content:space-between;align-items:center;padding:.5rem 0;flex-wrap:wrap;gap:8px;'>
  <div style='display:flex;align-items:center;gap:12px;'>
        <div style='font-size:28px;'>{'☽' if 'الحضارة' in st.session_state['theme'] else ('⬡' if 'العملة' in st.session_state['theme'] else '◈')}</div>
        <div>
          <div style='font-family:{C['font_h']},serif;font-size:9px;letter-spacing:.2em;color:{C['dim']};'>
        {L['app_name']} · {L['gold_intelligence']}
          </div>
          <div style='font-family:{C['font_b']},serif;font-size:9px;color:{C['dim']};font-style:italic;margin-top:2px;'>
        Inspired by Mansa Musa · The Golden King of Mali · 1312 CE
          </div>
        </div>
  </div>
  <span style='font-family:{C['font_b']},serif;font-size:11px;font-style:italic;color:{C['dim']};'>
        Yahoo Finance · ~15 min · {L['not_financial']} · v{__version__} · {datetime.datetime.now().year}
  </span>
</div>""",unsafe_allow_html=True)