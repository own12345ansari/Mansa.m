"""
mansa_dashboard.py  —  MANSA · Gold Intelligence Platform  v3
==============================================================
Fixes in this version:
  · Default theme is Islamic & Arab Civilization
  · Theme switching is fully reliable (CSS always matches active theme)
  · Multi-language: Arabic (primary), English, French, Turkish, Urdu
  · Language selector in sidebar; all UI labels switch
  · AI Predictions fully rebuilt:
      - Uses last real row from training data as feature input
      - Prophet predicts next trading day properly
      - Prediction date shown (tomorrow)
      - Best model highlighted (highest R²)
      - Selector to switch between models
      - Output in multiple units (oz, gram, kg, tola) × multiple currencies
  · Dashboard direct values: every live number shown explicitly
  · Navigation: sidebar radio always drives page, no stale state

Run:
    streamlit run mansa_dashboard.py
"""

import os, datetime, time, warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import streamlit as st
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="مانسا · ذكاء الذهب",
    page_icon="☽",
    layout="wide",
    initial_sidebar_state="expanded",
)

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
        "nav_advisor":   "💡  مستشار التداول",
        "nav_settings":  "⚙️  الإعدادات",
        "live": "مباشر",
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
        "nav_advisor":   "💡  Trading Advisor",
        "nav_settings":  "⚙️  Settings",
        "live": "Live",
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
        "nav_advisor":   "💡  Conseiller trading",
        "nav_settings":  "⚙️  Paramètres",
        "live": "En direct",
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
        "nav_advisor":   "💡  İşlem Danışmanı",
        "nav_settings":  "⚙️  Ayarlar",
        "live": "Canlı",
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
        "nav_advisor":   "💡  تجارتی مشاورت",
        "nav_settings":  "⚙️  ترتیبات",
        "live": "لائیو",
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
        "gold_glow":"#D4A01744","text":"#F0E8D5","muted":"#806050","dim":"#4A3828",
        "green":"#5CC86A","red":"#D9534F","blue":"#7DAACC","accent":"#B8860B",
        "brand":"☽","font_h":"Amiri","font_b":"Amiri","font_m":"JetBrains Mono",
        "desc":"الحضارة الإسلامية والعربية",
        "arabesque":"rgba(212,160,23,0.06)",
    },
    "العملة الذهبية القديمة ⬡": {
        "bg":"#080807","bg2":"#0D0C0A","card":"#111109","card2":"#161410",
        "border":"#2C2510","border2":"#3A3018",
        "gold":"#C9960C","gold_hi":"#F2C94C","gold_pale":"#F5DFA0","gold_dark":"#7A5A04",
        "gold_glow":"#C9960C44","text":"#EDE8D8","muted":"#7A7060","dim":"#4A4535",
        "green":"#52B788","red":"#E05C5C","blue":"#6EA8C8","accent":"#C9960C",
        "brand":"⬡","font_h":"Cinzel","font_b":"Cormorant Garamond","font_m":"JetBrains Mono",
        "desc":"العملات الذهبية القديمة",
        "arabesque":"rgba(201,150,12,0.04)",
    },
    "قاعة التداول ◈": {
        "bg":"#050A0F","bg2":"#080F16","card":"#0C1520","card2":"#101C28",
        "border":"#162030","border2":"#1E2E40",
        "gold":"#00C9A7","gold_hi":"#00FFD4","gold_pale":"#A0F0E0","gold_dark":"#007A66",
        "gold_glow":"#00C9A744","text":"#D0ECE8","muted":"#507060","dim":"#304540",
        "green":"#39D98A","red":"#FF4D6D","blue":"#4DA6FF","accent":"#F2C94C",
        "brand":"◈","font_h":"Share Tech Mono","font_b":"IBM Plex Sans","font_m":"Share Tech Mono",
        "desc":"قاعة التداول / بلومبرغ",
        "arabesque":"rgba(0,201,167,0.04)",
    },
}

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
    "Jordan (JOD)":      {"flag":"🇯🇴","currency":"JOD","fx_ticker":"USDJOD=X","fx_approx":0.709,  "unit_label":"gram",   "unit_factor_from_oz":1/31.1035,"note":"سوق عمّان للذهب",        "fx_inverse":False},
    "Saudi Arabia (SAR)":{"flag":"🇸🇦","currency":"SAR","fx_ticker":"USDSAR=X","fx_approx":3.75,   "unit_label":"gram",   "unit_factor_from_oz":1/31.1035,"note":"سوق الذهب بالرياض",      "fx_inverse":False},
    "UAE (AED)":         {"flag":"🇦🇪","currency":"AED","fx_ticker":"USDAED=X","fx_approx":3.6725, "unit_label":"gram",   "unit_factor_from_oz":1/31.1035,"note":"سوق دبي للذهب",          "fx_inverse":False},
    "Egypt (EGP)":       {"flag":"🇪🇬","currency":"EGP","fx_ticker":"USDEGP=X","fx_approx":50.9,   "unit_label":"gram",   "unit_factor_from_oz":1/31.1035,"note":"سوق القاهرة",            "fx_inverse":False},
    "Kuwait (KWD)":      {"flag":"🇰🇼","currency":"KWD","fx_ticker":"USDKWD=X","fx_approx":0.307,  "unit_label":"gram",   "unit_factor_from_oz":1/31.1035,"note":"سوق الكويت للذهب",       "fx_inverse":False},
    "Qatar (QAR)":       {"flag":"🇶🇦","currency":"QAR","fx_ticker":"USDQAR=X","fx_approx":3.64,   "unit_label":"gram",   "unit_factor_from_oz":1/31.1035,"note":"سوق الدوحة",             "fx_inverse":False},
    "Bahrain (BHD)":     {"flag":"🇧🇭","currency":"BHD","fx_ticker":"USDBHD=X","fx_approx":0.376,  "unit_label":"gram",   "unit_factor_from_oz":1/31.1035,"note":"سوق المنامة",            "fx_inverse":False},
    "Oman (OMR)":        {"flag":"🇴🇲","currency":"OMR","fx_ticker":"USDOMR=X","fx_approx":0.385,  "unit_label":"gram",   "unit_factor_from_oz":1/31.1035,"note":"سوق مسقط",               "fx_inverse":False},
    "Iraq (IQD)":        {"flag":"🇮🇶","currency":"IQD","fx_ticker":"USDIQD=X","fx_approx":1310.0, "unit_label":"gram",   "unit_factor_from_oz":1/31.1035,"note":"سوق بغداد",              "fx_inverse":False},
    "Turkey (TRY)":      {"flag":"🇹🇷","currency":"TRY","fx_ticker":"USDTRY=X","fx_approx":38.0,   "unit_label":"gram",   "unit_factor_from_oz":1/31.1035,"note":"البازار الكبير - إسطنبول","fx_inverse":False},
    "USA (USD)":         {"flag":"🇺🇸","currency":"USD","fx_ticker":None,      "fx_approx":1.0,    "unit_label":"troy oz","unit_factor_from_oz":1.0,       "note":"COMEX Spot",             "fx_inverse":False},
    "UK (GBP)":          {"flag":"🇬🇧","currency":"GBP","fx_ticker":"GBPUSD=X","fx_approx":1.27,   "unit_label":"troy oz","unit_factor_from_oz":1.0,       "note":"London Bullion Market",  "fx_inverse":True},
    "EU (EUR)":          {"flag":"🇪🇺","currency":"EUR","fx_ticker":"EURUSD=X","fx_approx":1.08,   "unit_label":"troy oz","unit_factor_from_oz":1.0,       "note":"Frankfurt / Paris",      "fx_inverse":True},
    "India (INR)":       {"flag":"🇮🇳","currency":"INR","fx_ticker":"USDINR=X","fx_approx":84.5,   "unit_label":"10g",    "unit_factor_from_oz":10/31.1035,"note":"MCX Mumbai",             "fx_inverse":False},
    "China (CNY)":       {"flag":"🇨🇳","currency":"CNY","fx_ticker":"USDCNY=X","fx_approx":7.27,   "unit_label":"gram",   "unit_factor_from_oz":1/31.1035,"note":"SGE Shanghai",           "fx_inverse":False},
    "Lebanon (USD)":     {"flag":"🇱🇧","currency":"USD","fx_ticker":None,      "fx_approx":1.0,    "unit_label":"gram",   "unit_factor_from_oz":1/31.1035,"note":"بيروت (USD)",            "fx_inverse":False},
}

STOCK_OPTIONS = {
    "S&P 500":"^GSPC","NASDAQ":"^IXIC","Dow Jones":"^DJI",
    "Apple":"AAPL","Microsoft":"MSFT","Amazon":"AMZN","Google":"GOOGL",
    "Tesla":"TSLA","NVIDIA":"NVDA","Meta":"META",
    "Aramco":"2222.SR","Bitcoin":"BTC-USD","Ethereum":"ETH-USD",
}

MODELS_DIR  = "models"
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
TIME_STEP = 10

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
    "auto_refresh":      False,
    "show_purity_table": True,
    "active_stocks":     ["S&P 500","Bitcoin","NVIDIA","Aramco"],
    "nav":               None,   # will be set from L below
    "advisor_profile":   None,
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Language shortcut ─────────────────────────────────────────────────────────
L  = LANGS.get(st.session_state["lang"], LANGS["العربية 🇸🇦"])
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
    if C["font_h"] == "Amiri":
        fi = "@import url('https://fonts.googleapis.com/css2?family=Amiri:wght@400;700&family=JetBrains+Mono:wght@300;400;500&display=swap');"
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
    font-family:'{C['font_b']}',Georgia,serif;
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
    font-family:'{C['font_h']}',serif; font-size:9px; font-weight:700;
    letter-spacing:.3em; text-transform:uppercase; color:{C['gold']};
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
.hero-price {{ font-family:'{C['font_h']}',serif; font-size:52px; font-weight:900; color:{C['gold_pale']}; line-height:.95; }}
.hero-unit  {{ font-family:'{C['font_b']}',serif; font-size:18px; font-style:italic; color:{C['muted']}; margin-left:8px; }}
.hero-change {{ font-family:'{C['font_m']}',monospace; font-size:16px; font-weight:500; margin-top:12px; }}
.hero-meta   {{ font-family:'{C['font_h']}',serif; font-size:8px; letter-spacing:.3em; color:{C['dim']}; margin-top:10px; text-transform:uppercase; }}
.stat-card {{
    background:{C['card']}; border:1px solid {C['border']}; border-radius:4px;
    padding:14px 16px; position:relative; overflow:hidden; margin-bottom:6px;
}}
.stat-card::after {{
    content:""; position:absolute; bottom:0; left:0; right:0; height:1px;
    background:linear-gradient(90deg,transparent,{C['gold']}33,transparent);
}}
.stat-label {{ font-family:'{C['font_h']}',serif; font-size:8px; letter-spacing:.2em; color:{C['muted']}; text-transform:uppercase; margin-bottom:6px; }}
.stat-value {{ font-family:'{C['font_m']}',monospace; font-size:18px; color:{C['gold_pale']}; line-height:1.1; }}
.ticker-item {{ background:{C['card']}; border:1px solid {C['border']}; border-radius:4px; padding:10px 12px; text-align:center; }}
.ticker-name  {{ font-family:'{C['font_h']}',serif; font-size:8px; letter-spacing:.2em; color:{C['muted']}; text-transform:uppercase; }}
.ticker-price {{ font-family:'{C['font_m']}',monospace; font-size:13px; color:{C['gold_pale']}; margin:3px 0; }}
.ticker-chg   {{ font-family:'{C['font_m']}',monospace; font-size:11px; }}
@keyframes pulse {{ 0%,100%{{opacity:1}} 50%{{opacity:.5}} }}
.live-badge {{ display:inline-flex; align-items:center; gap:6px; background:{C['green']}18; border:1px solid {C['green']}44; border-radius:20px; padding:4px 12px; }}
.live-dot   {{ width:7px; height:7px; border-radius:50%; background:{C['green']}; animation:pulse 2s infinite; }}
.live-text  {{ font-family:'{C['font_h']}',serif; font-size:8px; letter-spacing:.2em; color:{C['green']}; text-transform:uppercase; }}
.mkt-card {{
    background:{C['card2']}; border:1px solid {C['border2']}; border-radius:5px;
    padding:16px 18px; margin-bottom:8px; position:relative; overflow:hidden;
}}
.mkt-card::after {{
    content:""; position:absolute; top:0; left:0; right:0; height:1px;
    background:linear-gradient(90deg,transparent,{C['gold']}55,transparent);
}}
.mkt-name  {{ font-family:'{C['font_h']}',serif; font-size:9px; letter-spacing:.2em; color:{C['gold']}; text-transform:uppercase; margin-bottom:6px; }}
.mkt-price {{ font-family:'{C['font_m']}',monospace; font-size:20px; color:{C['gold_pale']}; }}
.mkt-unit  {{ font-family:'{C['font_b']}',serif; font-size:11px; font-style:italic; color:{C['muted']}; margin-left:4px; }}
.mkt-chg   {{ font-family:'{C['font_m']}',monospace; font-size:11px; margin-top:4px; }}
.purity-badge {{
    display:inline-block; background:{C['gold_dark']}44; border:1px solid {C['gold']}66;
    border-radius:3px; padding:2px 8px; font-family:'{C['font_h']}',serif;
    font-size:8px; letter-spacing:.15em; color:{C['gold']}; margin-right:4px;
}}
.pred-card {{
    background:{C['card2']}; border:1px solid {C['border2']}; border-radius:6px;
    padding:16px 20px; margin-bottom:10px; position:relative; overflow:hidden;
}}
.pred-card::before {{
    content:""; position:absolute; left:0; top:0; bottom:0; width:3px;
    background:linear-gradient(180deg,{C['gold']},{C['gold_dark']});
}}
.pred-best::before {{ background:linear-gradient(180deg,{C['gold_hi']},{C['gold']}) !important; }}
.pred-algo  {{ font-family:'{C['font_h']}',serif; font-size:8px; letter-spacing:.25em; color:{C['muted']}; text-transform:uppercase; margin-bottom:4px; }}
.pred-price {{ font-family:'{C['font_m']}',monospace; font-size:26px; color:{C['gold_hi']}; font-weight:700; }}
.pred-diff  {{ font-family:'{C['font_m']}',monospace; font-size:12px; margin-top:4px; }}
.pred-r2    {{ font-family:'{C['font_h']}',serif; font-size:8px; letter-spacing:.2em; color:{C['dim']}; margin-top:6px; }}
.settings-card {{
    background:{C['card2']}; border:1px solid {C['border2']}; border-radius:5px;
    padding:20px 22px; margin-bottom:12px;
}}
.settings-title {{
    font-family:'{C['font_h']}',serif; font-size:10px; font-weight:700;
    letter-spacing:.25em; color:{C['gold']}; text-transform:uppercase;
    margin-bottom:14px; padding-bottom:8px; border-bottom:1px solid {C['border']};
}}
.stButton > button {{
    font-family:'{C['font_h']}',serif !important; font-size:10px !important;
    letter-spacing:.2em !important; text-transform:uppercase !important;
    background:linear-gradient(135deg,{C['gold_dark']},{C['gold']}) !important;
    color:#050400 !important; border:none !important; border-radius:3px !important;
    padding:10px 28px !important; transition:opacity .2s !important;
}}
.stButton > button:hover {{ opacity:.85 !important; }}
.stRadio > div {{ gap:6px; }}
</style>"""

st.markdown(build_css(C), unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# DATA HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=60)
def fetch_live():
    tickers = {"gold":"GC=F","silver":"SI=F","oil":"CL=F","spx":"^GSPC",
               "dxy":"DX-Y.NYB","vix":"^VIX","us10y":"^TNX","btc":"BTC-USD","plat":"PL=F"}
    out = {}
    for k, sym in tickers.items():
        try:
            fi = yf.Ticker(sym).fast_info
            p  = float(fi["last_price"])
            pv = float(fi.get("previous_close", p))
            out[k] = {"price":p,"prev":pv,"change":p-pv,"pct":(p-pv)/pv*100 if pv else 0}
        except:
            out[k] = {"price":0,"prev":0,"change":0,"pct":0}
    return out

@st.cache_data(ttl=60)
def fetch_stock(sym):
    try:
        fi = yf.Ticker(sym).fast_info
        p  = float(fi["last_price"])
        pv = float(fi.get("previous_close", p))
        return {"price":p,"prev":pv,"change":p-pv,"pct":(p-pv)/pv*100 if pv else 0}
    except:
        return {"price":0,"prev":0,"change":0,"pct":0}

@st.cache_data(ttl=120)
def fetch_fx(ticker, fx_inverse, fx_approx):
    if ticker is None: return fx_approx
    try:
        return float(yf.Ticker(ticker).fast_info["last_price"])
    except:
        return fx_approx

@st.cache_data(ttl=300)
def fetch_history(period="1y", ticker="GC=F"):
    try:
        df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
        df = df.reset_index()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ['_'.join(str(c) for c in col).strip('_') for col in df.columns]
        return df
    except:
        return pd.DataFrame()

def find_col(df, candidates):
    for c in candidates:
        if c in df.columns: return c
    return None

def mkt_price(spot_usd_oz, mkt_cfg, purity_key):
    pm   = PURITIES[purity_key]["mult"]
    uf   = mkt_cfg["unit_factor_from_oz"]
    rate = fetch_fx(mkt_cfg["fx_ticker"], mkt_cfg.get("fx_inverse",False), mkt_cfg["fx_approx"])
    fx   = 1.0/rate if mkt_cfg.get("fx_inverse",False) else rate
    return spot_usd_oz * pm * uf * fx, fx

@st.cache_data
def load_csv():
    for p in ["updated_financial_data.csv","merged_financial_data.csv",
              "/mnt/user-data/uploads/merged_financial_data.csv",
              "/mnt/user-data/uploads/1986_updated.csv"]:
        if os.path.exists(p):
            df = pd.read_csv(p, parse_dates=["Date"])
            return df.sort_values("Date").reset_index(drop=True)
    return pd.DataFrame()

@st.cache_resource
def load_models():
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
            except: pass
    r2p = os.path.join(MODELS_DIR,"r2_scores.csv")
    if os.path.exists(r2p):
        try:
            r2df = pd.read_csv(r2p, index_col=0, names=["Model","R2"], header=0)
            r2s  = r2df["R2"].to_dict()
        except: pass
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
    except Exception as e:
        return None

# ── Build feature row from last CSV row + live prices ─────────────────────────
def build_features(g_ref, live):
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

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    C2 = THEMES[st.session_state["theme"]]
    st.markdown(f"""
    <div style="text-align:center;padding:1rem 0 .5rem;">
      <div style="font-size:44px;filter:drop-shadow(0 0 14px {C2['gold']}99);">{C2['brand']}</div>
      <div style="font-family:'{C2['font_h']}',serif;font-size:22px;font-weight:900;
                  color:{C2['gold_pale']};letter-spacing:.25em;margin-top:4px;">{L['app_name']}</div>
      <div style="font-family:'{C2['font_b']}',serif;font-size:12px;font-style:italic;
                  color:{C2['muted']};">{L['tagline']}</div>
    </div>""", unsafe_allow_html=True)
    st.markdown("<div class='mansa-divider'></div>", unsafe_allow_html=True)

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

    # Navigation
    nav_opts = [L["nav_dashboard"],L["nav_markets"],L["nav_charts"],L["nav_simulator"],
                L["nav_predictions"],L["nav_data"],L["nav_advisor"],L["nav_settings"]]
    # Ensure stored nav is in current language's options
    if st.session_state["nav"] not in nav_opts:
        st.session_state["nav"] = nav_opts[0]

    nav = st.radio("nav", nav_opts,
                   index=nav_opts.index(st.session_state["nav"]),
                   label_visibility="collapsed", key="nav_radio")
    st.session_state["nav"] = nav

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

    st.session_state["auto_refresh"] = st.checkbox(
        L["auto_refresh"], st.session_state["auto_refresh"])

    st.markdown("<div class='mansa-divider'></div>", unsafe_allow_html=True)
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    st.markdown(f"""
    <div style='text-align:center;'>
      <span class='live-badge'><span class='live-dot'></span>
        <span class='live-text'>{L['live']} · {ts}</span></span>
    </div>
    <p style='text-align:center;font-family:{C['font_h']},serif;font-size:8px;
              color:{C['dim']};margin-top:.8rem;letter-spacing:.1em;'>
      Yahoo Finance · ~15 min<br>{L['not_financial']}
    </p>""", unsafe_allow_html=True)

if st.session_state["auto_refresh"]:
    time.sleep(1); st.rerun()

# ── Shared live values ────────────────────────────────────────────────────────
live      = fetch_live()
g_ref     = live["gold"]["price"] if live["gold"]["price"] > 0 else 3200.0
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

# ── Page header helper ────────────────────────────────────────────────────────
def ph(title, sub=""):
    rtl = "direction:rtl;text-align:right;" if L["dir"]=="rtl" else ""
    st.markdown(f"""
    <h1 style='font-family:{C['font_h']},serif;font-size:26px;font-weight:900;
               color:{C['gold_pale']};letter-spacing:.06em;margin-bottom:2px;{rtl}'>{title}</h1>
    <p style='font-family:{C['font_b']},serif;font-size:14px;font-style:italic;
              color:{C['muted']};{rtl}'>{sub}</p>""", unsafe_allow_html=True)
    st.markdown("<div class='mansa-divider'></div>", unsafe_allow_html=True)

def stat_card(label, value, sub="", col_override=None):
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

    # ── Hero ──────────────────────────────────────────────────────────────────
    h1, h2 = st.columns([3,2], gap="medium")
    with h1:
        gold_chg_col = C["green"] if live["gold"]["change"]>=0 else C["red"]
        gold_arr     = "▲" if live["gold"]["change"]>=0 else "▼"
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

    # ── Live direct values strip (commodities) ────────────────────────────────
    st.markdown(f"<div class='section-label'>{L['market_overview']}</div>", unsafe_allow_html=True)
    strip_data = [
        ("Silver",     live["silver"], "$", "/oz"),
        ("Crude Oil",  live["oil"],    "$", "/bbl"),
        ("DXY",        live["dxy"],    "",  ""),
        ("VIX",        live["vix"],    "",  ""),
        ("US 10Y",     live["us10y"],  "",  "%"),
        ("Platinum",   live["plat"],   "$", "/oz"),
    ]
    scols = st.columns(len(strip_data))
    for col, (name, d, pfx, sfx) in zip(scols, strip_data):
        p   = d["price"]
        pct = d["pct"]
        ch  = d["change"]
        c2  = C["green"] if ch>=0 else C["red"]
        arr = "▲" if ch>=0 else "▼"
        with col:
            st.markdown(f"""
            <div class='ticker-item'>
              <div class='ticker-name'>{name}</div>
              <div class='ticker-price'>{pfx}{p:,.2f}{sfx}</div>
              <div class='ticker-chg' style='color:{c2};'>{arr} {pct:+.2f}%</div>
            </div>""", unsafe_allow_html=True)

    # ── Stocks strip ──────────────────────────────────────────────────────────
    act_stks = st.session_state.get("active_stocks",[])
    if act_stks:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"<div class='section-label'>{L['stocks_indices']}</div>", unsafe_allow_html=True)
        stk_cols = st.columns(min(len(act_stks),7))
        for i, sn in enumerate(act_stks[:7]):
            sym = STOCK_OPTIONS.get(sn, sn)
            sd  = fetch_stock(sym)
            c2  = C["green"] if sd["change"]>=0 else C["red"]
            arr = "▲" if sd["change"]>=0 else "▼"
            with stk_cols[i]:
                st.markdown(f"""
                <div class='ticker-item'>
                  <div class='ticker-name'>{sn}</div>
                  <div class='ticker-price'>${sd['price']:,.2f}</div>
                  <div class='ticker-chg' style='color:{c2};'>{arr} {sd['pct']:+.2f}%</div>
                </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── All purities table ────────────────────────────────────────────────────
    if st.session_state.get("show_purity_table", True):
        st.markdown(f"<div class='section-label'>{L['all_purities']} · {st.session_state['primary_mkt']} · {PM['unit_label']}</div>", unsafe_allow_html=True)
        pcols = st.columns(4)
        for idx,(pur_name, pc) in enumerate(PURITIES.items()):
            pp,_ = mkt_price(g_ref, PM, pur_name)
            is_sel = pur_name == st.session_state["purity"]
            bc = C["gold"] if is_sel else C["border"]
            badge = "<span class='purity-badge'>✓</span>" if is_sel else ""
            with pcols[idx%4]:
                st.markdown(f"""
                <div class='stat-card' style='border-color:{bc};'>
                  <div class='stat-label'>{pc['label']} · {pc['fine']}‰</div>
                  <div class='stat-value' style='font-size:16px;'>{pp:,.3f}
                    <span style='font-size:11px;color:{C['muted']};'>{PM['currency']}</span>
                  </div>
                  <div style='font-size:11px;color:{C['dim']};'>/ {PM['unit_label']} {badge}</div>
                </div>""", unsafe_allow_html=True)

    # ── 52-week stats ─────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"<div class='section-label'>{L['key_stats']} (24K · USD)</div>", unsafe_allow_html=True)
    df1y = fetch_history("1y","GC=F")
    kc = st.columns(6)
    if not df1y.empty:
        cc = find_col(df1y,["Close","Close_GC=F"])
        if cc:
            hi52  = df1y[cc].max()
            lo52  = df1y[cc].min()
            av52  = df1y[cc].mean()
            ytd   = (df1y[cc].iloc[-1]/df1y[cc].iloc[0]-1)*100
            gsr   = g_ref/live["silver"]["price"] if live["silver"]["price"] else 0
            gor   = g_ref/live["oil"]["price"]    if live["oil"]["price"]    else 0
            for col,lbl,val in [
                (kc[0],L["wk52_high"],  f"${hi52:,.0f}"),
                (kc[1],L["wk52_low"],   f"${lo52:,.0f}"),
                (kc[2],L["wk52_avg"],   f"${av52:,.0f}"),
                (kc[3],L["ytd_return"], f"{ytd:+.1f}%"),
                (kc[4],L["gold_silver"],f"{gsr:.1f}×"),
                (kc[5],L["gold_oil"],   f"{gor:.1f}×"),
            ]:
                with col:
                    st.markdown(f"""
                    <div class='stat-card' style='text-align:center;'>
                      <div class='stat-label'>{lbl}</div>
                      <div class='stat-value' style='font-size:15px;'>{val}</div>
                    </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: MARKETS
# ═══════════════════════════════════════════════════════════════════════════════
elif nav == L["nav_markets"]:
    ph(L["nav_markets"], f"{P_LABEL} · {PM['unit_label']}")

    arab_flags  = {"🇯🇴","🇸🇦","🇦🇪","🇪🇬","🇰🇼","🇶🇦","🇧🇭","🇴🇲","🇱🇧","🇮🇶","🇹🇷"}
    arab_mkts   = {k:v for k,v in MARKETS.items() if v["flag"] in arab_flags}
    global_mkts = {k:v for k,v in MARKETS.items() if k not in arab_mkts}

    def render_grid(mkt_dict, ncols=3):
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
                pb    = f"<span class='purity-badge'>✦</span>" if is_pm else ""
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
                      <div style='font-size:11px;font-style:italic;color:{C['dim']};margin-top:3px;'>{m['note']}</div>
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
    ph(L["nav_charts"])
    ca1,ca2,ca3 = st.columns(3)
    with ca1: chart_asset  = st.selectbox("Asset", ["Gold","Silver","Crude Oil","S&P 500","Platinum","Bitcoin"])
    with ca2: chart_period = st.selectbox("Period", ["5d","1mo","3mo","6mo","1y","2y","5y"], index=4)
    with ca3: chart_type   = st.selectbox("Chart Type", ["Candlestick","Line","Area"])

    tm = {"Gold":"GC=F","Silver":"SI=F","Crude Oil":"CL=F","S&P 500":"^GSPC","Platinum":"PL=F","Bitcoin":"BTC-USD"}
    sym   = tm[chart_asset]
    df_ch = fetch_history(chart_period, sym)
    fch   = U_FACTOR*P_MULT if chart_asset=="Gold" else 1.0

    if not df_ch.empty:
        cl=find_col(df_ch,["Close",f"Close_{sym}"])
        op=find_col(df_ch,["Open",f"Open_{sym}"])
        hi=find_col(df_ch,["High",f"High_{sym}"])
        lo=find_col(df_ch,["Low",f"Low_{sym}"])
        vo=find_col(df_ch,["Volume",f"Volume_{sym}"])
        dt=find_col(df_ch,["Date","Datetime"])
        fig=make_subplots(rows=2,cols=1,shared_xaxes=True,vertical_spacing=.04,row_heights=[.78,.22])
        if chart_type=="Candlestick" and all([op,hi,lo,cl,dt]):
            fig.add_trace(go.Candlestick(x=df_ch[dt],open=df_ch[op]*fch,high=df_ch[hi]*fch,
                low=df_ch[lo]*fch,close=df_ch[cl]*fch,
                increasing_line_color=C["gold"],decreasing_line_color="#884422",
                increasing_fillcolor=C["gold_dark"],decreasing_fillcolor="#331100",name=chart_asset),row=1,col=1)
        elif cl and dt:
            fig.add_trace(go.Scatter(x=df_ch[dt],y=df_ch[cl]*fch,name=chart_asset,
                line=dict(color=C["gold"],width=2),
                fill="tozeroy" if chart_type=="Area" else None,
                fillcolor=C["gold_dark"]+"33"),row=1,col=1)
        if cl and dt:
            for maw,mac in [(20,"#7B9FD4"),(50,C["gold_hi"]),(200,C["red"])]:
                if len(df_ch)>=maw:
                    df_ch[f"MA{maw}"]=df_ch[cl].rolling(maw).mean()*fch
                    fig.add_trace(go.Scatter(x=df_ch[dt],y=df_ch[f"MA{maw}"],name=f"MA{maw}",
                        line=dict(color=mac,width=1.2,dash="dot"),opacity=.75),row=1,col=1)
        if vo and dt:
            vc2=[C["gold"] if (cl and op and df_ch[cl].iloc[i]>=df_ch[op].iloc[i]) else "#884422" for i in range(len(df_ch))]
            fig.add_trace(go.Bar(x=df_ch[dt],y=df_ch[vo],name="Volume",marker_color=vc2,opacity=.5),row=2,col=1)
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor=C["card"],
            font=dict(color=C["text"],family=C["font_m"],size=10),
            legend=dict(bgcolor=C["card2"],bordercolor=C["border2"],font=dict(size=9)),
            margin=dict(l=0,r=0,t=0,b=0),height=520,xaxis_rangeslider_visible=False)
        for ax in ["xaxis","xaxis2","yaxis","yaxis2"]:
            fig.update_layout(**{ax:dict(gridcolor=C["border2"],color=C["muted"])})
        st.plotly_chart(fig, use_container_width=True)
        if cl and dt:
            v=df_ch[cl]*fch
            s1,s2,s3,s4,s5=st.columns(5)
            for sc,lbl,val in [(s1,"High",f"${v.max():,.2f}"),(s2,"Low",f"${v.min():,.2f}"),
                (s3,"Avg",f"${v.mean():,.2f}"),(s4,"Std",f"${v.std():,.2f}"),
                (s5,"Return",f"{((v.iloc[-1]/v.iloc[0])-1)*100:+.2f}%")]:
                with sc:
                    st.markdown(f"""<div class='stat-card' style='text-align:center;'>
                      <div class='stat-label'>{lbl}</div>
                      <div class='stat-value' style='font-size:13px;'>{val}</div></div>""",
                      unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: SIMULATOR
# ═══════════════════════════════════════════════════════════════════════════════
elif nav == L["nav_simulator"]:
    ph(L["nav_simulator"])
    def safe(d): return round(d["price"],2) if d["price"]>0 else 0.0
    sc1,sc2 = st.columns(2,gap="large")
    with sc1:
        sim_spx=st.number_input("S&P 500",value=safe(live["spx"]),step=10.0,format="%.2f")
        sim_dxy=st.number_input("DXY",value=safe(live["dxy"]),step=0.1,format="%.2f")
        sim_oil=st.number_input("Oil (USD/bbl)",value=safe(live["oil"]),step=1.0,format="%.2f")
        sim_silver=st.number_input("Silver (USD/oz)",value=safe(live["silver"]),step=0.1,format="%.2f")
        sim_vix=st.number_input("VIX",value=safe(live["vix"]),step=0.5,format="%.2f")
    with sc2:
        sim_cpi=st.number_input("CPI",value=314.0,step=1.0,format="%.1f")
        sim_effr=st.number_input("EFFR (%)",value=4.33,step=0.25,format="%.2f")
        sim_real=st.number_input("Real Rate (%)",value=2.0,step=0.1,format="%.2f")
        sim_us10y=st.number_input("US 10Y (%)",value=safe(live["us10y"]),step=0.05,format="%.2f")
        sim_btc=st.number_input("Bitcoin",value=safe(live["btc"]),step=100.0,format="%.0f")
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

    if not models_dict:
        st.warning(L["no_models"])
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
        use_ar = st.session_state["lang"].startswith("ال") or st.session_state["lang"].startswith("اردو")

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
                        fillcolor=C["blue"]+"33"),
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
    ph(L["data_explorer"], L["training_data"])
    tdf = load_csv()
    if tdf.empty:
        st.warning(L["not_found_data"])
    else:
        s1,s2,s3,s4 = st.columns(4)
        for col,lbl,val in [(s1,L["total_rows"],f"{len(tdf):,}"),
            (s2,L["features"],str(len(tdf.columns))),
            (s3,L["from_date"],tdf["Date"].min().strftime("%d %b %Y") if "Date" in tdf else "—"),
            (s4,L["to_date"],  tdf["Date"].max().strftime("%d %b %Y") if "Date" in tdf else "—")]:
            with col:
                st.markdown(f"""<div class='stat-card' style='text-align:center;'>
                  <div class='stat-label'>{lbl}</div>
                  <div class='stat-value'>{val}</div></div>""",unsafe_allow_html=True)
        st.markdown("<br>",unsafe_allow_html=True)
        dc1,dc2,dc3 = st.columns([2,2,1])
        with dc1:
            yr = st.slider(L["filter_year"],int(tdf["Date"].dt.year.min()),
                int(tdf["Date"].dt.year.max()),(2010,int(tdf["Date"].dt.year.max()))) if "Date" in tdf else (1990,2025)
        with dc2:
            cs = st.multiselect(L["columns_display"],list(tdf.columns),
                default=["Date","Gold_Price","SPX_Close","CPI","EFFR","USD_Index","Oil_Price","Silver_Price","VIX"])
        with dc3:
            nr = st.number_input(L["rows_show"],min_value=10,max_value=500,value=50,step=10)
        filt = tdf.copy()
        if "Date" in tdf:
            filt = filt[(filt["Date"].dt.year>=yr[0])&(filt["Date"].dt.year<=yr[1])]
        if cs:
            filt = filt[[c for c in cs if c in filt.columns]]
        st.markdown(f"<div class='section-label'>{len(filt):,} rows</div>",unsafe_allow_html=True)
        st.dataframe(filt.tail(nr),use_container_width=True,hide_index=True)
        if "Date" in tdf and "Gold_Price" in tdf:
            st.markdown(f"<div class='section-label'>{L['gold_history']}</div>",unsafe_allow_html=True)
            fig_h=go.Figure(go.Scatter(x=tdf["Date"],y=tdf["Gold_Price"],name="Gold",
                line=dict(color=C["gold"],width=1.5),fill="tozeroy",fillcolor=C["gold_dark"]+"22"))
            fig_h.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor=C["card"],
                font=dict(color=C["text"],family=C["font_m"],size=10),
                yaxis_title="USD/oz",height=340,margin=dict(l=0,r=0,t=0,b=0),hovermode="x unified",
                xaxis=dict(gridcolor=C["border2"],color=C["muted"]),
                yaxis=dict(gridcolor=C["border2"],color=C["muted"]))
            st.plotly_chart(fig_h,use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: TRADING ADVISOR
# ═══════════════════════════════════════════════════════════════════════════════
elif nav == L["nav_advisor"]:
    ph(L["trading_advisor"])

    @st.cache_data(ttl=300)
    def get_ta():
        df = fetch_history("1y","GC=F")
        if df.empty: return pd.DataFrame()
        cl=find_col(df,["Close","Close_GC=F"]); dt=find_col(df,["Date","Datetime"])
        if not cl or not dt: return pd.DataFrame()
        out=df[[dt,cl]].copy(); out.columns=["Date","Close"]
        out=out.dropna().sort_values("Date").reset_index(drop=True)
        out["MA20"]=out["Close"].rolling(20).mean()
        out["MA50"]=out["Close"].rolling(50).mean()
        out["MA200"]=out["Close"].rolling(200).mean()
        out["EMA12"]=out["Close"].ewm(span=12,adjust=False).mean()
        out["EMA26"]=out["Close"].ewm(span=26,adjust=False).mean()
        out["MACD"]=out["EMA12"]-out["EMA26"]
        out["Signal"]=out["MACD"].ewm(span=9,adjust=False).mean()
        out["MACD_hist"]=out["MACD"]-out["Signal"]
        d=out["Close"].diff(); g=d.clip(lower=0).rolling(14).mean(); lo=(-d.clip(upper=0)).rolling(14).mean()
        out["RSI"]=100-(100/(1+g/lo.replace(0,np.nan)))
        out["BB_mid"]=out["Close"].rolling(20).mean(); out["BB_std"]=out["Close"].rolling(20).std()
        out["BB_upper"]=out["BB_mid"]+2*out["BB_std"]; out["BB_lower"]=out["BB_mid"]-2*out["BB_std"]
        out["BB_pct"]=(out["Close"]-out["BB_lower"])/(out["BB_upper"]-out["BB_lower"]).replace(0,np.nan)
        lo14=out["Close"].rolling(14).min(); hi14=out["Close"].rolling(14).max()
        out["Stoch_K"]=100*(out["Close"]-lo14)/(hi14-lo14).replace(0,np.nan)
        out["Stoch_D"]=out["Stoch_K"].rolling(3).mean()
        out["ATR"]=out["Close"].diff().abs().rolling(14).mean()
        out["ROC"]=out["Close"].pct_change(10)*100
        return out

    ta  = get_ta()
    use_ar = st.session_state["lang"].startswith("ال") or st.session_state["lang"].startswith("اردو")

    def compute_signals(ta, g_ref):
        if ta.empty or len(ta)<10: return {}, {}
        last=ta.iloc[-1]; scores=[]
        sigs={}
        def add(name,act,reason,col):
            sigs[name]=(act,reason,col)
            scores.append(1 if "BUY" in act else (-1 if "SELL" in act else 0))
        if pd.notna(last["MA50"]):
            if last["Close"]>last["MA50"]: add("MA50","BUY",f"Price ${last['Close']:,.0f} > MA50 ${last['MA50']:,.0f}",C["green"])
            else: add("MA50","SELL",f"Price ${last['Close']:,.0f} < MA50 ${last['MA50']:,.0f}",C["red"])
        if pd.notna(last["RSI"]):
            r=last["RSI"]
            if r<30: add("RSI(14)","BUY",f"RSI={r:.1f} — Oversold",C["green"])
            elif r>70: add("RSI(14)","SELL",f"RSI={r:.1f} — Overbought",C["red"])
            else: add("RSI(14)","HOLD",f"RSI={r:.1f} — Neutral",C["muted"])
        if pd.notna(last["MACD"]) and pd.notna(last["Signal"]):
            if last["MACD"]>last["Signal"]: add("MACD","BUY",f"MACD {last['MACD']:+.2f} > Signal",C["green"])
            else: add("MACD","SELL",f"MACD {last['MACD']:+.2f} < Signal",C["red"])
        if pd.notna(last["BB_pct"]):
            bp=last["BB_pct"]
            if bp<0.1: add("Bollinger","BUY",f"Near lower band ({bp:.0%})",C["green"])
            elif bp>0.9: add("Bollinger","SELL",f"Near upper band ({bp:.0%})",C["red"])
            else: add("Bollinger","HOLD",f"Mid-band ({bp:.0%})",C["muted"])
        if pd.notna(last["MA200"]):
            if last["MA50"]>last["MA200"]: add("MA50/200","BUY",f"MA50 > MA200 · Bullish",C["green"])
            else: add("MA50/200","SELL",f"MA50 < MA200 · Bearish",C["red"])
        vix_p=live["vix"]["price"]
        if vix_p>25: add("VIX","BUY",f"VIX={vix_p:.1f} High fear → safe haven demand",C["green"])
        elif vix_p<14: add("VIX","HOLD",f"VIX={vix_p:.1f} Low volatility",C["muted"])
        else: add("VIX","HOLD",f"VIX={vix_p:.1f} Moderate",C["muted"])
        dxy_c=live["dxy"]["pct"]
        if dxy_c<-0.5: add("DXY","BUY",f"DXY {dxy_c:+.2f}% → Weak USD bullish for gold",C["green"])
        elif dxy_c>0.5: add("DXY","SELL",f"DXY {dxy_c:+.2f}% → Strong USD headwind",C["red"])
        else: add("DXY","HOLD",f"DXY {dxy_c:+.2f}%",C["muted"])
        roc=last["ROC"] if pd.notna(last["ROC"]) else 0
        if roc>3: add("Momentum","BUY",f"10d ROC +{roc:.1f}%",C["green"])
        elif roc<-3: add("Momentum","SELL",f"10d ROC {roc:.1f}%",C["red"])
        else: add("Momentum","HOLD",f"10d ROC {roc:.1f}%",C["muted"])
        bull=scores.count(1); bear=scores.count(-1); total=len(scores) or 1
        bp=bull/total*100
        if bp>=65: ov=("STRONG BUY" if use_ar else "STRONG BUY",C["green"],"▲▲")
        elif bp>=50: ov=("BUY",C["green"],"▲")
        elif bear/total*100>=65: ov=("STRONG SELL",C["red"],"▼▼")
        elif bear/total*100>=50: ov=("SELL",C["red"],"▼")
        else: ov=("NEUTRAL",C["muted"],"◆")
        hi52=ta["Close"].tail(252).max(); lo52=ta["Close"].tail(252).min()
        atr=last["ATR"] if pd.notna(last["ATR"]) else g_ref*0.01
        return sigs, {"overall":ov,"bull":bull,"bear":bear,"neut":scores.count(0),"total":total,
                      "last":last,"hi52":hi52,"lo52":lo52,"atr":atr}

    signals, summary = compute_signals(ta, g_ref)

    # Profile selector
    p_col,_ = st.columns([2,3])
    with p_col:
        prof_opts = L["profile_opts"]
        if st.session_state["advisor_profile"] not in prof_opts:
            st.session_state["advisor_profile"] = prof_opts[0]
        profile = st.selectbox(L["profile"], prof_opts,
            index=prof_opts.index(st.session_state["advisor_profile"]))
        st.session_state["advisor_profile"] = profile

    st.markdown("<br>",unsafe_allow_html=True)

    # Overall signal banner
    if summary:
        ov_label,ov_col,ov_icon = summary["overall"]
        bull,bear,neut,total = summary["bull"],summary["bear"],summary["neut"],summary["total"]
        bw=int(bull/total*100); bew=int(bear/total*100); nw=100-bw-bew
        st.markdown(f"""
        <div class='hero-wrap' style='text-align:center;margin-bottom:1.5rem;'>
          <div class='stat-label' style='margin-bottom:8px;'>{L['overall_signal']} · {total} indicators</div>
          <div style='font-family:{C['font_h']},serif;font-size:44px;font-weight:900;color:{ov_col};'>
            {ov_icon} &nbsp; {ov_label}</div>
          <div style='font-family:{C['font_m']},monospace;font-size:12px;color:{C['muted']};margin-top:8px;'>
            ${g_ref:,.2f} · {L['bullish']}: {bull} · {L['bearish']}: {bear} · {L['neutral']}: {neut}
          </div>
          <div style='display:flex;height:7px;border-radius:4px;overflow:hidden;margin-top:12px;gap:1px;'>
            <div style='width:{bw}%;background:{C['green']};'></div>
            <div style='width:{nw}%;background:{C['muted']}44;'></div>
            <div style='width:{bew}%;background:{C['red']};'></div>
          </div>
        </div>""",unsafe_allow_html=True)

    # Signal cards
    st.markdown(f"<div class='section-label'>📡 {L['live_signals']}</div>",unsafe_allow_html=True)
    sig_items = list(signals.items())
    for rs in range(0,len(sig_items),3):
        row=sig_items[rs:rs+3]; cols3=st.columns(3)
        for col,(nm,(act,reason,color)) in zip(cols3,row):
            ic="▲" if "BUY" in act else ("▼" if "SELL" in act else "◆")
            bg=f"{C['green']}12" if "BUY" in act else (f"{C['red']}12" if "SELL" in act else C["card2"])
            with col:
                st.markdown(f"""
                <div style='background:{bg};border:1px solid {color}55;border-radius:4px;
                            padding:12px 14px;margin-bottom:8px;position:relative;'>
                  <div style='position:absolute;top:0;left:0;right:0;height:2px;background:{color};'></div>
                  <div class='stat-label'>{nm}</div>
                  <div style='font-family:{C['font_h']},serif;font-size:15px;font-weight:700;color:{color};'>
                    {ic} {act}</div>
                  <div style='font-family:{C['font_b']},serif;font-size:12px;color:{C['text']};
                              margin-top:4px;font-style:italic;'>{reason}</div>
                </div>""",unsafe_allow_html=True)

    # Price levels
    if summary and not ta.empty:
        last=summary["last"]; atr=summary["atr"]
        sup1=round(last["BB_lower"],2) if pd.notna(last["BB_lower"]) else round(g_ref-2*atr,2)
        sup2=round(last["MA200"],2)    if pd.notna(last["MA200"])    else round(g_ref-4*atr,2)
        res1=round(last["BB_upper"],2) if pd.notna(last["BB_upper"]) else round(g_ref+2*atr,2)
        res2=round(summary["hi52"],2)
        sl  =round(g_ref-1.5*atr,2); tp1=round(g_ref+1.5*atr,2); tp2=round(g_ref+3.0*atr,2)
        st.markdown("<br>",unsafe_allow_html=True)
        st.markdown(f"<div class='section-label'>🎯 {L['price_levels']}</div>",unsafe_allow_html=True)
        lc=st.columns(4)
        for col,lbl,val,col_c in [
            (lc[0],f"🟢 {L['support']} 1",f"${sup1:,.2f}","BB lower",),
            (lc[1],f"🟢 {L['support']} 2",f"${sup2:,.2f}","MA200",),
            (lc[2],f"🔴 {L['resistance']} 1",f"${res1:,.2f}","BB upper"),
            (lc[3],f"🔴 {L['resistance']} 2",f"${res2:,.2f}","52W high"),
        ]:
            vc=C["green"] if "🟢" in lbl else C["red"]
            with col:
                st.markdown(f"""<div class='stat-card' style='border-color:{vc}55;text-align:center;'>
                  <div class='stat-label'>{lbl}</div>
                  <div class='stat-value' style='color:{vc};'>{val}</div>
                  <div style='font-size:10px;color:{C['dim']};'>{col_c}</div></div>""",unsafe_allow_html=True)
        st.markdown("<br>",unsafe_allow_html=True)
        st.markdown(f"<div class='section-label'>📐 {L['trade_setup']} (ATR=${atr:,.2f})</div>",unsafe_allow_html=True)
        tc=st.columns(4)
        for col,lbl,val,vc,note in [
            (tc[0],L['entry'],   f"${g_ref:,.2f}", C["gold"],   "Current spot"),
            (tc[1],L['stop_loss'],f"${sl:,.2f}",   C["red"],    "1.5× ATR below"),
            (tc[2],f"{L['take_profit']} 1",f"${tp1:,.2f}",C["green"],"1.5× ATR above"),
            (tc[3],f"{L['take_profit']} 2",f"${tp2:,.2f}",C["green"],"3.0× ATR above"),
        ]:
            with col:
                st.markdown(f"""<div class='stat-card' style='border-color:{vc};text-align:center;'>
                  <div class='stat-label'>{lbl}</div>
                  <div class='stat-value' style='color:{vc};'>{val}</div>
                  <div style='font-size:10px;color:{C['dim']};'>{note}</div></div>""",unsafe_allow_html=True)

    # TA chart
    if not ta.empty:
        st.markdown("<br>",unsafe_allow_html=True)
        fig_ta=make_subplots(rows=3,cols=1,shared_xaxes=True,vertical_spacing=.04,
            row_heights=[.60,.20,.20])
        fig_ta.add_trace(go.Scatter(x=ta["Date"],y=ta["BB_upper"],name="BB+",
            line=dict(color=C["muted"],width=1,dash="dot"),showlegend=False),row=1,col=1)
        fig_ta.add_trace(go.Scatter(x=ta["Date"],y=ta["BB_lower"],name="BB-",
            fill="tonexty",fillcolor=C["gold_dark"]+"18",
            line=dict(color=C["muted"],width=1,dash="dot"),showlegend=False),row=1,col=1)
        fig_ta.add_trace(go.Scatter(x=ta["Date"],y=ta["Close"],name="Gold",
            line=dict(color=C["gold"],width=2)),row=1,col=1)
        for maw,mac,mn in [(20,"#7B9FD4","MA20"),(50,C["gold_hi"],"MA50"),(200,C["red"],"MA200")]:
            if f"MA{maw}" in ta.columns:
                fig_ta.add_trace(go.Scatter(x=ta["Date"],y=ta[f"MA{maw}"],name=mn,
                    line=dict(color=mac,width=1.2,dash="dot"),opacity=.8),row=1,col=1)
        fig_ta.add_trace(go.Scatter(x=ta["Date"],y=ta["RSI"],name="RSI",
            line=dict(color=C["blue"],width=1.5)),row=2,col=1)
        for yv,lc2 in [(70,C["red"]),(30,C["green"])]:
            fig_ta.add_hline(y=yv,line_dash="dash",line_color=lc2,opacity=0.5,row=2,col=1)
        mhc=[C["green"] if v>=0 else C["red"] for v in ta["MACD_hist"].fillna(0)]
        fig_ta.add_trace(go.Bar(x=ta["Date"],y=ta["MACD_hist"],name="Hist",
            marker_color=mhc,opacity=0.7),row=3,col=1)
        fig_ta.add_trace(go.Scatter(x=ta["Date"],y=ta["MACD"],name="MACD",
            line=dict(color=C["gold"],width=1.5)),row=3,col=1)
        fig_ta.add_trace(go.Scatter(x=ta["Date"],y=ta["Signal"],name="Signal",
            line=dict(color=C["red"],width=1.2,dash="dot")),row=3,col=1)
        fig_ta.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor=C["card"],
            font=dict(color=C["text"],family=C["font_m"],size=9),
            legend=dict(bgcolor=C["card2"],bordercolor=C["border2"],font=dict(size=9),
                orientation="h",y=1.04),
            height=600,margin=dict(l=0,r=0,t=30,b=0),xaxis_rangeslider_visible=False)
        for ax in ["xaxis","xaxis2","xaxis3","yaxis","yaxis2","yaxis3"]:
            fig_ta.update_layout(**{ax:dict(gridcolor=C["border2"],color=C["muted"])})
        st.plotly_chart(fig_ta,use_container_width=True)

    # Advice cards — bilingual
    st.markdown(f"<div class='section-label'>🎓 {L['personalised_advice']} · {profile}</div>",unsafe_allow_html=True)
    ADVICE = {
        0: [  # Beginner
            ("🪙","ابدأ بمبلغ صغير · Start Small",
             "خصص 5-10% فقط من مدخراتك للذهب. ETF الذهب أو السبائك المادية هي أفضل نقطة انطلاق.\nAllocate only 5–10% of savings to gold. Gold ETFs or physical bars are the best starting point."),
            ("📉","اشترِ عند الانخفاض · Buy Dips",
             "تجنب الشراء بعد ارتفاع حاد. انتظر حتى يصل RSI إلى أقل من 40 أو يلامس السعر خط MA50.\nAvoid buying after a sharp surge. Wait for RSI below 40 or a touch of MA50 support."),
            ("📆","المتوسط التكلفة الدولاري · Dollar-Cost Average",
             "اشترِ مبلغاً ثابتاً كل شهر بغض النظر عن السعر. هذا يلغي ضغط توقيت السوق.\nBuy a fixed amount monthly regardless of price. This eliminates market-timing stress."),
            ("🧘","لا تتداول بناءً على العواطف · No Emotional Trading",
             "حدد سعر الدخول ووقف الخسارة قبل كل صفقة. التزم بخطتك حتى لو تحرك السوق ضدك.\nSet entry and stop-loss before every trade. Stick to your plan even when the market moves against you."),
        ],
        1: [  # Intermediate
            ("⚙️","RSI + MA تداخل الإشارات · RSI+MA Combo",
             "أفضل نقطة دخول: السعر فوق MA50 + RSI يرتد من 40-45. مزيج يؤكد الاتجاه والزخم.\nBest entry: price above MA50 + RSI bouncing from 40–45. Confirms both trend and momentum."),
            ("🎸","انضغاط Bollinger · BB Squeeze",
             "عندما تضيق نطاقات بولينجر، تتوقع حركة حادة. الكسر خارج النطاق بحجم تداول عالٍ هو الإشارة.\nWhen BB narrows, a sharp move is coming. A close outside the band on high volume is the signal."),
            ("💱","تداول العكس مع DXY · Trade DXY Inverse",
             "الذهب والدولار الأمريكي مرتبطان بعكس بنسبة 80%. كسر DXY لدعم رئيسي = إشارة شراء للذهب.\nGold and DXY are ~80% inversely correlated. DXY breaking key support = gold buy signal."),
            ("🪜","ادخل وأخرج تدريجياً · Scale In & Out",
             "أدخل 50% عند الإشارة الأولى، 30% عند التأكيد، 20% عند التراجع. اخرج تدريجياً عند TP1 و TP2.\nEnter 50% at first signal, 30% on confirmation, 20% on pullback. Exit in tranches at TP1/TP2."),
        ],
        2: [  # Advanced
            ("🔭","توافق الأطر الزمنية المتعددة · Multi-Timeframe",
             "لا تدخل إلا عند توافق الإشارة على أسبوعي + يومي + 4 ساعات. أي غياب = تجاهل الصفقة.\nOnly trade when signal aligns on weekly trend + daily setup + 4H entry. Missing one = skip."),
            ("📋","تقرير COT · COT Report",
             "راقب تقرير CFTC أسبوعياً. عقود المضاربة الطويلة فوق 300 ألف = قمة محتملة. المراكز القصيرة المرتفعة = قاع.\nMonitor CFTC COT weekly. Speculative longs >300k = likely top. Extreme short positions = bottoms."),
            ("🪤","صيد السيولة · Liquidity Grabs",
             "حركة سريعة تحت الدعم تغلق فوقه = فخ للبيع. اشترِ عند إغلاق الشمعة فوق الدعم.\nA wick below support closing back above it is a bear trap — buy the close above support."),
            ("🕸️","تحليل الأسواق المتشابكة · Intermarket",
             "النفط صاعد + DXY هابط + VIX مرتفع + عوائد حقيقية سالبة = أقوى بيئة صعودية للذهب.\nRising oil + falling DXY + high VIX + negative real yields = maximum bullish environment."),
        ],
        3: [  # Long-term
            ("🛡️","الذهب كتحوط للتضخم · Inflation Hedge",
             "تاريخياً، الذهب يحافظ على القوة الشرائية عبر العقود. 5-15% في محفظة طويلة الأجل تحسن نسبة شارب.\nHistorically, gold preserves purchasing power over decades. 5–15% in a long-term portfolio improves Sharpe ratio."),
            ("🔄","دورة معدلات الفائدة الحقيقية · Real Rate Cycle",
             "المحرك الأول للذهب على المدى البعيد: المعدلات الحقيقية. معدلات سلبية = ثور مستدام للذهب.\nGold's primary long-term driver is real interest rates. Negative real rates = sustained gold bull market."),
            ("☽","التمويل الإسلامي · Islamic Finance",
             "الذهب المادي مباح شرعاً. العقود الآجلة/CFDs قد تنطوي على ربا. حسابات الذهب في البنوك الإسلامية هي البديل المتوافق.\nPhysical gold is halal. Futures/CFDs may involve riba. Islamic bank gold savings accounts are the compliant alternative."),
            ("⚖️","إعادة التوازن السنوي · Annual Rebalance",
             "مرة في السنة أعد توازن محفظتك. تقليص الذهب عند الارتفاع وشراء الأصول الأخرى يفرض الشراء برخص والبيع بغلاء تلقائياً.\nOnce a year, rebalance. Trimming gold after gains and buying underweight assets enforces buy-low sell-high automatically."),
        ],
    }
    prof_idx = L["profile_opts"].index(profile) if profile in L["profile_opts"] else 0
    advice_list = ADVICE.get(prof_idx, ADVICE[0])
    for rs in range(0,len(advice_list),2):
        row=advice_list[rs:rs+2]; ac=st.columns(2)
        for col,(icon,title,body) in zip(ac,row):
            with col:
                st.markdown(f"""
                <div style='background:{C['card2']};border:1px solid {C['border2']};border-radius:5px;
                            padding:16px 18px;margin-bottom:10px;position:relative;overflow:hidden;'>
                  <div style='position:absolute;top:0;left:0;bottom:0;width:3px;
                              background:linear-gradient(180deg,{C['gold']},{C['gold_dark']});'></div>
                  <div style='display:flex;align-items:center;gap:8px;margin-bottom:8px;'>
                    <span style='font-size:20px;'>{icon}</span>
                    <div style='font-family:{C['font_h']},serif;font-size:10px;font-weight:700;
                                letter-spacing:.15em;color:{C['gold_pale']};'>{title}</div>
                  </div>
                  <div style='font-family:{C['font_b']},serif;font-size:13px;color:{C['text']};
                              line-height:1.65;white-space:pre-line;'>{body}</div>
                </div>""",unsafe_allow_html=True)

    # Golden rules
    st.markdown(f"<div class='section-label'>📜 {L['golden_rules']}</div>",unsafe_allow_html=True)
    RULES_AR = [
        ("لا تخاطر بأكثر من 1-2% من رأس المال في صفقة واحدة.","Never risk more than 1–2% of capital per trade. Position sizing is the #1 survival factor."),
        ("الاتجاه صديقك — حتى ينتهي.","The trend is your friend — until it ends. Trade in the direction of MA200."),
        ("أغلق الخسائر بسرعة، ودع الأرباح تجري.","Cut losses fast, let profits run. Move stop to break-even once price hits TP1."),
        ("لا تتوسط على صفقة خاسرة.","Never average down into a losing trade. It turns a small mistake into a catastrophe."),
        ("الاقتصاد الكلي يتفوق على التحليل الفني عند نقاط التحول.","Macro beats technicals at turning points. Always know the fundamental backdrop."),
        ("الجودة تتفوق على الكمية — 1-3 صفقات متقنة أفضل من 20 متوسطة.","Quality over quantity — 1–3 high-conviction setups outperform 20 mediocre ones."),
        ("احتفظ بدفتر يوميات تداول.","Keep a trading journal. Review it monthly. Your mistakes are goldmines."),
        ("الارتباطات تتغير في الأزمات.","Correlations shift in crises. Gold initially sold off in 2008 and 2020."),
        ("تحرك السوق دائماً قبل الأخبار.","Markets always move before the news. Don't chase; anticipate."),
        ("هذا ليس نصيحة مالية — أنت مسؤول عن قراراتك.","This is not financial advice — you are responsible for your own decisions."),
    ]
    for i,(ar,en) in enumerate(RULES_AR):
        nc=C["gold"] if i%2==0 else C["accent"]
        txt = f"{ar}\n{en}" if use_ar else en
        st.markdown(f"""
        <div style='display:flex;gap:12px;align-items:flex-start;padding:10px 14px;
                    background:{C['card']};border:1px solid {C['border']};border-radius:4px;margin-bottom:5px;'>
          <div style='font-family:{C['font_h']},serif;font-size:18px;font-weight:900;
                      color:{nc};min-width:26px;'>{i+1:02d}</div>
          <div style='font-family:{C['font_b']},serif;font-size:13px;color:{C['text']};
                      line-height:1.6;white-space:pre-line;'>{txt}</div>
        </div>""",unsafe_allow_html=True)

    st.markdown(f"""
    <div style='background:{C['card']};border:1px solid {C['border']};border-radius:4px;
                padding:12px 16px;text-align:center;margin-top:1rem;'>
      <div style='font-family:{C['font_b']},serif;font-size:12px;font-style:italic;color:{C['dim']};'>
        {L['disclaimer']}
      </div>
    </div>""",unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: SETTINGS
# ═══════════════════════════════════════════════════════════════════════════════
elif nav == L["nav_settings"]:
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
        row={"Unit":un,"Symbol":uc["symbol"]}
        for pn,pc2 in PURITIES.items():
            row[pc2["label"]]=f"${g_ref*uc['factor']*pc2['mult']:,.5f}"
        rows.append(row)
    st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)

# ═══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("<div class='mansa-divider'></div>",unsafe_allow_html=True)
st.markdown(f"""
<div style='display:flex;justify-content:space-between;align-items:center;padding:.5rem 0;'>
  <span style='font-family:{C['font_h']},serif;font-size:9px;letter-spacing:.25em;color:{C['dim']};'>
    {C['brand']} &nbsp; {L['app_name']} · {L['gold_intelligence']} · {st.session_state['theme']}
  </span>
  <span style='font-family:{C['font_b']},serif;font-size:11px;font-style:italic;color:{C['dim']};'>
    Yahoo Finance · ~15 min delay · {L['not_financial']} · {datetime.datetime.now().year}
  </span>
</div>""",unsafe_allow_html=True)
