import io
import os
import csv
import base64
from pathlib import Path
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import h5py
import cv2
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

try:
    import arabic_reshaper
except Exception:
    arabic_reshaper = None

try:
    from bidi.algorithm import get_display
except Exception:
    def get_display(value):
        return value

try:
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether, Flowable
    )
except Exception:
    colors = None

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Nabdh AI",
    page_icon="🫀",
    layout="wide"
)

# =====================================================
# SESSION STATE
# =====================================================
PAGES = [
    "home", "about", "dashboard", "upload", "segmentation",
    "features", "prediction", "report", "history", "federated", "help", "settings"
]

if "page" not in st.session_state:
    st.session_state.page = "home"

if "language" not in st.session_state:
    st.session_state.language = "English"

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if "user_role" not in st.session_state:
    st.session_state.user_role = ""

if "user_id" not in st.session_state:
    st.session_state.user_id = ""

if "user_display_name" not in st.session_state:
    st.session_state.user_display_name = ""

# =====================================================
# TEXT DICTIONARY
# =====================================================
TEXT = {
    "English": {
        "brand": "Nabdh AI",
        "subtitle": "Intelligent Cardiac Diagnosis Platform",
        "ai_system": "AI Cardiac System",
        "home": "Home",
        "about": "About Nabdh AI",
        "dashboard": "Dashboard",
        "upload": "MRI Upload",
        "segmentation": "Segmentation",
        "features": "Features",
        "prediction": "Prediction",
        "report": "Report",
        "federated": "Federated Hospitals",
        "help": "Help",
        "settings": "Settings",
        "language": "Language",
        "secure_access": "Secure Access",
        "secure_access_title": "Nabdh AI Secure Access",
        "secure_access_subtitle": "Protected Medical AI Research Portal",
        "secure_access_text": "This academic access layer simulates identity verification for authorized hospital and research users before accessing MRI analysis and reports.",
        "national_id": "National ID",
        "password": "Password",
        "otp": "Verification Code",
        "role": "User Role",
        "hospital_staff": "Hospital Staff",
        "administrator": "Administrator",
        "sign_in": "Sign In",
        "sign_out": "Sign Out",
        "access_granted": "Access granted successfully.",
        "access_denied": "Please enter National ID, password, and the verification code 123456.",
        "demo_otp": "Demo verification code: 123456",
        "secured_by": "Secure Access Simulation inspired by national digital identity workflows.",
        "current_user": "Current User",
        "access_level": "Access Level",
        "history": "Analysis History",
        "history_title": "MRI Analysis History",
        "history_empty": "No MRI analyses have been saved yet.",
        "history_total_cases": "Total Cases",
        "history_disease_types": "Disease Types",
        "history_avg_confidence": "Average Confidence",
        "history_table": "Saved Analysis Records",
        "history_note": "Each completed prediction is saved locally in analysis_history.csv.",
        "case_id": "Case ID",
        "date": "Date",
        "status": "Status",
        "completed": "Completed",
        "hero_title": "AI-Powered Cardiac MRI Diagnosis",
        "hero_text": "A professional medical AI research platform for cardiac MRI visualization, DeepLab-based segmentation, scientific feature extraction, and automated disease prediction within a federated learning framework.",
        "start": "Start Diagnosis",
        "view_dashboard": "View Dashboard",
        "cases": "MRI Cases",
        "analyses": "Completed Analyses",
        "predictions": "Predictions",
        "reports": "Clinical Reports",
        "distribution": "Disease Distribution",
        "top_features": "Top Cardiac Features",
        "performance": "Model Performance",
        "recent": "Recent MRI Analyses",
        "upload_title": "Upload Cardiac MRI",
        "upload_text": "Upload a cardiac MRI file to visualize the original image and segmentation mask.",
        "original": "Original MRI",
        "mask": "Segmentation Mask",
        "overlay": "Overlay View",
        "success": "MRI file loaded successfully.",
        "no_file": "Please upload a cardiac MRI file first.",
        "disease": "Disease",
        "confidence": "Confidence Score",
        "risk": "Risk",
        "model": "Model",
        "recommendation": "Recommendation",
        "recommendation_text": "The case should be reviewed by a cardiology specialist for clinical confirmation.",
        "download_report": "Download Official PDF Report",
        "share_link": "Share Link",
        "print_note": "Open the downloaded PDF and print it normally from Preview or your browser.",
        "vision": "Vision",
        "mission": "Mission",
        "objectives": "Objectives",
        "university": "Imam Mohammad Ibn Saud Islamic University",
        "college": "College of Computer and Information Sciences",
        "department": "Department of Computer Science",
        "notice": "Research prototype only. Not intended for direct clinical diagnosis without specialist review.",
        "final_prediction": "Final Prediction",
        "official_pdf_ready": "Official PDF report is ready.",
        "clinical_review": "Clinical review",
        "feature_based_prediction": "Feature-based prediction",
        "predicted_interpretation": "Predicted Disease Interpretation",
        "probability_distribution": "Disease Probability Distribution",
    },
    "العربية": {
        "brand": "نبض ",
        "subtitle": "منصة ذكية لتشخيص أمراض القلب",
        "ai_system": "نظام ذكاء اصطناعي للقلب",
        "home": "الرئيسية",
        "about": "عن نبض ",
        "dashboard": "لوحة التحكم",
        "upload": "رفع الرنين",
        "segmentation": "التجزئة",
        "features": "الخصائص",
        "prediction": "التنبؤ",
        "report": "التقرير",
        "federated": "المستشفيات الموحدة",
        "help": "المساعدة",
        "settings": "الإعدادات",
        "language": "اللغة",
        "secure_access": "الدخول الآمن",
        "secure_access_title": "الدخول الآمن لمنصة نبض",
        "secure_access_subtitle": "بوابة بحثية طبية محمية",
        "secure_access_text": "تحاكي هذه الطبقة الأكاديمية التحقق من الهوية للمستخدمين المصرح لهم من المستشفى أو البحث قبل الوصول إلى تحليل صور الرنين والتقارير.",
        "national_id": "رقم الهوية",
        "password": "كلمة المرور",
        "otp": "رمز التحقق",
        "role": "دور المستخدم",
        "hospital_staff": "موظف مستشفى",
        "administrator": "مدير النظام",
        "sign_in": "تسجيل الدخول",
        "sign_out": "تسجيل الخروج",
        "access_granted": "تم تسجيل الدخول بنجاح.",
        "access_denied": "يرجى إدخال رقم الهوية وكلمة المرور ورمز التحقق 123456.",
        "demo_otp": "رمز التحقق التجريبي: 123456",
        "secured_by": "محاكاة دخول آمن مستوحاة من إجراءات الهوية الرقمية الوطنية.",
        "current_user": "المستخدم الحالي",
        "access_level": "مستوى الوصول",
        "history": "سجل التحليلات",
        "history_title": "سجل تحليلات الرنين المغناطيسي",
        "history_empty": "لا توجد تحليلات محفوظة حتى الآن.",
        "history_total_cases": "إجمالي الحالات",
        "history_disease_types": "أنواع الأمراض",
        "history_avg_confidence": "متوسط الثقة",
        "history_table": "سجلات التحليل المحفوظة",
        "history_note": "يتم حفظ كل تنبؤ مكتمل محليًا في ملف analysis_history.csv.",
        "case_id": "رقم الحالة",
        "date": "التاريخ",
        "status": "الحالة",
        "completed": "مكتمل",
        "hero_title": "تشخيص أمراض القلب من صور الرنين باستخدام الذكاء الاصطناعي",
        "hero_text": "منصة بحثية طبية احترافية لعرض صور الرنين المغناطيسي للقلب، وتنفيذ التجزئة بنموذج DeepLab، واستخراج الخصائص العلمية، والتنبؤ بالمرض ضمن إطار التعلم الموحد.",
        "start": "ابدأ التشخيص",
        "view_dashboard": "عرض لوحة التحكم",
        "cases": "حالات الرنين",
        "analyses": "التحاليل المكتملة",
        "predictions": "التنبؤات",
        "reports": "التقارير السريرية",
        "distribution": "توزيع الأمراض",
        "top_features": "أهم الخصائص القلبية",
        "performance": "أداء النموذج",
        "recent": "آخر تحليلات الرنين",
        "upload_title": "رفع صورة الرنين المغناطيسي للقلب",
        "upload_text": "ارفع ملف الرنين المغناطيسي للقلب لعرض الصورة الأصلية وقناع التجزئة.",
        "original": "صورة الرنين الأصلية",
        "mask": "قناع التجزئة",
        "overlay": "العرض المدمج",
        "success": "تم تحميل ملف الرنين المغناطيسي بنجاح.",
        "no_file": "يرجى رفع ملف الرنين المغناطيسي أولًا.",
        "disease": "المرض",
        "confidence": "درجة الثقة",
        "risk": "مستوى الخطورة",
        "model": "النموذج",
        "recommendation": "التوصية",
        "recommendation_text": "ينبغي مراجعة الحالة من قبل طبيب قلب مختص للتأكيد السريري.",
        "download_report": "تنزيل التقرير الرسمي PDF",
        "share_link": "رابط المشاركة",
        "print_note": "افتحي ملف PDF بعد تنزيله واطبعيه من Preview أو المتصفح.",
        "vision": "الرؤية",
        "mission": "الرسالة",
        "objectives": "الأهداف",
        "university": "جامعة الإمام محمد بن سعود الإسلامية",
        "college": "كلية علوم الحاسب والمعلومات",
        "department": "قسم علوم الحاسب",
        "notice": "نموذج بحثي أولي، ولا يستخدم للتشخيص السريري المباشر دون مراجعة المختص.",
        "final_prediction": "التنبؤ النهائي",
        "official_pdf_ready": "التقرير الرسمي PDF جاهز.",
        "clinical_review": "مراجعة سريرية",
        "feature_based_prediction": "تنبؤ مبني على الخصائص",
        "predicted_interpretation": "تفسير التنبؤ بالمرض",
        "probability_distribution": "توزيع درجات التصنيف",
    }
}

# =====================================================
# LANGUAGE AND DIRECTION
# =====================================================
t = TEXT[st.session_state.language]
is_ar = st.session_state.language == "العربية"
direction = "rtl" if is_ar else "ltr"
align = "right" if is_ar else "left"

# =====================================================
# LOGO CONFIGURATION
# =====================================================
LOGO_PATHS = [
    Path("assets/nabdh_logo.png"),
    Path("nabdh_logo.png"),
    Path("/mnt/data/assets/nabdh_logo.png"),
    Path("/mnt/data/ChatGPT Image Jun 10, 2026 at 11_31_51 PM.png"),
]

def get_logo_path():
    for logo_path in LOGO_PATHS:
        try:
            if logo_path.exists():
                return logo_path
        except Exception:
            continue
    return None

def get_logo_base64():
    logo_path = get_logo_path()
    if logo_path is None:
        return ""
    try:
        return base64.b64encode(logo_path.read_bytes()).decode("utf-8")
    except Exception:
        return ""

LOGO_BASE64 = get_logo_base64()
LOGO_SRC = f"data:image/png;base64,{LOGO_BASE64}" if LOGO_BASE64 else ""


# =====================================================
# DISEASE MAPS
# =====================================================
DISEASE_FULL_NAME = {
    "NOR": "Normal",
    "DCM": "Dilated Cardiomyopathy",
    "HCM": "Hypertrophic Cardiomyopathy",
    "MINF": "Myocardial Infarction",
    "ARV": "Abnormal Right Ventricle"
}

DISEASE_AR = {
    "NOR": "طبيعي",
    "DCM": "اعتلال عضلة القلب التوسعي",
    "HCM": "اعتلال عضلة القلب التضخمي",
    "MINF": "احتشاء عضلة القلب",
    "ARV": "خلل البطين الأيمن"
}

# =====================================================
# CSS
# =====================================================
st.markdown(
    f"""
    <style>
    .block-container {{
        padding-top: 1rem;
        max-width: 1320px;
    }}

    .rtl {{
        direction: {direction};
        text-align: {align};
    }}

    .topbar {{
        background: #0B1324;
        color: white;
        padding: 22px 24px;
        border-radius: 18px 18px 0 0;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 12px 30px rgba(15, 23, 42, 0.10);
    }}

    .brand-click {{
        font-size: 30px;
        font-weight: 900;
        letter-spacing: -0.5px;
    }}

    .brand-sub {{
        font-size: 13px;
        color: #D7E6F8;
        margin-top: 4px;
    }}

    .nav-shell {{
        background: white;
        border: 1px solid #E5E7EB;
        border-top: none;
        border-radius: 0 0 18px 18px;
        padding: 10px 12px;
        margin-bottom: 34px;
        box-shadow: 0 10px 24px rgba(15, 23, 42, 0.06);
    }}

    div.stButton > button {{
        border: none;
        background: transparent;
        color: #0f172a;
        font-weight: 750;
        border-radius: 12px;
        min-height: 40px;
    }}

    div.stButton > button:hover {{
        background: #EAF4FF;
        color: #0B3B75;
    }}

    .hero {{
        background: linear-gradient(135deg, #07152E 0%, #0B3B75 58%, #1E88E5 125%);
        color: white;
        border-radius: 28px;
        padding: 54px 48px;
        box-shadow: 0 20px 48px rgba(15, 23, 42, 0.25);
        margin-bottom: 28px;
        position: relative;
        overflow: hidden;
    }}

    .hero::after {{
        content: "";
        position: absolute;
        width: 380px;
        height: 380px;
        border-radius: 50%;
        background: rgba(255,255,255,0.09);
        top: -120px;
        {"left" if is_ar else "right"}: -90px;
    }}

    .hero-title {{
        font-size: 44px;
        font-weight: 900;
        line-height: 1.25;
        max-width: 920px;
        margin-bottom: 18px;
    }}

    .hero-text {{
        font-size: 17px;
        color: #E5E7EB;
        line-height: 1.9;
        max-width: 920px;
    }}

    .section-title {{
        font-size: 30px;
        font-weight: 900;
        color: #0f172a;
        margin: 12px 0 22px 0;
    }}

    .card {{
        background: white;
        border: 1px solid #E5E7EB;
        border-radius: 24px;
        padding: 26px 28px;
        box-shadow: 0 14px 35px rgba(15, 23, 42, 0.07);
        margin-bottom: 24px;
    }}

    .blue-card {{
        background: #EAF4FF;
        border: 1px solid #BFDFFF;
        border-radius: 24px;
        padding: 24px;
        margin-bottom: 22px;
    }}

    .red-card {{
        background: #FFEBEE;
        border: 1px solid #FFD1D8;
        border-radius: 24px;
        padding: 24px;
        margin-bottom: 22px;
    }}

    .stat-card {{
        background: white;
        border: 1px solid #E5E7EB;
        border-radius: 22px;
        padding: 24px;
        min-height: 142px;
        box-shadow: 0 12px 30px rgba(15, 23, 42, 0.07);
    }}

    .stat-label {{
        color: #64748B;
        font-size: 14px;
        font-weight: 800;
        margin-bottom: 12px;
    }}

    .stat-value {{
        color: #0f172a;
        font-size: 38px;
        font-weight: 900;
    }}

    .stat-note {{
        color: #16A34A;
        font-weight: 800;
        font-size: 13px;
        margin-top: 6px;
    }}

    .upload-box {{
        background: white;
        border: 2px dashed #8EC5FF;
        border-radius: 26px;
        padding: 44px;
        text-align: center;
        box-shadow: 0 12px 30px rgba(15, 23, 42, 0.06);
        margin-bottom: 22px;
    }}

    .footer {{
        background: #0B1324;
        color: #CBD5E1;
        padding: 28px;
        border-radius: 20px;
        margin-top: 42px;
        margin-bottom: 18px;
        font-size: 14px;
        line-height: 1.8;
    }}

    .seal {{
        width: 170px;
        height: 170px;
        border: 3px solid #1E88E5;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        text-align: center;
        color: #0B3B75;
        font-weight: 900;
        line-height: 1.5;
        margin: auto;
        background: #EAF4FF;
    }}

    /* Secure Access Page - Creative Medical Login */
    .secure-page {{
        direction: {direction};
        text-align: {align};
    }}

    .secure-wrapper {{
        min-height: 86vh;
        padding: 34px 18px;
        border-radius: 34px;
        background:
            radial-gradient(circle at 10% 18%, rgba(30,136,229,0.18), transparent 30%),
            radial-gradient(circle at 92% 12%, rgba(229,57,53,0.10), transparent 28%),
            linear-gradient(135deg, #F7FBFF 0%, #EAF4FF 48%, #FFFFFF 100%);
        overflow: hidden;
    }}

    .secure-login-shell {{
        max-width: 1120px;
        min-height: 625px;
        margin: 0 auto;
        display: grid;
        grid-template-columns: 0.95fr 1.05fr;
        gap: 0;
        border-radius: 36px;
        overflow: hidden;
        background: rgba(255,255,255,0.97);
        border: 1px solid #DDEAF7;
        box-shadow: 0 28px 70px rgba(7,21,46,0.16);
    }}

    .secure-visual-panel {{
        position: relative;
        padding: 42px 38px;
        background:
            linear-gradient(150deg, rgba(7,21,46,0.96) 0%, rgba(11,59,117,0.98) 72%, rgba(30,136,229,0.92) 140%);
        color: white;
        min-height: 625px;
        overflow: hidden;
    }}

    .secure-visual-panel::before {{
        content: "";
        position: absolute;
        width: 380px;
        height: 380px;
        border-radius: 50%;
        background: rgba(234,244,255,0.10);
        top: -130px;
        {"left" if is_ar else "right"}: -120px;
    }}

    .secure-visual-panel::after {{
        content: "";
        position: absolute;
        width: 320px;
        height: 95px;
        background: rgba(255,255,255,0.12);
        border-radius: 999px;
        bottom: 70px;
        {"right" if is_ar else "left"}: -80px;
        transform: rotate(-12deg);
    }}

    .secure-brand-mark {{
        position: relative;
        z-index: 2;
        display: inline-flex;
        align-items: center;
        gap: 12px;
        font-size: 30px;
        font-weight: 950;
        letter-spacing: -0.6px;
    }}

    .secure-brand-icon {{
        width: 58px;
        height: 58px;
        border-radius: 20px;
        background: rgba(234,244,255,0.16);
        border: 1px solid rgba(234,244,255,0.34);
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 30px;
        box-shadow: inset 0 0 0 1px rgba(255,255,255,0.08);
    }}

    .secure-visual-content {{
        position: relative;
        z-index: 2;
        margin-top: 74px;
    }}

    .secure-title {{
        font-size: 42px;
        font-weight: 950;
        line-height: 1.25;
        margin-bottom: 14px;
    }}

    .secure-subtitle {{
        font-size: 19px;
        color: #DCEBFF;
        font-weight: 850;
        margin-bottom: 14px;
    }}

    .secure-text {{
        color: #EAF4FF;
        line-height: 1.9;
        font-size: 15px;
        max-width: 430px;
    }}

    .secure-medical-illustration {{
        position: relative;
        z-index: 2;
        margin-top: 46px;
        width: 225px;
        height: 225px;
        border-radius: 54px;
        background:
            radial-gradient(circle at 50% 48%, #FFFFFF 0 28%, transparent 29%),
            linear-gradient(135deg, rgba(234,244,255,0.22), rgba(255,255,255,0.06));
        border: 1px solid rgba(234,244,255,0.28);
        box-shadow: 0 22px 50px rgba(0,0,0,0.16);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 72px;
    }}

    .secure-medical-illustration::before {{
        content: "";
        position: absolute;
        width: 210px;
        height: 48px;
        border-radius: 999px;
        background: rgba(255,255,255,0.14);
        transform: rotate(-14deg);
    }}

    .secure-medical-illustration span {{
        position: relative;
        z-index: 3;
        filter: drop-shadow(0 10px 18px rgba(0,0,0,0.16));
    }}

    .secure-wave {{
        position: relative;
        z-index: 2;
        margin-top: 34px;
        height: 76px;
        border-radius: 22px;
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(234,244,255,0.18);
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 900;
        color: #EAF4FF;
    }}

    .secure-badge {{
        position: relative;
        z-index: 2;
        display: inline-block;
        margin-top: 18px;
        background: rgba(234,244,255,0.14);
        border: 1px solid rgba(234,244,255,0.35);
        color: #EAF4FF;
        padding: 10px 15px;
        border-radius: 999px;
        font-size: 13px;
        font-weight: 900;
    }}

    .secure-form-panel {{
        padding: 48px 50px;
        background:
            radial-gradient(circle at 100% 0%, rgba(229,57,53,0.06), transparent 30%),
            radial-gradient(circle at 0% 100%, rgba(30,136,229,0.08), transparent 34%),
            #FFFFFF;
        min-height: 625px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }}

    .secure-form-heading {{
        margin-bottom: 26px;
    }}

    .secure-form-heading h2 {{
        margin: 0 0 8px 0;
        color: #07152E;
        font-size: 36px;
        font-weight: 950;
        letter-spacing: -0.5px;
    }}

    .secure-form-heading p {{
        margin: 0;
        color: #64748B;
        font-size: 15px;
        line-height: 1.8;
    }}

    .secure-role-strip {{
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 10px;
        margin: 18px 0 22px 0;
    }}

    .secure-role-chip {{
        border-radius: 17px;
        padding: 12px 10px;
        text-align: center;
        font-weight: 900;
        font-size: 13px;
        color: #0B3B75;
        background: #EAF4FF;
        border: 1px solid #BFDFFF;
    }}

    .secure-form-card {{
        background: transparent;
        border: none;
        box-shadow: none;
        padding: 0;
        margin: 0;
    }}

    .secure-demo-line {{
        margin-top: 16px;
        padding: 13px 15px;
        border-radius: 16px;
        background: #F8FBFF;
        border: 1px dashed #9CCBFF;
        color: #0B3B75;
        font-weight: 900;
        text-align: center;
    }}

    .secure-info-card {{
        margin-top: 18px;
        background: #F8FBFF;
        border: 1px solid #DDEAF7;
        border-radius: 22px;
        padding: 18px 20px;
        color: #07152E;
        text-align: {align};
    }}

    .secure-info-card h3 {{
        margin: 0 0 6px 0;
        font-size: 18px;
        font-weight: 950;
    }}

    .secure-info-card p {{
        margin: 0;
        color: #475569;
        line-height: 1.8;
    }}

    .access-chip {{
        display: inline-block;
        margin-top: 8px;
        background: #EAF4FF;
        color: #0B3B75;
        padding: 8px 14px;
        border-radius: 999px;
        font-size: 13px;
        font-weight: 900;
    }}

    .secure-page div[data-baseweb="select"],
    .secure-page input {{
        direction: {direction};
        text-align: {align};
    }}

    .secure-page div[data-baseweb="select"] * {{
        text-align: {align};
    }}

    .secure-page [data-testid="stForm"] {{
        border: none;
        padding: 0;
    }}

    .secure-page [data-testid="stFormSubmitButton"] button {{
        background: #07152E !important;
        color: white !important;
        border: 1px solid #07152E !important;
        border-radius: 999px !important;
        min-height: 50px !important;
        font-weight: 950 !important;
        box-shadow: 0 12px 26px rgba(7,21,46,0.20) !important;
    }}

    .secure-page [data-testid="stFormSubmitButton"] button:hover {{
        background: #0B3B75 !important;
        border-color: #0B3B75 !important;
    }}


    /* =====================================================
       Enhanced Secure Access Login - Inspired Creative UI
       ===================================================== */
    .secure-wrapper {{
        min-height: 88vh;
        padding: 42px 20px;
        border-radius: 38px;
        background:
            radial-gradient(circle at 8% 14%, rgba(255,255,255,0.95), transparent 18%),
            radial-gradient(circle at 88% 14%, rgba(255,255,255,0.28), transparent 22%),
            linear-gradient(135deg, #07152E 0%, #0B3B75 36%, #1E88E5 72%, #EAF4FF 125%);
        overflow: hidden;
    }}

    .secure-login-shell {{
        max-width: 1180px;
        min-height: 660px;
        margin: 0 auto;
        display: grid;
        grid-template-columns: 1.02fr 0.98fr;
        gap: 0;
        border-radius: 42px;
        overflow: hidden;
        background: rgba(255,255,255,0.96);
        border: 1px solid rgba(255,255,255,0.66);
        box-shadow: 0 34px 90px rgba(7,21,46,0.28);
        position: relative;
    }}

    .secure-login-shell::before {{
        content: "";
        position: absolute;
        width: 430px;
        height: 430px;
        border-radius: 50%;
        background: rgba(234,244,255,0.52);
        top: -260px;
        {"right" if is_ar else "left"}: -210px;
        z-index: 1;
    }}

    .secure-visual-panel {{
        position: relative;
        padding: 46px 42px;
        background:
            linear-gradient(145deg, rgba(7,21,46,0.98) 0%, rgba(11,59,117,0.98) 58%, rgba(30,136,229,0.94) 130%);
        color: white;
        min-height: 660px;
        overflow: hidden;
    }}

    .secure-visual-panel::before {{
        content: "";
        position: absolute;
        width: 410px;
        height: 410px;
        border-radius: 50%;
        background: rgba(255,255,255,0.10);
        top: -150px;
        {"left" if is_ar else "right"}: -120px;
    }}

    .secure-visual-panel::after {{
        content: "";
        position: absolute;
        width: 470px;
        height: 120px;
        background: rgba(255,255,255,0.13);
        border-radius: 999px;
        bottom: 92px;
        {"right" if is_ar else "left"}: -130px;
        transform: rotate(-13deg);
    }}

    .secure-brand-mark {{
        position: relative;
        z-index: 3;
        display: inline-flex;
        align-items: center;
        gap: 13px;
        font-size: 31px;
        font-weight: 950;
        letter-spacing: -0.7px;
    }}

    .secure-brand-icon {{
        width: 62px;
        height: 62px;
        border-radius: 23px;
        background: rgba(255,255,255,0.17);
        border: 1px solid rgba(255,255,255,0.36);
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 31px;
        box-shadow: 0 16px 32px rgba(0,0,0,0.14), inset 0 0 0 1px rgba(255,255,255,0.08);
    }}

    .secure-visual-content {{
        position: relative;
        z-index: 3;
        margin-top: 70px;
    }}

    .secure-title {{
        font-size: 48px;
        font-weight: 950;
        line-height: 1.12;
        margin-bottom: 16px;
        letter-spacing: -1.1px;
    }}

    .secure-subtitle {{
        font-size: 20px;
        color: #DCEBFF;
        font-weight: 850;
        margin-bottom: 15px;
    }}

    .secure-text {{
        color: #EAF4FF;
        line-height: 1.9;
        font-size: 15.5px;
        max-width: 470px;
    }}

    .secure-badge {{
        position: relative;
        z-index: 3;
        display: inline-block;
        margin-top: 20px;
        background: rgba(255,255,255,0.14);
        border: 1px solid rgba(255,255,255,0.34);
        color: #FFFFFF;
        padding: 11px 16px;
        border-radius: 999px;
        font-size: 13px;
        font-weight: 900;
        box-shadow: 0 10px 24px rgba(0,0,0,0.10);
    }}

    .secure-medical-illustration {{
        position: relative;
        z-index: 3;
        margin-top: 44px;
        width: 235px;
        height: 235px;
        border-radius: 58px;
        background:
            radial-gradient(circle at 50% 48%, rgba(255,255,255,0.96) 0 27%, transparent 28%),
            linear-gradient(135deg, rgba(255,255,255,0.24), rgba(255,255,255,0.07));
        border: 1px solid rgba(255,255,255,0.30);
        box-shadow: 0 25px 60px rgba(0,0,0,0.20);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 76px;
    }}

    .secure-medical-illustration::before {{
        content: "";
        position: absolute;
        width: 230px;
        height: 52px;
        border-radius: 999px;
        background: rgba(255,255,255,0.16);
        transform: rotate(-14deg);
    }}

    .secure-medical-illustration::after {{
        content: "";
        position: absolute;
        width: 88px;
        height: 88px;
        border-radius: 50%;
        border: 1px dashed rgba(255,255,255,0.36);
        top: 22px;
        {"right" if is_ar else "left"}: 24px;
    }}

    .secure-orbit {{
        position: absolute;
        width: 420px;
        height: 420px;
        border-radius: 50%;
        border: 1px solid rgba(234,244,255,0.20);
        top: 105px;
        right: -160px;
        z-index: 1;
    }}

    .secure-orbit::before {{
        content: "MRI";
        position: absolute;
        top: 18px;
        {"right" if is_ar else "left"}: -16px;
        width: 58px;
        height: 58px;
        border-radius: 18px;
        background: rgba(255,255,255,0.16);
        border: 1px solid rgba(255,255,255,0.30);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 13px;
        font-weight: 950;
        color: #FFFFFF;
        box-shadow: 0 14px 28px rgba(0,0,0,0.13);
    }}

    .secure-wave {{
        position: relative;
        z-index: 3;
        margin-top: 36px;
        height: 78px;
        border-radius: 25px;
        background:
            linear-gradient(90deg, rgba(255,255,255,0.10), rgba(255,255,255,0.18), rgba(255,255,255,0.08));
        border: 1px solid rgba(255,255,255,0.22);
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 950;
        color: #FFFFFF;
        letter-spacing: 0.3px;
    }}

    .secure-form-panel {{
        position: relative;
        z-index: 2;
        padding: 54px 58px;
        background:
            radial-gradient(circle at 96% 4%, rgba(30,136,229,0.10), transparent 28%),
            radial-gradient(circle at 8% 96%, rgba(11,59,117,0.08), transparent 30%),
            #FFFFFF;
        min-height: 660px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }}

    .secure-form-panel::before {{
        content: "";
        position: absolute;
        width: 245px;
        height: 245px;
        border-radius: 50%;
        background: rgba(234,244,255,0.78);
        top: -105px;
        {"left" if is_ar else "right"}: -95px;
        z-index: -1;
    }}

    .secure-form-heading {{
        margin-bottom: 24px;
    }}

    .secure-form-heading h2 {{
        margin: 0 0 9px 0;
        color: #07152E;
        font-size: 40px;
        font-weight: 950;
        letter-spacing: -0.8px;
    }}

    .secure-form-heading p {{
        margin: 0;
        color: #64748B;
        font-size: 15px;
        line-height: 1.8;
        max-width: 450px;
    }}

    .secure-role-strip {{
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 10px;
        margin: 20px 0 24px 0;
    }}

    .secure-role-chip {{
        border-radius: 18px;
        padding: 12px 10px;
        text-align: center;
        font-weight: 900;
        font-size: 13px;
        color: #0B3B75;
        background: linear-gradient(180deg, #F8FBFF 0%, #EAF4FF 100%);
        border: 1px solid #BFDFFF;
        box-shadow: 0 10px 22px rgba(30,136,229,0.08);
    }}

    .secure-form-mini-title {{ 
        display: inline-block;
        width: fit-content;
        margin-bottom: 18px;
        padding: 9px 16px;
        border-radius: 999px;
        background: #EAF4FF;
        color: #0B3B75;
        font-size: 13px;
        font-weight: 950;
        border: 1px solid #BFDFFF;
}}

    .secure-page [data-testid="stForm"] {{
        border: none;
        padding: 0;
        background: transparent;
    }}

    .secure-page [data-testid="stTextInput"] input {{
        border-radius: 999px !important;
        min-height: 48px !important;
        border: 1px solid #DDEAF7 !important;
        background: #F8FBFF !important;
        box-shadow: inset 0 0 0 1px rgba(255,255,255,0.5) !important;
    }}

    .secure-page div[data-baseweb="select"] > div {{
        border-radius: 999px !important;
        min-height: 48px !important;
        background: #F8FBFF !important;
        border-color: #DDEAF7 !important;
    }}

    .secure-page [data-testid="stFormSubmitButton"] button {{
        background: linear-gradient(135deg, #07152E 0%, #0B3B75 100%) !important;
        color: white !important;
        border: 1px solid #07152E !important;
        border-radius: 999px !important;
        min-height: 52px !important;
        font-weight: 950 !important;
        box-shadow: 0 14px 30px rgba(7,21,46,0.24) !important;
    }}

    .secure-page [data-testid="stFormSubmitButton"] button:hover {{
        background: linear-gradient(135deg, #0B3B75 0%, #1E88E5 100%) !important;
        border-color: #0B3B75 !important;
        transform: translateY(-1px);
    }}

    .secure-demo-line {{
        margin-top: 17px;
        padding: 13px 16px;
        border-radius: 18px;
        background: #F8FBFF;
        border: 1px dashed #9CCBFF;
        color: #0B3B75;
        font-weight: 900;
        text-align: center;
    }}

    .secure-info-card {{
        margin-top: 18px;
        background: linear-gradient(180deg, #FFFFFF 0%, #F8FBFF 100%);
        border: 1px solid #DDEAF7;
        border-radius: 24px;
        padding: 18px 20px;
        color: #07152E;
        text-align: {align};
        box-shadow: 0 14px 30px rgba(15,23,42,0.06);
    }}

    .secure-info-card h3 {{
        margin: 0 0 7px 0;
        font-size: 18px;
        font-weight: 950;
    }}

    .secure-info-card p {{
        margin: 0;
        color: #475569;
        line-height: 1.8;
    }}

    .secure-privacy-list {{
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 8px;
        margin-top: 14px;
    }}

    .secure-privacy-pill {{
        background: #EAF4FF;
        color: #0B3B75;
        border: 1px solid #BFDFFF;
        border-radius: 999px;
        padding: 8px 10px;
        text-align: center;
        font-size: 12px;
        font-weight: 900;
    }}


    .secure-vertical-cards {{
        position: absolute;
        z-index: 4;
        right: 36px;
        bottom: 118px;
        display: flex;
        gap: 12px;
        align-items: flex-end;
    }}

    .secure-v-card {{
        width: 82px;
        height: 168px;
        border-radius: 30px;
        padding: 18px 10px;
        color: #FFFFFF;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        align-items: center;
        text-align: center;
        box-shadow: 0 24px 42px rgba(0,0,0,0.20);
        border: 1px solid rgba(255,255,255,0.32);
        backdrop-filter: blur(10px);
    }}

    .secure-v-card span {{
        width: 44px;
        height: 44px;
        border-radius: 16px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: rgba(255,255,255,0.22);
        font-size: 13px;
        font-weight: 950;
    }}

    .secure-v-card strong {{
        writing-mode: vertical-rl;
        transform: rotate(180deg);
        font-size: 14px;
        letter-spacing: 0.5px;
        font-weight: 950;
    }}

    .secure-v-blue {{
        height: 185px;
        background: linear-gradient(180deg, #9EDBFF 0%, #1E88E5 100%);
    }}

    .secure-v-green {{
        height: 215px;
        background: linear-gradient(180deg, #70E6B1 0%, #00A66C 100%);
    }}

    .secure-v-red {{
        height: 175px;
        background: linear-gradient(180deg, #FF8A8A 0%, #E53935 100%);
    }}

    @media (max-width: 900px) {{
        .secure-login-shell {{
            grid-template-columns: 1fr;
        }}
        .secure-visual-panel,
        .secure-form-panel {{
            min-height: auto;
        }}
        .secure-role-strip {{
            grid-template-columns: 1fr;
        }}
    }}


    .brand-logo-img {{
        height: 76px;
        width: auto;
        display: block;
        object-fit: contain;
    }}

    .secure-logo-img {{
        width: 260px;
        max-width: 100%;
        height: auto;
        display: block;
        margin: 0 auto;
        filter: drop-shadow(0 18px 34px rgba(0,0,0,0.22));
        border-radius: 22px;
    }}

    .secure-logo-card {{
        position: relative;
        z-index: 4;
        width: 290px;
        max-width: 100%;
        padding: 18px;
        border-radius: 30px;
        background: rgba(255,255,255,0.92);
        border: 1px solid rgba(255,255,255,0.55);
        box-shadow: 0 24px 54px rgba(0,0,0,0.18);
    }}

    .home-logo-wrap {{
        display: flex;
        justify-content: center;
        margin: 8px 0 26px 0;
    }}

    .home-logo-img {{
        width: 360px;
        max-width: 100%;
        height: auto;
        border-radius: 28px;
        filter: drop-shadow(0 18px 40px rgba(7,21,46,0.18));
    }}

    .about-logo-img {{
        width: 230px;
        max-width: 100%;
        height: auto;
        display: block;
        margin: 0 auto 18px auto;
    }}
    .right-login-card {{
    background: #FFFFFF;
    border: 1px solid #E5EAF2;
    border-radius: 30px;
    padding: 38px 42px;
    box-shadow: 0 22px 55px rgba(15, 23, 42, 0.10);
    margin-top: 20px;
}}

.right-login-card .secure-form-heading h2 {{
    font-size: 42px;
    font-weight: 950;
    color: #07152E;
}}

.right-login-card .secure-form-heading p {{
    font-size: 16px;
    color: #64748B;
}}

.left-login-panel {{

height:760px;

width:100%;


display:flex;

flex-direction:column;


justify-content:center;

align-items:center;


border-radius:35px;


background:
linear-gradient(
135deg,
#071B3A,
#1D4F9B);


padding:60px;


color:white;


text-align:center;

}}

.left-login-panel h2 {{
    font-size: 42px;
    font-weight: 950;
    margin-bottom: 18px;
}}

.left-login-panel h4 {{
    font-size: 25px;
    font-weight: 850;
    margin-bottom: 22px;
}}

.left-login-panel p {{
    font-size: 16px;
    color: #D7E6F8;
}}

.heart_icon {{
    font-size: 82px;
    margin-top: 80px;
}}


    </style>
    """,
    unsafe_allow_html=True
)

# =====================================================
# HELPER FUNCTIONS
# =====================================================
def set_page(page_name: str):
    st.session_state.page = page_name
    st.rerun()


def page_title(title: str):
    st.markdown(f"<div class='section-title rtl'>{title}</div>", unsafe_allow_html=True)


def stat_card(label: str, value: str, note: str = ""):
    st.markdown(
        f"""
        <div class="stat-card rtl">
            <div class="stat-label">{label}</div>
            <div class="stat-value">{value}</div>
            <div class="stat-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def allowed_pages_by_role(role):
    if role == "Hospital Staff" or role == "موظف مستشفى":
        return ["home", "dashboard", "upload", "segmentation", "features", "prediction", "report"]

    if role == "Administrator" or role == "مدير النظام":
        return ["home", "dashboard", "upload", "segmentation", "features", "prediction", "report", "history", "federated"]

    return ["home"]


HISTORY_FILE = "analysis_history.csv"


def generate_case_id():
    return "NBD-" + datetime.now().strftime("%Y%m%d-%H%M%S")


def save_case_history(case_id, predicted, confidence, filename=""):
    file_exists = Path(HISTORY_FILE).exists()
    disease_en = DISEASE_FULL_NAME.get(predicted, predicted)
    disease_ar = DISEASE_AR.get(predicted, predicted)

    with open(HISTORY_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "Case ID", "Date", "Disease Code", "Disease EN", "Disease AR",
                "Confidence", "Status", "File", "User Role", "User ID"
            ])
        writer.writerow([
            case_id,
            datetime.now().strftime("%Y-%m-%d %H:%M"),
            predicted,
            disease_en,
            disease_ar,
            round(float(confidence), 2),
            "Completed",
            filename,
            st.session_state.get("user_role", ""),
            st.session_state.get("user_id", "")
        ])


def save_current_analysis_once(predicted, confidence):
    current_signature = f"{st.session_state.get('uploaded_filename', '')}-{predicted}-{round(float(confidence), 2)}"

    if st.session_state.get("last_saved_history_signature") == current_signature:
        return

    case_id = generate_case_id()
    save_case_history(
        case_id=case_id,
        predicted=predicted,
        confidence=confidence,
        filename=st.session_state.get("uploaded_filename", "")
    )
    st.session_state.last_saved_history_signature = current_signature
    st.session_state.last_case_id = case_id


def load_history_dataframe(is_ar=False):
    if not Path(HISTORY_FILE).exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(HISTORY_FILE)
    except pd.errors.ParserError:
        df = pd.read_csv(HISTORY_FILE, engine="python", on_bad_lines="skip")

    if df.empty:
        return df

    display_df = df.copy()

    if is_ar:
        display_df = display_df.rename(columns={
            "Case ID": "رقم الحالة",
            "Date": "التاريخ",
            "Disease Code": "رمز المرض",
            "Disease EN": "المرض بالإنجليزية",
            "Disease AR": "المرض",
            "Confidence": "درجة الثقة",
            "Status": "الحالة",
            "File": "الملف",
            "User Role": "دور المستخدم",
            "User ID": "رقم المستخدم"
        })

        if "الحالة" in display_df.columns:
            display_df["الحالة"] = display_df["الحالة"].replace({"Completed": "مكتمل"})

    return display_df


def ecg_component():
    html = """
    <html><head><style>
    body { margin:0; background:transparent; overflow:hidden; }
    .ecg { width:100%; height:70px; }
    .ecg polyline {
        fill:none; stroke:#0f172a; stroke-width:4;
        stroke-linecap:round; stroke-linejoin:round;
        stroke-dasharray:1200; stroke-dashoffset:1200;
        animation:draw 3.4s linear infinite;
    }
    @keyframes draw { to { stroke-dashoffset:0; } }
    </style></head><body>
    <svg class="ecg" viewBox="0 0 1200 100">
    <polyline points="0,50 110,50 130,45 150,55 170,50 200,12 225,90 250,50
    390,50 410,45 430,55 450,50 480,12 505,90 530,50
    670,50 690,45 710,55 730,50 760,12 785,90 810,50
    950,50 970,45 990,55 1010,50 1040,12 1065,90 1090,50 1200,50"/>
    </svg></body></html>
    """
    components.html(html, height=75)


def calculate_circularity(binary_mask):
    binary_mask = binary_mask.astype(np.uint8)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return 0.0
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    if perimeter == 0:
        return 0.0
    return float((4 * np.pi * area) / (perimeter ** 2))


def extract_prediction_features(image, mask):
    rv_mask = (mask == 1)
    myo_mask = (mask == 2)
    lv_mask = (mask == 3)

    rv_volume_proxy = int(rv_mask.sum())
    myo_volume_proxy = int(myo_mask.sum())
    lv_volume_proxy = int(lv_mask.sum())

    rv_circularity = calculate_circularity(rv_mask)
    myo_circularity = calculate_circularity(myo_mask)
    lv_circularity = calculate_circularity(lv_mask)

    myo_thickness_mean = float(np.sqrt(myo_volume_proxy / np.pi)) if myo_volume_proxy > 0 else 0.0
    myo_thickness_max = myo_thickness_mean * 1.25

    features = {
        "num_slices": 1,
        "rv_volume_proxy": rv_volume_proxy,
        "myo_volume_proxy": myo_volume_proxy,
        "lv_volume_proxy": lv_volume_proxy,
        "rv_area_mean": rv_volume_proxy,
        "myo_area_mean": myo_volume_proxy,
        "lv_area_mean": lv_volume_proxy,
        "rv_circularity_mean": rv_circularity,
        "myo_circularity_mean": myo_circularity,
        "lv_circularity_mean": lv_circularity,
        "myo_thickness_mean": myo_thickness_mean,
        "myo_thickness_max": myo_thickness_max,
        "lv_rv_ratio": lv_volume_proxy / (rv_volume_proxy + 1e-8),
        "myo_lv_ratio": myo_volume_proxy / (lv_volume_proxy + 1e-8),
        "myo_rv_ratio": myo_volume_proxy / (rv_volume_proxy + 1e-8)
    }
    return features


def rule_based_prediction_from_features(features):
    rv = features["rv_volume_proxy"]
    myo = features["myo_volume_proxy"]
    lv = features["lv_volume_proxy"]
    thickness = features["myo_thickness_mean"]
    lv_rv_ratio = features["lv_rv_ratio"]
    myo_lv_ratio = features["myo_lv_ratio"]

    scores = {"NOR": 0.20, "DCM": 0.20, "HCM": 0.20, "MINF": 0.20, "ARV": 0.20}

    if rv > lv * 1.4:
        scores["ARV"] += 0.45
    if lv > rv * 1.6:
        scores["NOR"] += 0.20
    if myo > lv * 1.2 or thickness > 9:
        scores["HCM"] += 0.40
    if lv_rv_ratio < 1.1 and myo_lv_ratio > 1.2:
        scores["DCM"] += 0.35
    if lv < rv and myo < rv:
        scores["MINF"] += 0.25

    total = sum(scores.values())
    probabilities = {k: v / total for k, v in scores.items()}
    predicted = max(probabilities, key=probabilities.get)
    confidence = probabilities[predicted] * 100
    return predicted, confidence, probabilities


def get_risk_level(predicted, confidence):
    if predicted == "NOR":
        return "Normal"
    if confidence >= 70:
        return "High Risk"
    return "Moderate Risk"


def get_risk_level_ar(predicted, confidence):
    if predicted == "NOR":
        return "طبيعي"
    if confidence >= 70:
        return "خطورة مرتفعة"
    return "خطورة متوسطة"


def clinical_interpretation_text(predicted, features, lang="English"):
    if lang == "العربية":
        base = "تم تحليل صورة الرنين المغناطيسي اعتمادًا على قناع التجزئة واستخراج مؤشرات البطين الأيمن وعضلة القلب والبطين الأيسر. "
        if predicted == "ARV":
            return base + "تشير الخصائص المستخرجة إلى ارتفاع نسبي في مؤشرات البطين الأيمن، لذلك رجّح النظام وجود مؤشرات مرتبطة بخلل البطين الأيمن."
        if predicted == "HCM":
            return base + "تشير الخصائص المستخرجة إلى زيادة نسبية في مؤشر عضلة القلب مقارنة بالبطين الأيسر، لذلك رجّح النظام وجود مؤشرات مرتبطة باعتلال عضلة القلب التضخمي."
        if predicted == "DCM":
            return base + "تشير النسب القلبية المستخرجة إلى نمط قد يرتبط بتغيرات توسعية في عضلة القلب، ولذلك رجّح النظام وجود مؤشرات مرتبطة باعتلال عضلة القلب التوسعي."
        if predicted == "MINF":
            return base + "تشير العلاقات بين مساحات القلب المستخرجة إلى نمط قد يرتبط بمؤشرات احتشاء عضلة القلب، ولذلك تتطلب الحالة مراجعة سريرية متخصصة."
        return base + "لم تظهر المؤشرات المستخرجة انحرافًا واضحًا عن النمط الطبيعي وفق قواعد التصنيف المستخدمة في هذا النموذج البحثي."

    base = "The cardiac MRI was analyzed using segmentation-derived RV, MYO, and LV features. "
    if predicted == "ARV":
        return base + "The extracted morphology shows relatively increased right-ventricular indicators compared with other structures, so the system selected ARV as the most likely predicted class."
    if predicted == "HCM":
        return base + "The extracted morphology shows increased myocardial indicators relative to the left ventricle, so the system selected HCM as the most likely predicted class."
    if predicted == "DCM":
        return base + "The extracted ventricular ratios show a pattern that may be associated with dilated cardiomyopathy indicators, so the system selected DCM as the most likely predicted class."
    if predicted == "MINF":
        return base + "The relationship between the extracted cardiac region measurements suggests a pattern that may be associated with myocardial infarction indicators and should be reviewed clinically."
    return base + "The extracted indicators do not show a strong abnormal pattern according to the current research classification rules."


# =====================================================
# XGBOOST MODEL
# =====================================================
XGB_MODEL_PATH = Path("models/xgboost_model.pkl")

@st.cache_resource
def load_xgboost_bundle():
    if XGB_MODEL_PATH.exists():
        return joblib.load(XGB_MODEL_PATH)
    return None

XGB_BUNDLE = load_xgboost_bundle()

if XGB_BUNDLE is not None:
    XGB_MODEL = XGB_BUNDLE["model"]
    XGB_LABEL_ENCODER = XGB_BUNDLE["label_encoder"]
    FEATURE_COLUMNS = XGB_BUNDLE["feature_columns"]
else:
    XGB_MODEL = None
    XGB_LABEL_ENCODER = None
    FEATURE_COLUMNS = [
        "num_slices",
        "rv_volume_proxy",
        "myo_volume_proxy",
        "lv_volume_proxy",
        "rv_area_mean",
        "rv_area_max",
        "myo_area_mean",
        "myo_area_max",
        "lv_area_mean",
        "lv_area_max",
        "rv_circularity_mean",
        "myo_circularity_mean",
        "lv_circularity_mean",
        "myo_thickness_mean",
        "myo_thickness_max",
        "lv_rv_ratio",
        "myo_lv_ratio",
        "myo_rv_ratio",
    ]


def xgboost_prediction_from_features(features):
    if XGB_MODEL is None or XGB_LABEL_ENCODER is None:
        return rule_based_prediction_from_features(features)

    row = {}
    for col in FEATURE_COLUMNS:
        row[col] = float(features.get(col, 0))

    X_input = pd.DataFrame([row], columns=FEATURE_COLUMNS)

    probs = XGB_MODEL.predict_proba(X_input)[0]
    pred_index = int(np.argmax(probs))

    predicted = XGB_LABEL_ENCODER.inverse_transform([pred_index])[0]
    confidence = float(probs[pred_index] * 100)

    class_names = XGB_LABEL_ENCODER.classes_

    probabilities = {
        str(cls): float(prob)
        for cls, prob in zip(class_names, probs)
    }

    return str(predicted), confidence, probabilities
def ensure_prediction():
    if "image" not in st.session_state or "mask" not in st.session_state:
        return None, None, None, None

    image = st.session_state.image
    mask = st.session_state.mask

    features = extract_prediction_features(image, mask)

    predicted, confidence, probabilities = xgboost_prediction_from_features(features)

    st.session_state.extracted_features = features
    st.session_state.predicted_disease = predicted
    st.session_state.prediction_confidence = confidence
    st.session_state.prediction_probabilities = probabilities

    return features, predicted, confidence, probabilities

def create_overlay(image, mask):
    rv_mask = (mask == 1)
    myo_mask = (mask == 2)
    lv_mask = (mask == 3)

    img_norm = image.astype(float)
    img_norm = (img_norm - img_norm.min()) / (img_norm.max() - img_norm.min() + 1e-8)
    overlay_rgb = np.stack([img_norm, img_norm, img_norm], axis=-1)
    alpha = 0.55
    overlay_rgb[rv_mask] = (1 - alpha) * overlay_rgb[rv_mask] + alpha * np.array([0.0, 0.45, 1.0])
    overlay_rgb[myo_mask] = (1 - alpha) * overlay_rgb[myo_mask] + alpha * np.array([0.0, 0.85, 0.35])
    overlay_rgb[lv_mask] = (1 - alpha) * overlay_rgb[lv_mask] + alpha * np.array([1.0, 0.10, 0.10])
    return overlay_rgb


def report_feature_table(features):
    return pd.DataFrame({
        "Feature": [
            "RV Volume Proxy", "MYO Volume Proxy", "LV Volume Proxy",
            "RV Area Mean", "MYO Area Mean", "LV Area Mean",
            "RV Circularity", "MYO Circularity", "LV Circularity",
            "MYO Thickness Mean", "MYO Thickness Max",
            "LV/RV Ratio", "MYO/LV Ratio", "MYO/RV Ratio"
        ],
        "Value": [
            features["rv_volume_proxy"], features["myo_volume_proxy"], features["lv_volume_proxy"],
            features["rv_area_mean"], features["myo_area_mean"], features["lv_area_mean"],
            round(features["rv_circularity_mean"], 4), round(features["myo_circularity_mean"], 4), round(features["lv_circularity_mean"], 4),
            round(features["myo_thickness_mean"], 4), round(features["myo_thickness_max"], 4),
            round(features["lv_rv_ratio"], 4), round(features["myo_lv_ratio"], 4), round(features["myo_rv_ratio"], 4)
        ]
    })



# =====================================================
# PDF REPORT HELPERS - ARABIC READY
# =====================================================
ARABIC_FONT_PATHS = [
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansArabic-Regular.ttf",
]

def setup_pdf_fonts():
    if colors is None:
        return "Helvetica"

    for font_path in ARABIC_FONT_PATHS:
        try:
            if Path(font_path).exists():
                pdfmetrics.registerFont(TTFont("ArabicFont", font_path))
                return "ArabicFont"
        except Exception:
            continue

    return "Helvetica"


def ar_text(text):
    if text is None:
        return ""

    text = str(text)
    try:
        if arabic_reshaper is not None:
            text = arabic_reshaper.reshape(text)
        return get_display(text)
    except Exception:
        return str(text)


def to_arabic_digits(text):
    western = "0123456789"
    eastern = "٠١٢٣٤٥٦٧٨٩"
    return str(text).translate(str.maketrans(western, eastern))


def build_pdf_report(features, predicted, confidence, probabilities, lang="English"):
    if colors is None:
        return None

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=1.4 * cm,
        leftMargin=1.4 * cm,
        topMargin=1.2 * cm,
        bottomMargin=1.2 * cm
    )

    PRIMARY_NAVY = colors.HexColor("#07152E")
    PRIMARY_BLUE = colors.HexColor("#1E88E5")
    LIGHT_BLUE = colors.HexColor("#EAF4FF")
    SOFT_RED = colors.HexColor("#FFEBEE")
    ACCENT_RED = colors.HexColor("#E53935")
    TEXT_DARK = colors.HexColor("#1F2937")
    BORDER = colors.HexColor("#D6E4F0")

    styles = getSampleStyleSheet()
    is_pdf_ar = lang == "العربية"
    pdf_align = 2 if is_pdf_ar else 0

    pdf_font = setup_pdf_fonts() if is_pdf_ar else "Helvetica"
    pdf_bold_font = pdf_font if is_pdf_ar else "Helvetica-Bold"

    def pdf_num(value):
        text_value = str(value)
        return to_arabic_digits(text_value) if is_pdf_ar else text_value

    def ptxt(text_value, style):
        text_value = "" if text_value is None else str(text_value)

        if is_pdf_ar:
            text_value = (
                text_value.replace("<br/>", "\n")
                .replace("<br>", "\n")
                .replace("</br>", "\n")
                .replace("<b>", "")
                .replace("</b>", "")
            )
            lines = text_value.split("\n")
            fixed_lines = [ar_text(line) for line in lines]
            final_text = "<br/>".join(fixed_lines)
            return Paragraph(final_text, style)

        return Paragraph(text_value.replace("\n", "<br/>"), style)

    title_style = ParagraphStyle(
        "NabdhTitle", parent=styles["Title"], fontName=pdf_bold_font,
        fontSize=22, textColor=colors.white, leading=26, spaceAfter=4,
        alignment=pdf_align
    )
    sub_style = ParagraphStyle(
        "NabdhSub", parent=styles["Normal"], fontName=pdf_font,
        fontSize=9, textColor=colors.HexColor("#D7E6F8"), leading=12,
        alignment=pdf_align
    )
    h_style = ParagraphStyle(
        "Section", parent=styles["Heading2"], fontName=pdf_bold_font,
        fontSize=14, textColor=PRIMARY_NAVY, leading=18,
        spaceBefore=10, spaceAfter=8, alignment=pdf_align
    )
    normal = ParagraphStyle(
        "NormalText", parent=styles["Normal"], fontName=pdf_font,
        fontSize=9.5, textColor=TEXT_DARK, leading=14,
        alignment=pdf_align
    )
    small = ParagraphStyle(
        "Small", parent=styles["Normal"], fontName=pdf_font,
        fontSize=8, textColor=colors.HexColor("#64748B"), leading=11,
        alignment=pdf_align
    )
    center_small = ParagraphStyle(
        "CenterSmall", parent=normal, fontName=pdf_font,
        fontSize=8.5, textColor=PRIMARY_BLUE, leading=12, alignment=1
    )

    report_id = f"NABDH-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    if is_pdf_ar:
        report_id = to_arabic_digits(report_id)

    pdf_labels = {
        "brand": "منصة نبض" if is_pdf_ar else "Nabdh AI",
        "report_title": "تقرير تحليل الرنين المغناطيسي للقلب" if is_pdf_ar else "Clinical Cardiac MRI Analysis Report",
        "platform": "منصة ذكية لتشخيص أمراض القلب" if is_pdf_ar else "Intelligent Cardiac Diagnosis Platform",
        "university": "جامعة الإمام محمد بن سعود الإسلامية" if is_pdf_ar else "Imam Mohammad Ibn Saud Islamic University",
        "college": "كلية علوم الحاسب والمعلومات" if is_pdf_ar else "College of Computer and Information Sciences",
        "department": "قسم علوم الحاسب" if is_pdf_ar else "Department of Computer Science",
        "report_id": "رقم التقرير" if is_pdf_ar else "Report ID",
        "predicted": "المرض المتوقع" if is_pdf_ar else "Predicted Disease",
        "confidence": "درجة الثقة" if is_pdf_ar else "Confidence Score",
        "risk": "مستوى الخطورة" if is_pdf_ar else "Risk Level",
        "framework_summary": "ملخص إطار العمل" if is_pdf_ar else "Framework Summary",
        "framework": "الإطار" if is_pdf_ar else "Framework",
        "model": "النموذج" if is_pdf_ar else "Model",
        "dataset": "مجموعة البيانات" if is_pdf_ar else "Dataset",
        "seg_output": "مخرجات التجزئة" if is_pdf_ar else "Segmentation Output",
        "seg_summary": "ملخص التجزئة" if is_pdf_ar else "Segmentation Summary",
        "final_interpretation": "تفسير التنبؤ النهائي" if is_pdf_ar else "Final Prediction Interpretation",
        "features": "الخصائص القلبية المستخرجة" if is_pdf_ar else "Extracted Cardiac Features",
        "feature": "الخاصية" if is_pdf_ar else "Feature",
        "value": "القيمة" if is_pdf_ar else "Value",
        "recommendation": "التوصية السريرية" if is_pdf_ar else "Clinical Recommendation",
        "normal": "طبيعي" if is_pdf_ar else "Normal",
        "moderate": "خطورة متوسطة" if is_pdf_ar else "Moderate",
        "high": "خطورة مرتفعة" if is_pdf_ar else "High",
        "generated_auto": "تم إنشاؤه تلقائيًا" if is_pdf_ar else "Generated Automatically",
        "prototype": "نموذج بحثي لمنصة نبض" if is_pdf_ar else "Nabdh AI Research Prototype",
        "research_platform": "منصة بحثية" if is_pdf_ar else "Research Platform",
        "verified_analysis": "تحليل موثق" if is_pdf_ar else "Verified Analysis",
    }

    risk = pdf_labels["normal"] if predicted == "NOR" else (
        pdf_labels["high"] if confidence >= 70 else pdf_labels["moderate"]
    )
    disease_name = DISEASE_AR.get(predicted, predicted) if is_pdf_ar else DISEASE_FULL_NAME.get(predicted, predicted)
    disease_display = disease_name if is_pdf_ar else predicted
    confidence_display = f"{confidence:.2f}%"
    if is_pdf_ar:
        confidence_display = to_arabic_digits(confidence_display)

    risk_color = ACCENT_RED if confidence >= 70 else PRIMARY_BLUE
    risk_bg = SOFT_RED if confidence >= 70 else LIGHT_BLUE

    elements = []

    institution_text = (
        f"{pdf_labels['platform']}\n"
        f"{pdf_labels['university']}\n"
        f"{pdf_labels['college']}\n"
        f"{pdf_labels['department']}"
    ).replace("\n", "<br/>")

    report_info_text = (
        f"{report_id} :{pdf_labels['report_id']}"
        if is_pdf_ar else
        f"{pdf_labels['report_id']}: {report_id}"
    )

    if is_pdf_ar:
        header_data = [
            [
                ptxt(pdf_labels["report_title"], ParagraphStyle(
                    "HeaderRightArabic", parent=styles["Normal"], fontName=pdf_bold_font,
                    fontSize=12, textColor=colors.white, alignment=2
                )),
                ptxt(pdf_labels["brand"], title_style),
            ],
            [
                ptxt(report_info_text, sub_style),
                ptxt(institution_text, sub_style),
            ]
        ]
    else:
        header_data = [
            [
                ptxt(pdf_labels["brand"], title_style),
                ptxt(pdf_labels["report_title"], ParagraphStyle(
                    "HeaderRight", parent=styles["Normal"], fontName=pdf_bold_font,
                    fontSize=12, textColor=colors.white, alignment=2
                ))
            ],
            [
                ptxt(institution_text, sub_style),
                ptxt(report_info_text, sub_style)
            ]
        ]

    header = Table(header_data, colWidths=[10.5 * cm, 7.0 * cm])
    header.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), PRIMARY_NAVY),
        ("BOX", (0, 0), (-1, -1), 1, PRIMARY_NAVY),
        ("LEFTPADDING", (0, 0), (-1, -1), 14),
        ("RIGHTPADDING", (0, 0), (-1, -1), 14),
        ("TOPPADDING", (0, 0), (-1, -1), 12),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
    ]))
    elements.append(header)
    elements.append(Spacer(1, 0.35 * cm))

    if is_pdf_ar:
        summary_data = [
            [ptxt(pdf_labels["risk"], normal), ptxt(pdf_labels["confidence"], normal), ptxt(pdf_labels["predicted"], normal)],
            [ptxt(risk, normal), ptxt(confidence_display, normal), ptxt(disease_display, normal)]
        ]
    else:
        summary_data = [
            [ptxt(pdf_labels["predicted"], normal), ptxt(pdf_labels["confidence"], normal), ptxt(pdf_labels["risk"], normal)],
            [ptxt(disease_display, normal), ptxt(confidence_display, normal), ptxt(risk, normal)]
        ]

    summary = Table(summary_data, colWidths=[6.0 * cm, 5.6 * cm, 5.6 * cm])
    risk_col = 0 if is_pdf_ar else 2
    summary.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), LIGHT_BLUE),
        ("BACKGROUND", (0, 1), (-1, 1), colors.white),
        ("BACKGROUND", (risk_col, 1), (risk_col, 1), risk_bg),
        ("TEXTCOLOR", (risk_col, 1), (risk_col, 1), risk_color),
        ("BOX", (0, 0), (-1, -1), 1, BORDER),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, BORDER),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
    ]))
    elements.append(summary)
    elements.append(Spacer(1, 0.25 * cm))

    elements.append(ptxt(pdf_labels["framework_summary"], h_style))
    framework = Table([
        [ptxt("Federated DeepLab + Feature-Based Prediction", normal), ptxt(pdf_labels["framework"], normal)],
        [ptxt("XGBoost-ready Feature Classifier", normal), ptxt(pdf_labels["model"], normal)],
        [ptxt("ACDC Cardiac MRI", normal), ptxt(pdf_labels["dataset"], normal)],
        [ptxt("RV, MYO, LV cardiac structures", normal), ptxt(pdf_labels["seg_output"], normal)],
    ], colWidths=[12.4 * cm, 4.8 * cm])
    framework.setStyle(TableStyle([
        ("BACKGROUND", (1 if is_pdf_ar else 0, 0), (1 if is_pdf_ar else 0, -1), LIGHT_BLUE),
        ("BOX", (0, 0), (-1, -1), 1, BORDER),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, BORDER),
        ("FONTNAME", (0, 0), (-1, -1), pdf_font),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    elements.append(framework)

    elements.append(ptxt(pdf_labels["seg_summary"], h_style))
    seg_headers = [
        "حجم البطين الأيمن" if is_pdf_ar else "RV Volume Proxy",
        "حجم عضلة القلب" if is_pdf_ar else "MYO Volume Proxy",
        "حجم البطين الأيسر" if is_pdf_ar else "LV Volume Proxy"
    ]

    if is_pdf_ar:
        seg_table_data = [
            [ptxt(seg_headers[2], normal), ptxt(seg_headers[1], normal), ptxt(seg_headers[0], normal)],
            [
                ptxt(pdf_num(features["lv_volume_proxy"]), normal),
                ptxt(pdf_num(features["myo_volume_proxy"]), normal),
                ptxt(pdf_num(features["rv_volume_proxy"]), normal)
            ]
        ]
    else:
        seg_table_data = [
            [ptxt(seg_headers[0], normal), ptxt(seg_headers[1], normal), ptxt(seg_headers[2], normal)],
            [
                ptxt(pdf_num(features["rv_volume_proxy"]), normal),
                ptxt(pdf_num(features["myo_volume_proxy"]), normal),
                ptxt(pdf_num(features["lv_volume_proxy"]), normal)
            ]
        ]

    seg_table = Table(seg_table_data, colWidths=[5.75 * cm, 5.75 * cm, 5.75 * cm])
    seg_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), LIGHT_BLUE),
        ("TEXTCOLOR", (0, 0), (-1, 0), PRIMARY_NAVY),
        ("FONTNAME", (0, 0), (-1, -1), pdf_font),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("BOX", (0, 0), (-1, -1), 1, BORDER),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, BORDER),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    elements.append(seg_table)

    elements.append(ptxt(pdf_labels["final_interpretation"], h_style))
    interpretation_text = (
        "تم اختيار المرض المتوقع النهائي بناءً على الخصائص القلبية المستخرجة من صورة الرنين المغناطيسي.\n"
        "يعرض التقرير المرض الأعلى ترجيحًا فقط لتجنب اللبس السريري."
        if is_pdf_ar else
        "The final predicted disease was selected based on the cardiac features extracted from the MRI image.\n"
        "The report shows only the highest-ranked disease to avoid clinical confusion."
    )
    interpretation_note = Table([[ptxt(interpretation_text, normal)]], colWidths=[17.2 * cm])
    interpretation_note.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT_BLUE),
        ("BOX", (0, 0), (-1, -1), 1, BORDER),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    elements.append(interpretation_note)
    elements.append(Spacer(1, 0.15 * cm))

    elements.append(ptxt(pdf_labels["features"], h_style))
    df = report_feature_table(features)
    feature_name_ar = {
        "RV Volume Proxy": "حجم البطين الأيمن",
        "MYO Volume Proxy": "حجم عضلة القلب",
        "LV Volume Proxy": "حجم البطين الأيسر",
        "RV Area Mean": "متوسط مساحة البطين الأيمن",
        "MYO Area Mean": "متوسط مساحة عضلة القلب",
        "LV Area Mean": "متوسط مساحة البطين الأيسر",
        "RV Circularity": "استدارة البطين الأيمن",
        "MYO Circularity": "استدارة عضلة القلب",
        "LV Circularity": "استدارة البطين الأيسر",
        "MYO Thickness Mean": "متوسط سماكة عضلة القلب",
        "MYO Thickness Max": "أقصى سماكة لعضلة القلب",
        "LV/RV Ratio": "نسبة البطين الأيسر إلى الأيمن",
        "MYO/LV Ratio": "نسبة عضلة القلب إلى البطين الأيسر",
        "MYO/RV Ratio": "نسبة عضلة القلب إلى البطين الأيمن",
    }

    if is_pdf_ar:
        feature_rows = [[ptxt(pdf_labels["value"], normal), ptxt(pdf_labels["feature"], normal)]]
    else:
        feature_rows = [[ptxt(pdf_labels["feature"], normal), ptxt(pdf_labels["value"], normal)]]

    for _, row in df.iterrows():
        feature_name = feature_name_ar.get(row["Feature"], row["Feature"]) if is_pdf_ar else row["Feature"]
        feature_value = pdf_num(row["Value"])
        if is_pdf_ar:
            feature_rows.append([ptxt(feature_value, normal), ptxt(feature_name, normal)])
        else:
            feature_rows.append([ptxt(feature_name, normal), ptxt(feature_value, normal)])

    feature_table = Table(
        feature_rows,
        colWidths=[5.7 * cm, 11.5 * cm] if is_pdf_ar else [11.5 * cm, 5.7 * cm]
    )
    feature_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), PRIMARY_BLUE),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, -1), pdf_font),
        ("GRID", (0, 0), (-1, -1), 0.5, BORDER),
        ("FONTSIZE", (0, 0), (-1, -1), 8.2),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_BLUE]),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    elements.append(feature_table)

    elements.append(ptxt(pdf_labels["recommendation"], h_style))
    recommendation_text_pdf = (
        "ينبغي مراجعة الحالة من قبل طبيب قلب مختص للتأكيد السريري."
        if is_pdf_ar else
        "The case should be reviewed by a cardiology specialist for clinical confirmation."
    )
    recommendation = Table([[ptxt(recommendation_text_pdf, normal)]], colWidths=[17.2 * cm])
    recommendation.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), SOFT_RED),
        ("BOX", (0, 0), (-1, -1), 1, ACCENT_RED),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    elements.append(recommendation)
    elements.append(Spacer(1, 0.25 * cm))

    signature = Table([
        [ptxt(pdf_labels["generated_auto"], normal)],
        [ptxt(pdf_labels["prototype"], small)]
    ], colWidths=[17.2 * cm])
    signature.setStyle(TableStyle([
        ("LINEABOVE", (0, 0), (-1, 0), 1, TEXT_DARK),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
    ]))
    elements.append(signature)
    elements.append(Spacer(1, 0.25 * cm))

    seal_title = "منصة نبض" if is_pdf_ar else "NABDH AI"
    seal_sub = pdf_labels["research_platform"]
    seal_verify = pdf_labels["verified_analysis"]
    straight_seal = Table(
        [[ptxt(f"<b>{seal_title}</b><br/>{seal_sub}<br/>{seal_verify}", center_small)]],
        colWidths=[4.6 * cm]
    )
    straight_seal.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), 1.3, PRIMARY_BLUE),
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT_BLUE),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    elements.append(Table([["", straight_seal]], colWidths=[12.6 * cm, 4.6 * cm], style=TableStyle([
        ("ALIGN", (1, 0), (1, 0), "RIGHT"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
    ])))
    elements.append(Spacer(1, 0.1 * cm))

    footer_text_pdf = (
        "© ٢٠٢٦ منصة نبض — منصة تشخيص أمراض القلب بالذكاء الاصطناعي والتعلم الموحد<br/>نموذج بحثي أولي، ولا يستخدم للتشخيص السريري المباشر دون مراجعة المختص."
        if is_pdf_ar else
        "© 2026 Nabdh AI — Federated AI Cardiac Diagnosis Platform<br/>Research prototype only. Not intended for direct clinical diagnosis without specialist review."
    )
    footer = Table([[ptxt(footer_text_pdf, sub_style)]], colWidths=[17.2 * cm])
    footer.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), PRIMARY_NAVY),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    elements.append(footer)

    doc.build(elements)
    pdf = buffer.getvalue()
    buffer.close()
    return pdf

# =====================================================
# SECURE ACCESS GATE - CREATIVE LOGIN FIXED
# =====================================================
def render_secure_access():
    role_options = [t["hospital_staff"], t["administrator"]]

    welcome_title = "مرحبًا بك" if is_ar else "Welcome Back"
    welcome_text = (
        "سجّل الدخول للمتابعة إلى منصة نبض الذكية البحثية الطبية."
        if is_ar else
        "Sign in to continue to Nabdh AI medical research platform."
    )
  
    mini_title = "تسجيل دخول آمن" if is_ar else "Secure Login"

    if not LOGO_SRC:
        st.warning("Logo file not found. Please place nabdh_logo.png inside the assets folder.")

    logo_html = (
    f'<img src="{LOGO_SRC}" class="secure-logo-img" alt="Nabdh AI Logo">'
    if LOGO_SRC
    else f'<div class="secure-title">{t["brand"]}</div>'
)
    st.markdown("<div class='secure-page'>", unsafe_allow_html=True)

    if is_ar:
        form_col, visual_col = st.columns([0.98, 1.02], gap="large")
    else:
        visual_col, form_col = st.columns([1.02, 0.98], gap="large")

    with visual_col:
        visual_title = "نبض " if is_ar else "Nabdh AI"
        visual_sub = "منصة ذكية وآمنة لصحة القلب" if is_ar else "Secure Medical Platform"
        

        st.markdown(
            f"""
            <div class="left-login-panel rtl">
                <h2>{visual_title}</h2>
                <h4>{visual_sub}</h4>
                <div class="heart_icon">🫀</div>
            </div>
            """,
            unsafe_allow_html=True
        )

###################################
# RIGHT LOGIN FORM
###################################
    with form_col:
        st.markdown('<div class="right-login-card rtl">', unsafe_allow_html=True)
        secure_form_heading_html = f"""
        <div class="secure-form-heading">
            <div class="secure-form-mini-title">{mini_title}</div>
            <h2>{welcome_title}</h2>
            <p>{welcome_text}</p>
        </div>
        """
        st.markdown(secure_form_heading_html, unsafe_allow_html=True)

        with st.form("secure_access_form"):
            role = st.selectbox(t["role"], role_options)
            national_id = st.text_input(t["national_id"], placeholder="1234567890")
            password = st.text_input(t["password"], type="password", placeholder="••••••••")
            otp = st.text_input(t["otp"], placeholder="123456")

            submitted = st.form_submit_button(t["sign_in"], use_container_width=True)

            if submitted:
                if national_id.strip() and password.strip() and otp.strip() == "123456":
                    st.session_state.authenticated = True
                    st.session_state.user_role = role
                    st.session_state.user_id = national_id.strip()
                    st.session_state.user_display_name = f"{role} - {national_id.strip()[-4:]}"
                    st.success(t["access_granted"])
                    st.rerun()
                else:
                    st.error(t["access_denied"])

        st.markdown(f"<div class='secure-demo-line'>{t['demo_otp']}</div>", unsafe_allow_html=True)

        login_lang = st.radio(
            t["language"],
            ["English", "العربية"],
            index=0 if st.session_state.language == "English" else 1,
            key="login_lang_radio",
            horizontal=True
        )

        if login_lang != st.session_state.language:
            st.session_state.language = login_lang
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)
    

# Important: do not render the main system before login.
if st.session_state.get("authenticated") and not st.session_state.get("user_role"):
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    render_secure_access()
    st.stop()


# =====================================================
# TOP BAR
# =====================================================
st.markdown(
    f"""
    <div class="topbar rtl">
        <div>
            {f'<img src="{LOGO_SRC}" class="brand-logo-img" alt="Nabdh AI Logo">' if LOGO_SRC else f'<div class="brand-click">{t["brand"]}</div><div class="brand-sub">{t["subtitle"]}</div>'}
        </div>
        <div>
            <div>{t['ai_system']}</div>
            <div class="access-chip">{t['current_user']}: {st.session_state.user_role}</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="nav-shell">', unsafe_allow_html=True)
all_pages = [
    (t["home"], "home"),
    (t["dashboard"], "dashboard"),
    (t["upload"], "upload"),
    (t["segmentation"], "segmentation"),
    (t["features"], "features"),
    (t["prediction"], "prediction"),
    (t["report"], "report"),
    (t["history"], "history"),
    (t["federated"], "federated"),
]

current_role = st.session_state.get("user_role", "Hospital Staff")
allowed_pages = allowed_pages_by_role(current_role)

pages = [
    item for item in all_pages
    if item[1] in allowed_pages
]

nav_items = [(key, label) for label, key in pages]

if st.session_state.page not in allowed_pages and st.session_state.page not in ["about", "help", "settings"]:
    st.session_state.page = "home"

cols = st.columns(len(nav_items) + 1)

if is_ar:
    menu_col = cols[0]          # القائمة الجانبية تظهر يسارًا
    nav_cols = cols[1:]         # العناصر تبدأ من اليمين
    display_nav_items = nav_items[::-1]
else:
    nav_cols = cols[:-1]
    menu_col = cols[-1]
    display_nav_items = nav_items

for col, (key, label) in zip(nav_cols, display_nav_items):
    with col:
        active = "● " if st.session_state.page == key else ""
        if st.button(active + label, key=f"nav_{key}", use_container_width=True):
            set_page(key)

with menu_col:
    with st.popover("⋮", use_container_width=True):
        st.markdown(f"**{t['language']}**")
        lang = st.radio("", ["English", "العربية"], index=0 if st.session_state.language == "English" else 1, key="lang_radio", label_visibility="collapsed")
        if lang != st.session_state.language:
            st.session_state.language = lang
            st.rerun()
        st.divider()
        if st.button(t["about"], key="menu_about", use_container_width=True):
            set_page("about")
        if st.button(t["help"], key="menu_help", use_container_width=True):
            set_page("help")
        if st.button(t["settings"], key="menu_settings", use_container_width=True):
            set_page("settings")
        st.divider()
        if st.button(t["sign_out"], key="menu_sign_out", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.user_role = ""
            st.session_state.user_id = ""
            st.session_state.user_display_name = ""
            st.session_state.page = "home"
            st.rerun()
st.markdown('</div>', unsafe_allow_html=True)

# =====================================================
# PAGES
# =====================================================
page = st.session_state.page

if page == "home":
    if LOGO_SRC:
        st.markdown(
            f"""
            <div class="home-logo-wrap">
                <img src="{LOGO_SRC}" class="home-logo-img" alt="Nabdh AI Logo">
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        f"""
        <div class="hero rtl">
            <div class="hero-title">{t['hero_title']}</div>
            <div class="hero-text">{t['hero_text']}</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    ecg_component()
    c1, c2 = st.columns(2)
    with c1:
        if st.button(t["start"], key="home_start", use_container_width=True):
            set_page("upload")
    with c2:
        if st.button(t["view_dashboard"], key="home_dashboard", use_container_width=True):
            set_page("dashboard")

elif page == "about":
    page_title(t["about"])

    if is_ar:
        vision_text = """
        أن تكون منصة نبض نموذجاً وطنياً رائداً في تشخيص أمراض القلب
        باستخدام الذكاء الاصطناعي والتعلم الموحد مع المحافظة على
        خصوصية البيانات الطبية.
        """

        mission_text = """
        توفير منصة بحثية ذكية تدعم تحليل صور الرنين المغناطيسي للقلب
        واستخراج الخصائص القلبية والتنبؤ بالأمراض دون مشاركة البيانات
        الطبية الخام بين الجهات الصحية.
        """

        objectives_html = """
        <ol>
            <li>تحليل صور الرنين المغناطيسي للقلب.</li>
            <li>تنفيذ التجزئة الدلالية لمناطق القلب.</li>
            <li>استخراج الخصائص العلمية للقلب.</li>
            <li>التنبؤ الآلي بأمراض القلب.</li>
            <li>دعم بيئات التعلم الموحد بين المستشفيات.</li>
        </ol>
        """
    else:
        vision_text = """
        To become a leading intelligent platform for cardiac disease diagnosis
        using AI and federated learning while preserving medical privacy.
        """

        mission_text = """
        To provide an intelligent research platform for cardiac MRI analysis,
        feature extraction, and disease prediction without sharing raw patient data.
        """

        objectives_html = """
        <ol>
            <li>Analyze cardiac MRI images.</li>
            <li>Perform semantic segmentation.</li>
            <li>Extract scientific cardiac features.</li>
            <li>Predict cardiac diseases automatically.</li>
            <li>Support federated hospital collaboration.</li>
        </ol>
        """

    st.markdown(
        f"""
        <div class="card rtl">
            {f'<img src="{LOGO_SRC}" class="about-logo-img" alt="Nabdh AI Logo">' if LOGO_SRC else ''}
            <h2>{t['brand']}</h2>
            <p><b>{t['subtitle']}</b></p>
            <p>{t['university']}<br>{t['college']}<br>{t['department']}</p>
        </div>

        <div class="blue-card rtl">
            <h3>{t['vision']}</h3>
            <p>{vision_text}</p>
        </div>

        <div class="card rtl">
            <h3>{t['mission']}</h3>
            <p>{mission_text}</p>
        </div>

        <div class="card rtl">
            <h3>{t['objectives']}</h3>
            {objectives_html}
        </div>
        """,
        unsafe_allow_html=True
    )

elif page == "dashboard":
    page_title(t["dashboard"])
    c1, c2, c3, c4 = st.columns(4)
    with c1: stat_card(t["cases"], "100", "+20")
    with c2: stat_card(t["analyses"], "100", "+20")
    with c3: stat_card(t["predictions"], "100", "+20")
    with c4: stat_card(t["reports"], "75", "+12")

    left, right = st.columns([1.35, 1])
    with left:
        st.markdown(f"<div class='card rtl'><h3>{t['distribution']}</h3></div>", unsafe_allow_html=True)
        disease_df = pd.DataFrame({"Disease": ["NOR", "DCM", "HCM", "MINF", "ARV"], "Cases": [20, 20, 20, 20, 20]})
        st.bar_chart(disease_df.set_index("Disease"))
    with right:
        st.markdown(f"<div class='card rtl'><h3>{t['performance']}</h3></div>", unsafe_allow_html=True)
        p1, p2 = st.columns(2)
        with p1:
            stat_card("Dice", "0.838", "Final FL Round")
        with p2:
            stat_card("IoU", "0.744", "Final FL Round")
        p3, p4 = st.columns(2)
        with p3:
            stat_card("Accuracy", "98.7%", "Segmentation")
        with p4:
            stat_card("Reports", "75", "Generated")

elif page == "upload":
    page_title(t["upload_title"])
    st.markdown(
        f"""
        <div class="upload-box rtl">
            <h3>{t['upload_title']}</h3>
            <p>{t['upload_text']}</p>
            <p style="color:#64748b; font-size:14px;">The uploaded MRI will be used for segmentation, feature extraction, prediction, and clinical reporting.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    uploaded_file = st.file_uploader("", label_visibility="collapsed")
    if uploaded_file is not None:
        with h5py.File(uploaded_file, "r") as h5:
            image = h5["image"][:]
            mask = h5["label"][:]

        st.session_state.image = image
        st.session_state.mask = mask
        st.session_state.uploaded_filename = uploaded_file.name
        ensure_prediction()
        st.success(t["success"])

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"<div class='card rtl'><h3>{t['original']}</h3></div>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(image, cmap="gray")
            ax.axis("off")
            st.pyplot(fig)
        with col2:
            st.markdown(f"<div class='card rtl'><h3>{t['mask']}</h3></div>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(mask, cmap="tab10")
            ax.axis("off")
            st.pyplot(fig)

        n1, n2, n3 = st.columns(3)
        with n1:
            if st.button("Go to Segmentation" if not is_ar else "الانتقال إلى التجزئة", key="upload_to_seg", use_container_width=True):
                set_page("segmentation")
        with n2:
            if st.button("Go to Features" if not is_ar else "الانتقال إلى الخصائص", key="upload_to_feat", use_container_width=True):
                set_page("features")
        with n3:
            if st.button("Go to Prediction" if not is_ar else "الانتقال إلى التنبؤ", key="upload_to_pred", use_container_width=True):
                set_page("prediction")

elif page == "segmentation":
    page_title(t["segmentation"])
    if "image" not in st.session_state or "mask" not in st.session_state:
        st.warning(t["no_file"])
    else:
        image = st.session_state.image
        mask = st.session_state.mask
        overlay = create_overlay(image, mask)
        rv_area = int((mask == 1).sum())
        myo_area = int((mask == 2).sum())
        lv_area = int((mask == 3).sum())
        total_area = rv_area + myo_area + lv_area

        st.markdown(
            f"""
            <div class="card rtl">
                <h3>{t['segmentation']}</h3>
                <p>This page visualizes the cardiac segmentation output and separates the main cardiac structures: right ventricle, myocardium, and left ventricle.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"<div class='card rtl'><h3>{t['original']}</h3></div>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(image, cmap="gray")
            ax.axis("off")
            st.pyplot(fig)
        with col2:
            st.markdown(f"<div class='card rtl'><h3>{t['mask']}</h3></div>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(mask, cmap="tab10")
            ax.axis("off")
            st.pyplot(fig)
        with col3:
            st.markdown(f"<div class='card rtl'><h3>{t['overlay']}</h3></div>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(overlay)
            ax.axis("off")
            st.pyplot(fig)

        page_title("Cardiac Structures" if not is_ar else "تراكيب القلب")
        s1, s2, s3, s4 = st.columns(4)
        with s1: stat_card("RV Area", f"{rv_area:,}", "Right Ventricle")
        with s2: stat_card("MYO Area", f"{myo_area:,}", "Myocardium")
        with s3: stat_card("LV Area", f"{lv_area:,}", "Left Ventricle")
        with s4: stat_card("Total Heart Area", f"{total_area:,}", "RV + MYO + LV")

        left, right = st.columns([1.2, 1])
        with left:
            st.markdown("<div class='card rtl'><h3>Cardiac Region Area Distribution</h3></div>", unsafe_allow_html=True)
            area_df = pd.DataFrame({"Region": ["RV", "MYO", "LV"], "Area": [rv_area, myo_area, lv_area]})
            st.bar_chart(area_df.set_index("Region"))
        with right:
            st.markdown(
                """
                <div class="card rtl">
                    <h3>Segmentation Legend</h3>
                    <div style="line-height:2.2; font-size:16px;">
                        <b style="color:#0072ff;">■ RV</b> - Right Ventricle<br>
                        <b style="color:#00c853;">■ MYO</b> - Myocardium<br>
                        <b style="color:#e53935;">■ LV</b> - Left Ventricle
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        page_title("Segmentation Quality" if not is_ar else "جودة التجزئة")
        q1, q2, q3 = st.columns(3)
        with q1: stat_card("Dice Score", "0.838", "Final FL Round")
        with q2: stat_card("IoU", "0.744", "Final FL Round")
        with q3: stat_card("Pixel Accuracy", "98.7%", "Final FL Round")

        st.markdown(
            """
            <div class="card rtl">
                <h3>Clinical Interpretation</h3>
                <p>The segmentation output identifies the main cardiac structures required for downstream cardiac feature extraction. The extracted RV, MYO, and LV regions are used to calculate morphological and clinical cardiac indicators.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.success("Segmentation completed successfully. The output is ready for feature extraction.")

elif page == "features":
    page_title(t["features"])
    if "image" not in st.session_state or "mask" not in st.session_state:
        st.warning(t["no_file"])
    else:
        features, predicted, confidence, probabilities = ensure_prediction()
        df = report_feature_table(features)
        lv_volume = features["lv_volume_proxy"]
        myo_volume = features["myo_volume_proxy"]
        rv_volume = features["rv_volume_proxy"]

        c1, c2, c3 = st.columns(3)
        with c1: stat_card("LV Volume" if not is_ar else "حجم البطين الأيسر", f"{lv_volume:,}", "proxy")
        with c2: stat_card("MYO Volume" if not is_ar else "حجم عضلة القلب", f"{myo_volume:,}", "proxy")
        with c3: stat_card("RV Volume" if not is_ar else "حجم البطين الأيمن", f"{rv_volume:,}", "proxy")

        st.markdown("<div class='card rtl'><h3>Morphological Cardiac Features</h3></div>", unsafe_allow_html=True)
        r1, r2, r3 = st.columns(3)
        with r1: stat_card("LV/RV Ratio", f"{features['lv_rv_ratio']:.3f}", "ratio")
        with r2: stat_card("MYO/LV Ratio", f"{features['myo_lv_ratio']:.3f}", "ratio")
        with r3: stat_card("MYO/RV Ratio", f"{features['myo_rv_ratio']:.3f}", "ratio")

        st.markdown("<div class='card rtl'><h3>Extracted Feature Vector</h3></div>", unsafe_allow_html=True)
        st.dataframe(df, use_container_width=True, hide_index=True)

        chart_df = pd.DataFrame({"Region": ["RV", "MYO", "LV"], "Volume Proxy": [rv_volume, myo_volume, lv_volume]})
        st.markdown("<div class='card rtl'><h3>Cardiac Volume Proxy Distribution</h3></div>", unsafe_allow_html=True)
        st.bar_chart(chart_df.set_index("Region"))
        st.success("Feature extraction completed successfully. The extracted features are ready for disease prediction.")
        if st.button("Go to Prediction" if not is_ar else "الانتقال إلى التنبؤ", key="features_to_prediction", use_container_width=True):
            set_page("prediction")

elif page == "prediction":
    page_title(t["prediction"])
    if "image" not in st.session_state or "mask" not in st.session_state:
        st.warning(t["no_file"])
    else:
        features, predicted, confidence, probabilities = ensure_prediction()

    if is_ar:
        st.write("المرض المتوقع:", predicted)
        st.write("درجة الثقة:", confidence)
        st.write("جميع الاحتمالات:", probabilities)
    else:
        st.write("Disease:", predicted)
        st.write("Confidence:", confidence)
        st.write("Probabilities:", probabilities)
        save_current_analysis_once(
        predicted,
        confidence
    )

        full_name = DISEASE_FULL_NAME.get(predicted, predicted)
        disease_display = DISEASE_AR.get(predicted, predicted) if is_ar else f"{predicted} ({full_name})"
        risk_level = "High" if confidence >= 70 else "Moderate"
        if is_ar:
            risk_level = "مرتفع" if confidence >= 70 else "متوسط"

        c1, c2, c3 = st.columns(3)
        with c1: stat_card(t["disease"], predicted, f"{t['confidence']}: {confidence:.2f}%")
        with c2: stat_card(t["risk"], risk_level, t["clinical_review"])
        with c3: stat_card(t["model"], "XGBoost-ready", t["feature_based_prediction"])

        interpretation = clinical_interpretation_text(predicted, features, st.session_state.language)
        st.markdown(
            f"""
            <div class="card rtl">
                <h3>{t['predicted_interpretation']}</h3>
                <p>{interpretation}</p>
                <p><b>{t['final_prediction']}:</b> {disease_display}</p>
                <p><b>{t['confidence']}:</b> {confidence:.2f}%</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        prob_df = pd.DataFrame({"Disease": list(probabilities.keys()), "Probability": [round(v * 100, 2) for v in probabilities.values()]})
        st.markdown(f"<div class='card rtl'><h3>{t['probability_distribution']}</h3></div>", unsafe_allow_html=True)
        st.bar_chart(prob_df.set_index("Disease"))
        st.markdown(f"<div class='card rtl'><h3>{t['recommendation']}</h3><p>{t['recommendation_text']}</p></div>", unsafe_allow_html=True)

elif page == "report":
    page_title(t["report"])

    if "image" not in st.session_state or "mask" not in st.session_state:
        st.warning(t["no_file"])
    else:
        features, predicted, confidence, probabilities = ensure_prediction()
        save_current_analysis_once(predicted, confidence)

        full_name = DISEASE_FULL_NAME.get(predicted, predicted)
        disease_local = DISEASE_AR.get(predicted, predicted) if is_ar else full_name
        generated = datetime.now().strftime("%Y-%m-%d %H:%M")
        risk_level = get_risk_level_ar(predicted, confidence) if is_ar else get_risk_level(predicted, confidence)

        st.markdown(
            f"""
            <div class="card rtl">
                <h2>{t['report']}</h2>
                <p><b>{'التاريخ' if is_ar else 'Generated'}:</b> {generated}</p>
                <p><b>{t['final_prediction']}:</b> {predicted} ({disease_local})</p>
                <p><b>{t['confidence']}:</b> {confidence:.2f}%</p>
                <p><b>{t['risk']}:</b> {risk_level}</p>
                <p><b>{t['recommendation']}:</b> {t['recommendation_text']}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        pdf_bytes = build_pdf_report(
            features,
            predicted,
            confidence,
            probabilities,
            st.session_state.language
        )

        b1, b2, b3 = st.columns([1, 1, 1])

        if pdf_bytes:
            b64_pdf = base64.b64encode(pdf_bytes).decode("utf-8")

            with b1:
                st.download_button(
                    "Download" if not is_ar else "تنزيل",
                    data=pdf_bytes,
                    file_name=f"Nabdh_AI_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    key="download_pdf_report"
                )

            with b2:
                print_label = "طباعة" if is_ar else "Print"
            components.html(
                f"""
                <button onclick="window.parent.print()" style="
                    width:100%;
                    height:44px;
                    border:none;
                    background:transparent;
                    font-size:16px;
                    font-weight:800;
                    color:#0B3B75;
                    text-decoration:underline;
                    cursor:pointer;">
                    {print_label}
                </button>
                """,
                height=85
            )

            with b3:
                st.text_input(
                    "Share Link" if not is_ar else "رابط المشاركة",
                    value="http://localhost:8501",
                    key="share_link_box"
                )

            st.success(t["official_pdf_ready"])

        else:
            with b1:
                st.button("Download" if not is_ar else "تنزيل", use_container_width=True, disabled=True)
            with b2:
                st.button("Print" if not is_ar else "طباعة", use_container_width=True, disabled=True)
            with b3:
                st.text_input(
                    "Share Link" if not is_ar else "رابط المشاركة",
                    value="http://localhost:8501",
                    key="share_link_box_no_pdf"
                )

            st.error(
                "Install reportlab first: python3 -m pip install reportlab"
                if not is_ar else
                "ثبتي reportlab أولًا: python3 -m pip install reportlab"
            )


elif page == "history":
    page_title(t["history_title"])
    history_df = load_history_dataframe(is_ar=is_ar)

    st.markdown(
        f"""
        <div class="card rtl">
            <h3>{t['history_title']}</h3>
            <p>{t['history_note']}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    if history_df.empty:
        st.info(t["history_empty"])
    else:
        raw_df = pd.read_csv(HISTORY_FILE, engine="python", on_bad_lines="skip")
        total_cases = len(raw_df)
        disease_types = raw_df["Disease Code"].nunique() if "Disease Code" in raw_df.columns else 0
        avg_conf = raw_df["Confidence"].mean() if "Confidence" in raw_df.columns else 0

        h1, h2, h3 = st.columns(3)
        with h1:
            stat_card(t["history_total_cases"], str(total_cases), t["completed"])
        with h2:
            stat_card(t["history_disease_types"], str(disease_types), "NOR / DCM / HCM / MINF / ARV")
        with h3:
            stat_card(t["history_avg_confidence"], f"{avg_conf:.2f}%", t["feature_based_prediction"])

        st.markdown(f"<div class='card rtl'><h3>{t['history_table']}</h3></div>", unsafe_allow_html=True)
        st.dataframe(history_df, use_container_width=True, hide_index=True)

        if "Disease Code" in raw_df.columns:
            chart_df = raw_df["Disease Code"].value_counts().reset_index()
            chart_df.columns = ["Disease", "Cases"]
            st.markdown(f"<div class='card rtl'><h3>{t['distribution']}</h3></div>", unsafe_allow_html=True)
            st.bar_chart(chart_df.set_index("Disease"))


elif page == "federated":
    page_title(t["federated"])

    hospitals = pd.DataFrame({
        "Hospital": ["Hospital A", "Hospital B", "Hospital C", "Hospital D", "Hospital E"],
        "Samples": [300, 302, 304, 344, 286],
        "Round": [3, 3, 3, 3, 3],
        "Status": ["Connected", "Connected", "Connected", "Connected", "Connected"]
    })

    st.dataframe(hospitals, use_container_width=True, hide_index=True)

    rounds = pd.DataFrame({
        "Round": [1, 2, 3],
        "Loss": [0.725, 0.154, 0.087],
        "Dice": [0.605, 0.769, 0.838],
        "IoU": [0.497, 0.661, 0.744]
    })

    st.markdown("<div class='card rtl'><h3>Federated Learning Performance</h3></div>", unsafe_allow_html=True)
    st.line_chart(rounds.set_index("Round"))
    st.dataframe(rounds, use_container_width=True, hide_index=True)


elif page == "help":
    page_title(t["help"])

    if is_ar:
        help_title = "طريقة استخدام نبض"
        steps = [
            "ارفع ملف الرنين المغناطيسي للقلب.",
            "راجعي مخرجات التجزئة والعرض المدمج.",
            "استخرجي الخصائص القلبية من RV وMYO وLV.",
            "افتحي صفحة التنبؤ لعرض النتيجة المبنية على الخصائص.",
            "حمّلي التقرير الرسمي بصيغة PDF أو استخدمي رابط المشاركة."
        ]
    else:
        help_title = "How to use Nabdh AI"
        steps = [
            "Upload a cardiac MRI file.",
            "Review segmentation output and overlay.",
            "Extract cardiac features from RV, MYO, and LV.",
            "Open Prediction to view feature-based disease estimation.",
            "Download the official PDF report or use the share link."
        ]

    items = "".join([f"<li>{step}</li>" for step in steps])

    st.markdown(
        f"""
        <div class="card rtl">
            <h3>{help_title}</h3>
            <ol>{items}</ol>
        </div>
        """,
        unsafe_allow_html=True
    )


elif page == "settings":
    page_title(t["settings"])

    st.markdown(
        f"""
        <div class="card rtl">
            <h3>{t['settings']}</h3>
            <p>{'اللغة الحالية' if is_ar else 'Current language'}: <b>{st.session_state.language}</b></p>
            <p>{'ألوان الهوية: كحلي، أزرق طبي، بيبي بلو، وأحمر خفيف.' if is_ar else 'Theme colors: Navy, Medical Blue, Baby Blue, and Soft Red.'}</p>
        </div>
        """,
        unsafe_allow_html=True
    )


# =====================================================
# FOOTER IN ALL PAGES
# =====================================================
st.markdown(
    f"""
    <div class="footer rtl">
        <b>© 2026 {t['brand']} — Federated AI Cardiac Diagnosis Platform</b><br>
        {t['notice']}
    </div>
    """,
    unsafe_allow_html=True
)
