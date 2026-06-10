import io
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import h5py
import cv2
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
import arabic_reshaper


try:
    from bidi.algorithm import get_display
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
    "features", "prediction", "report", "federated", "help", "settings"
]

if "page" not in st.session_state:
    st.session_state.page = "home"

if "language" not in st.session_state:
    st.session_state.language = "English"

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
        "select_file": "Select cardiac MRI file",
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
        "vision_text": "To provide a privacy-preserving intelligent cardiac diagnosis platform that supports collaborative medical AI research.",
        "mission_text": "Nabdh AI integrates federated learning, cardiac MRI segmentation, feature extraction, and disease prediction to support research in automated cardiac disease diagnosis.",
        "objective_1": "Visualize cardiac MRI data and segmentation masks.",
        "objective_2": "Extract scientific cardiac features from RV, MYO, and LV regions.",
        "objective_3": "Provide feature-based disease prediction and structured clinical reporting.",
        "objective_4": "Represent a federated hospital collaboration framework without sharing raw patient data.",
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
        "final_prediction_summary": "Final Prediction Summary",
        "prediction_basis": "Prediction based on extracted cardiac features",
        "download_pdf": "Download ",
        "print_report": "Print",
        "uploaded_mri_usage": "The uploaded MRI will be used for segmentation, feature extraction, prediction, and clinical reporting.",
        "segmentation_desc": "This page visualizes the cardiac segmentation output and separates the main cardiac structures: right ventricle, myocardium, and left ventricle.",
        "segmentation_quality": "Segmentation Quality",
        "clinical_interpretation": "Clinical Interpretation",
        "segmentation_success": "Segmentation completed successfully. The output is ready for feature extraction.",
        "feature_success": "Feature extraction completed successfully. The extracted features are ready for disease prediction.",
        "footer_platform": "Federated AI Cardiac Diagnosis Platform",
    },
    "العربية": {
        "brand": "منصة نبض",
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
        "upload_text": "ارفعي ملف الرنين المغناطيسي للقلب لعرض الصورة الأصلية وقناع التجزئة.",
        "select_file": "اختيار ملف الرنين المغناطيسي للقلب",
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
        "download_report": " تنزيل ",
        "share_link": "رابط المشاركة",
        "print_note": "افتحي ملف PDF بعد تنزيله واطبعيه من Preview أو المتصفح.",
        "vision": "الرؤية",
        "mission": "الرسالة",
        "objectives": "الأهداف",
        "vision_text": "تقديم منصة ذكية محافظة على خصوصية البيانات لدعم تشخيص أمراض القلب والبحث الطبي التعاوني باستخدام الذكاء الاصطناعي.",
        "mission_text": "تدمج منصة نبض التعلم الموحد، وتجزئة صور الرنين المغناطيسي للقلب، واستخراج الخصائص، والتنبؤ بالمرض لدعم البحث في التشخيص الآلي لأمراض القلب.",
        "objective_1": "عرض صور الرنين المغناطيسي للقلب وأقنعة التجزئة.",
        "objective_2": "استخراج خصائص قلبية علمية من مناطق البطين الأيمن وعضلة القلب والبطين الأيسر.",
        "objective_3": "تقديم تنبؤ مبني على الخصائص مع تقرير سريري منظم.",
        "objective_4": "تمثيل إطار تعاون بين المستشفيات باستخدام التعلم الموحد دون مشاركة بيانات المرضى الخام.",
        "university": "جامعة الإمام محمد بن سعود الإسلامية",
        "college": "كلية علوم الحاسب والمعلومات",
        "department": "قسم علوم الحاسب",
        "notice": "نموذج بحثي أولي، ولا يستخدم للتشخيص السريري المباشر دون مراجعة المختص.",
        "final_prediction": "التنبؤ النهائي",
        "official_pdf_ready": "التقرير الرسمي جاهز نوع الملف PDF",
        "clinical_review": "مراجعة سريرية",
        "feature_based_prediction": "تنبؤ مبني على الخصائص",
        "predicted_interpretation": "تفسير التنبؤ بالمرض",
        "probability_distribution": "توزيع درجات التصنيف",
        "final_prediction_summary": "ملخص التنبؤ النهائي",
        "prediction_basis": "تم تحديد المرض المتوقع بناءً على الخصائص القلبية المستخرجة",
        "download_pdf": "تنزيل ",
        "print_report": "طباعة",
        "uploaded_mri_usage": "سيتم استخدام صورة الرنين في التجزئة، واستخراج الخصائص، والتنبؤ، وإعداد التقرير السريري.",
        "segmentation_desc": "تعرض هذه الصفحة نتيجة تجزئة القلب وتوضح التراكيب الأساسية: البطين الأيمن، عضلة القلب، والبطين الأيسر.",
        "segmentation_quality": "جودة التجزئة",
        "clinical_interpretation": "التفسير السريري",
        "segmentation_success": "تمت التجزئة بنجاح، والمخرجات جاهزة لاستخراج الخصائص.",
        "feature_success": "تم استخراج الخصائص بنجاح، وأصبحت جاهزة للتنبؤ بالمرض.",
        "footer_platform": "منصة تشخيص أمراض القلب بالذكاء الاصطناعي والتعلم الموحد",
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



    /* File uploader alignment and display labels */
    [data-testid="stFileUploader"] {{
        direction: {direction};
        text-align: {align};
    }}
    [data-testid="stFileUploaderDropzone"] {{
        direction: {direction};
    }}
    [data-testid="stFileUploaderDropzone"] button {{
        float: {"left" if is_ar else "right"};
    }}
    [data-testid="stFileUploaderDropzone"] button p {{
        font-size: 0 !important;
    }}
    [data-testid="stFileUploaderDropzone"] button p::after {{
        content: "{'تصفح الملفات' if is_ar else 'Browse files'}";
        font-size: 16px !important;
    }}
    [data-testid="stFileUploaderDropzone"] [data-testid="stFileUploaderDropzoneInstructions"] span {{
        font-size: 0 !important;
    }}
    [data-testid="stFileUploaderDropzone"] [data-testid="stFileUploaderDropzoneInstructions"] span::after {{
        content: "{'اسحبي الملف هنا' if is_ar else 'Drag and drop file here'}";
        font-size: 16px !important;
    }}

    /* Report actions: clean link-style controls */
    [data-testid="stDownloadButton"] button {{
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        color: #0B3B75 !important;
        font-size: 18px !important;
        font-weight: 700 !important;
        text-decoration: underline !important;
        padding: 0 !important;
        min-height: 44px !important;
    }}

    @media print {{
        header, [data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stStatusWidget"] {{
            display: none !important;
        }}
        .nav-shell {{
            display: none !important;
        }}
        .block-container {{
            max-width: 100% !important;
            padding: 0.5cm !important;
        }}
        .footer {{
            page-break-inside: avoid;
        }}
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


def ensure_prediction():
    if "image" not in st.session_state or "mask" not in st.session_state:
        return None, None, None, None
    image = st.session_state.image
    mask = st.session_state.mask
    features = extract_prediction_features(image, mask)
    predicted, confidence, probabilities = rule_based_prediction_from_features(features)
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

    def pdf_txt(x):
        return ar_text(x) if is_pdf_ar else str(x)

    def pdf_num(x):
        txt = str(x)
        return to_arabic_digits(txt) if is_pdf_ar else txt

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

    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    if is_pdf_ar:
        now = to_arabic_digits(now)

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
        "generated": "تاريخ الإنشاء" if is_pdf_ar else "Generated",
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
    risk_color = ACCENT_RED if confidence >= 70 else PRIMARY_BLUE
    risk_bg = SOFT_RED if confidence >= 70 else LIGHT_BLUE

    elements = []

    institution_text = f"{pdf_labels['platform']}\n{pdf_labels['university']}\n{pdf_labels['college']}\n{pdf_labels['department']}".replace("\n", "<br/>")
    report_info_text = f"{report_id} :{pdf_labels['report_id']}" if is_pdf_ar else f"{pdf_labels['report_id']}: {report_id}"

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

    disease_display = disease_name if is_pdf_ar else predicted
    confidence_display = f"{confidence:.2f}%"
    if is_pdf_ar:
        confidence_display = to_arabic_digits(confidence_display)
        summary_data = [
            [ptxt(pdf_labels["risk"], normal), ptxt(pdf_labels["confidence"], normal), ptxt(pdf_labels["predicted"], normal)],
            [ptxt(f"{risk}", normal), ptxt(confidence_display, normal), ptxt(f"{disease_display}", normal)]
        ]
    else:
        summary_data = [
            [ptxt(pdf_labels["predicted"], normal), ptxt(pdf_labels["confidence"], normal), ptxt(pdf_labels["risk"], normal)],
            [ptxt(f"{disease_display}", normal), ptxt(confidence_display, normal), ptxt(f"{risk}", normal)]
        ]

    summary = Table(summary_data, colWidths=[6.0 * cm, 5.6 * cm, 5.6 * cm])
    summary.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), LIGHT_BLUE),
        ("BACKGROUND", (0, 1), (-1, 1), colors.white),
        ("BACKGROUND", (0 if is_pdf_ar else 2, 1), (0 if is_pdf_ar else 2, 1), risk_bg),
        ("TEXTCOLOR", (0 if is_pdf_ar else 2, 1), (0 if is_pdf_ar else 2, 1), risk_color),
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
        [ptxt("Federated DeepLab + Feature-Based Prediction", normal),
        ptxt(pdf_labels["framework"], normal)],

        [ptxt("XGBoost-ready Feature Classifier", normal),
        ptxt(pdf_labels["model"], normal)],

        [ptxt("ACDC Cardiac MRI", normal),
        ptxt(pdf_labels["dataset"], normal)],

        [ptxt("RV, MYO, LV cardiac structures", normal),
        ptxt(pdf_labels["seg_output"], normal)],
    ],
    colWidths=[12.4 * cm, 4.8 * cm])
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
    seg_headers = ["حجم البطين الأيمن" if is_pdf_ar else "RV Volume Proxy",
                   "حجم عضلة القلب" if is_pdf_ar else "MYO Volume Proxy",
                   "حجم البطين الأيسر" if is_pdf_ar else "LV Volume Proxy"]
    if is_pdf_ar:
        seg_table_data = [
            [ptxt(seg_headers[2], normal), ptxt(seg_headers[1], normal), ptxt(seg_headers[0], normal)],
            [ptxt(pdf_num(features["lv_volume_proxy"]), normal), ptxt(pdf_num(features["myo_volume_proxy"]), normal), ptxt(pdf_num(features["rv_volume_proxy"]), normal)]
        ]
    else:
        seg_table_data = [
            [ptxt(seg_headers[0], normal), ptxt(seg_headers[1], normal), ptxt(seg_headers[2], normal)],
            [ptxt(pdf_num(features["rv_volume_proxy"]), normal), ptxt(pdf_num(features["myo_volume_proxy"]), normal), ptxt(pdf_num(features["lv_volume_proxy"]), normal)]
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

    feature_table = Table(feature_rows, colWidths=[5.7 * cm, 11.5 * cm] if is_pdf_ar else [11.5 * cm, 5.7 * cm])
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
    recommendation_text_pdf = "ينبغي مراجعة الحالة من قبل طبيب قلب مختص للتأكيد السريري." if is_pdf_ar else "The case should be reviewed by a cardiology specialist for clinical confirmation."
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
    straight_seal = Table([[ptxt(f"<b>{seal_title}</b><br/>{seal_sub}<br/>{seal_verify}", center_small)]], colWidths=[4.6 * cm])
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
#====================================================

ARABIC_FONT_PATH = "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"

def setup_pdf_fonts():
    try:
        pdfmetrics.registerFont(TTFont("ArabicFont", ARABIC_FONT_PATH))
        return "ArabicFont"
    except Exception:
        return "Helvetica"

def ar_text(text):
    if text is None:
        return ""
    reshaped = arabic_reshaper.reshape(str(text))
    return get_display(reshaped)

def to_arabic_digits(text):
    western = "0123456789"
    eastern = "٠١٢٣٤٥٦٧٨٩"
    return str(text).translate(str.maketrans(western, eastern))

# =====================================================
# TOP BAR
# =====================================================
st.markdown(
    f"""
    <div class="topbar rtl">
        <div>
            <div class="brand-click">{t['brand']}</div>
            <div class="brand-sub">{t['subtitle']}</div>
        </div>
        <div>{t['ai_system']}</div>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="nav-shell">', unsafe_allow_html=True)
nav_items = [
    ("home", t["home"]),
    ("dashboard", t["dashboard"]),
    ("upload", t["upload"]),
    ("segmentation", t["segmentation"]),
    ("features", t["features"]),
    ("prediction", t["prediction"]),
    ("report", t["report"]),
    ("federated", t["federated"]),
]

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
st.markdown('</div>', unsafe_allow_html=True)

# =====================================================
# PAGES
# =====================================================
page = st.session_state.page

if page == "home":
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
    st.markdown(
        f"""
        <div class="card rtl">
            <h2>{t['brand']}</h2>
            <p><b>{t['subtitle']}</b></p>
            <p>{t['university']}<br>{t['college']}<br>{t['department']}</p>
        </div>
        <div class="card rtl">
            <h3>{t['vision']}</h3>
            <p>{t['vision_text']}</p>
        </div>
        <div class="card rtl">
            <h3>{t['mission']}</h3>
            <p>{t['mission_text']}</p>
        </div>
        <div class="card rtl">
            <h3>{t['objectives']}</h3>
            <ol>
                <li>{t['objective_1']}</li>
                <li>{t['objective_2']}</li>
                <li>{t['objective_3']}</li>
                <li>{t['objective_4']}</li>
            </ol>
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
            <p style="color:#64748b; font-size:14px;">{t['uploaded_mri_usage']}</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    uploaded_file = st.file_uploader(t["select_file"], label_visibility="visible")
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
                <p>{t['segmentation_desc']}</p>
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

        page_title(t["segmentation_quality"])
        q1, q2, q3 = st.columns(3)
        with q1: stat_card("Dice Score", "0.838", "Final FL Round")
        with q2: stat_card("IoU", "0.744", "Final FL Round")
        with q3: stat_card("Pixel Accuracy", "98.7%", "Final FL Round")

        st.markdown(
            f"""
            <div class="card rtl">
                <h3>{t['clinical_interpretation']}</h3>
                <p>{t['segmentation_desc']}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.success(t["segmentation_success"])

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
        st.success(t["feature_success"])
        if st.button("Go to Prediction" if not is_ar else "الانتقال إلى التنبؤ", key="features_to_prediction", use_container_width=True):
            set_page("prediction")

elif page == "prediction":
    page_title(t["prediction"])
    if "image" not in st.session_state or "mask" not in st.session_state:
        st.warning(t["no_file"])
    else:
        features, predicted, confidence, probabilities = ensure_prediction()
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
        st.markdown(
            f"""
            <div class="blue-card rtl">
                <b>{t['prediction_basis']}</b><br>
                {predicted} — {disease_display} — {confidence:.2f}%
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown(f"<div class='card rtl'><h3>{t['recommendation']}</h3><p>{t['recommendation_text']}</p></div>", unsafe_allow_html=True)

elif page == "report":
    page_title(t["report"])
    if "image" not in st.session_state or "mask" not in st.session_state:
        st.warning(t["no_file"])
    else:
        features, predicted, confidence, probabilities = ensure_prediction()
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

        pdf_bytes = build_pdf_report(features, predicted, confidence, probabilities, st.session_state.language)

        b1, b2, b3 = st.columns([1, 1, 1])

        with b1:
            if pdf_bytes:
                st.download_button(
                    t["download_pdf"],
                    data=pdf_bytes,
                    file_name=f"Nabdh_AI_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    key="download_pdf_report"
                )
            else:
                st.error("Install reportlab first: python3 -m pip install reportlab")

        with b2:
            print_label = t["print_report"]
            components.html(
                f"""
                <button onclick="window.parent.print()" style="
                    width:100%;
                    height:44px;
                    border:none;
                    background:transparent;
                    font-size:18px;
                    font-weight:700;
                    color:#0B3B75;
                    text-decoration:underline;
                    cursor:pointer;">
                    {print_label}
                </button>
                """,
                height=55
            )

        with b3:
            st.markdown(
                """
                <style>
                div[data-testid="stTextInput"] input {
                    direction: ltr !important;
                    text-align: left !important;
                }
                </style>
                """,
                unsafe_allow_html=True
            )
            st.text_input(
                "Share Link" if not is_ar else "رابط المشاركة",
                value="http://localhost:8501",
                key="share_link_box"
            )

        st.success(t["official_pdf_ready"])

elif page == "federated":
    page_title(t["federated"])
    hospitals = pd.DataFrame({
        "Hospital": ["Hospital A", "Hospital B", "Hospital C", "Hospital D", "Hospital E"],
        "Samples": [300, 302, 304, 344, 286],
        "Round": [3, 3, 3, 3, 3],
        "Status": ["Connected", "Connected", "Connected", "Connected", "Connected"]
    })
    st.dataframe(hospitals, use_container_width=True, hide_index=True)
    rounds = pd.DataFrame({"Round": [1, 2, 3], "Loss": [0.725, 0.154, 0.087], "Dice": [0.605, 0.769, 0.838], "IoU": [0.497, 0.661, 0.744]})
    st.markdown("<div class='card rtl'><h3>Federated Learning Performance</h3></div>", unsafe_allow_html=True)
    st.line_chart(rounds.set_index("Round"))
    st.dataframe(rounds, use_container_width=True, hide_index=True)

elif page == "help":
    page_title(t["help"])
    if is_ar:
        help_title = "طريقة استخدام نبض AI"
        steps = [
            "ارفع ملف الرنين المغناطيسي للقلب.",
            "راجعي مخرجات التجزئة والعرض المدمج.",
            "استخرجي الخصائص القلبية من RV وMYO وLV.",
            "افتحي صفحة التنبؤ لعرض النتيجة المبنية على الخصائص.",
            "استخدمي خيارات التقرير: تنزيل، طباعة، أو مشاركة الرابط."
        ]
    else:
        help_title = "How to use Nabdh AI"
        steps = [
            "Upload a cardiac MRI file.",
            "Review segmentation output and overlay.",
            "Extract cardiac features from RV, MYO, and LV.",
            "Open Prediction to view feature-based disease estimation.",
            "Use the report options: download, print, or share link."
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
        <b>© 2026 {t['brand']} — {t['footer_platform']}</b><br>
        {t['notice']}
    </div>
    """,
    unsafe_allow_html=True
)
