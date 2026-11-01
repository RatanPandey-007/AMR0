"""Clinical Decision-Support PDF Report Generator for AMR-Predict.

Produces publication-grade, standardized Antibiogram Intelligence reports
using ReportLab with patient demographics, laboratory markers, multi-agent
resistance risk matrix, model architecture provenance, and mandatory
research prototype and non-clinical-use disclaimers.
"""

import io
from datetime import datetime
from typing import Dict, Any

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable,
)

from src.config import MEDICAL_DISCLAIMER, MODEL_ATTRIBUTION_NOTICE


def generate_pdf_report(record: Dict[str, Any]) -> bytes:
    """Generates a professional PDF clinical report from a saved prediction record.

    Args:
        record: Full prediction record dictionary from database.

    Returns:
        Bytes of the generated PDF document.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#3D52A0"),
    )

    subtitle_style = ParagraphStyle(
        "DocSub",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=13,
        textColor=colors.HexColor("#7091E6"),
    )

    meta_style = ParagraphStyle(
        "DocMeta",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#666666"),
        alignment=2,  # Right aligned
    )

    section_heading = ParagraphStyle(
        "SectionHeading",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=15,
        textColor=colors.HexColor("#3D52A0"),
        spaceAfter=6,
    )

    body_style = ParagraphStyle(
        "BodyTextCustom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#222222"),
    )

    callout_style = ParagraphStyle(
        "CalloutStyle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=13,
        textColor=colors.HexColor("#8B0000"),
    )

    disclaimer_style = ParagraphStyle(
        "DisclaimerStyle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor("#555555"),
    )

    elements = []

    # 1. Header Banner
    pred_id = record.get("id", "PRD-UNKNOWN")
    rpt_id = f"RPT-{pred_id.replace('PRD-', '')}"
    analysis_date = record.get("created_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    header_table_data = [
        [
            Paragraph("<b>AMR-PREDICT &mdash; Antibiogram Intelligence Report</b>", title_style),
            Paragraph(f"<b>{rpt_id}</b><br/>Date: {analysis_date} UTC<br/>Status: Completed", meta_style),
        ],
        [
            Paragraph("AI-Powered Explainable Antimicrobial Resistance Decision Support &middot; Research Prototype", subtitle_style),
            Paragraph("Version: AMR-X v2.0", meta_style),
        ]
    ]
    header_table = Table(header_table_data, colWidths=[360, 180])
    header_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
    ]))
    elements.append(header_table)
    elements.append(Spacer(1, 10))
    elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#3D52A0"), spaceAfter=12))

    # 2. Patient Profile & Clinical Markers Grid
    elements.append(Paragraph("1. Patient Profile & Laboratory Markers", section_heading))

    patient_id = record.get("patient_id", "Unknown")
    age = record.get("age", 0)
    sex = "Female" if record.get("sex") == "F" else "Male" if record.get("sex") == "M" else record.get("sex", "Unknown")
    infection = str(record.get("infection_type", "Unknown")).replace("_", " ")
    organism = str(record.get("organism", "Unknown")).replace("_", " ")
    creat = record.get("creatinine_mg_dl", 0.0)
    wbc = record.get("wbc_count_k_ul", 0.0)

    factors = record.get("clinical_factors", {})
    factor_items = []
    if isinstance(factors, list):
        factor_items = [str(item) for item in factors]
    elif isinstance(factors, dict):
        if factors.get("recent_hospitalization_90d"):
            factor_items.append("Recent Hospitalization (90d)")
        if factors.get("recent_antibiotic_use_90d"):
            factor_items.append("Antibiotic Exposure (90d)")
        if factors.get("prior_resistant_culture_1yr"):
            factor_items.append("Prior Resistant Culture (1yr)")
        if factors.get("catheter_use"):
            factor_items.append("Indwelling Catheter")
        if factors.get("diabetes"):
            factor_items.append("Diabetes Mellitus")
        if factors.get("immunocompromised"):
            factor_items.append("Immunocompromised")
        if factors.get("nursing_home_resident"):
            factor_items.append("Nursing Home Resident")
        if factors.get("travel_last_6mo"):
            factor_items.append("Recent International Travel (6mo)")
        if factors.get("healthcare_worker"):
            factor_items.append("Healthcare Occupational Exposure")

    factors_str = ", ".join(factor_items) if factor_items else "No documented clinical exposure flags"

    patient_table_data = [
        [
            Paragraph("<b>Patient Identifier:</b>", body_style), Paragraph(str(patient_id), body_style),
            Paragraph("<b>Demographics:</b>", body_style), Paragraph(f"{age} Years / {sex}", body_style),
        ],
        [
            Paragraph("<b>Isolated Pathogen:</b>", body_style), Paragraph(f"<b>{organism}</b>", body_style),
            Paragraph("<b>Infection Syndrome:</b>", body_style), Paragraph(infection, body_style),
        ],
        [
            Paragraph("<b>Serum Creatinine:</b>", body_style), Paragraph(f"{creat} mg/dL", body_style),
            Paragraph("<b>WBC Count:</b>", body_style), Paragraph(f"{wbc} k/µL", body_style),
        ],
        [
            Paragraph("<b>Clinical Exposures:</b>", body_style),
            Paragraph(factors_str, body_style),
            Paragraph("", body_style), Paragraph("", body_style),
        ],
    ]

    patient_table = Table(patient_table_data, colWidths=[105, 165, 105, 165])
    patient_table.setStyle(TableStyle([
        ("SPAN", (1, 3), (3, 3)),
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8F9FD")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#DCE2F0")),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    elements.append(patient_table)
    elements.append(Spacer(1, 10))

    # 3. Highest Risk Callout Box
    highest_abx = record.get("highest_risk_antibiotic", "N/A")
    highest_prob = record.get("highest_risk_prob", 0.0)
    highest_cat = record.get("highest_risk_category", "Low")

    callout_data = [[
        Paragraph(
            f"<b>CRITICAL NON-SUSCEPTIBILITY ALERT:</b> Highest model-estimated resistance probability identified for "
            f"<b>{highest_abx}</b> ({highest_prob * 100:.1f}% &mdash; {highest_cat} Risk Tier).",
            callout_style
        )
    ]]
    callout_table = Table(callout_data, colWidths=[540])
    callout_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FFF0F0")),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#E04B4B")),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ]))
    elements.append(callout_table)
    elements.append(Spacer(1, 12))

    # 4. Multi-Agent Antibiogram Matrix
    elements.append(Paragraph("2. Multi-Agent Resistance Estimation Matrix (8 Targets)", section_heading))

    predictions = record.get("predictions", [])
    matrix_rows = [[
        Paragraph("<b>Target Antibiotic</b>", body_style),
        Paragraph("<b>Model Architecture</b>", body_style),
        Paragraph("<b>Estimated Risk</b>", body_style),
        Paragraph("<b>Risk Tier</b>", body_style),
        Paragraph("<b>Clinical Interpretation Label</b>", body_style),
    ]]

    for pred in predictions:
        p_val = pred.get("estimated_resistance_probability", 0.0)
        p_pct = f"{p_val * 100:.1f}%"
        tier = pred.get("risk_category", "Low")
        m_type = pred.get("model_type", "Ensemble")
        interp = pred.get("interpretation_label", "Resistance probability estimate")

        tier_color = "#E04B4B" if tier == "High" else "#D9822B" if tier == "Moderate" else "#2D9D78"

        matrix_rows.append([
            Paragraph(f"<b>{pred.get('antibiotic', 'N/A')}</b>", body_style),
            Paragraph(f"<font color='#3D52A0'><b>{m_type}</b></font>", body_style),
            Paragraph(f"<b>{p_pct}</b>", body_style),
            Paragraph(f"<font color='{tier_color}'><b>{tier} Risk</b></font>", body_style),
            Paragraph(f"<font size=8>{interp}</font>", body_style),
        ])

    matrix_table = Table(matrix_rows, colWidths=[150, 95, 80, 75, 140])
    matrix_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#3D52A0")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#DCE2F0")),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8F9FD")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    elements.append(matrix_table)
    elements.append(Spacer(1, 14))

    # 5. Model Provenance & Anti-Causality Disclaimers
    elements.append(Paragraph("3. Model Methodology, Ethical Scope & Limitations", section_heading))

    disclaimer_text = (
        f"<b>MANDATORY RESEARCH DISCLAIMER:</b> {MEDICAL_DISCLAIMER}<br/><br/>"
        f"<b>EXPLAINABILITY & ATTRIBUTION NOTICE:</b> {MODEL_ATTRIBUTION_NOTICE}<br/><br/>"
        "<b>DATASET LIMITATION:</b> Models are trained on a synthetic research cohort of 20,000 patient episodes. "
        "Biological and clinical performance on authentic laboratory culture isolates will vary substantially."
    )

    disclaimer_table = Table([[Paragraph(disclaimer_text, disclaimer_style)]], colWidths=[540])
    disclaimer_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F4F6FB")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CAD3E8")),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ]))
    elements.append(disclaimer_table)

    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()
