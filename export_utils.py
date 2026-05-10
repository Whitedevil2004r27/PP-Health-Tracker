import csv
import io
import datetime
from fpdf import FPDF
from flask import make_response

def generate_clinical_pdf(data, forecast_text, username):
    """Generates an advanced Clinical PDF Dossier."""
    class PDFReport(FPDF):
        def header(self):
            self.set_fill_color(0, 0, 0)
            self.rect(0, 0, 210, 45, 'F')
            self.set_font('Arial', 'B', 22)
            self.set_text_color(255, 255, 255)
            self.cell(0, 25, 'PP HEALTH TRACKER', ln=True, align='C')
            self.set_font('Arial', 'B', 10)
            self.set_text_color(34, 197, 94)
            self.cell(0, 5, 'NEURAL NETWORK DIAGNOSTIC AUDIT', ln=True, align='C')
            self.ln(10)

        def footer(self):
            self.set_y(-15)
            self.set_font('Arial', 'I', 8)
            self.set_text_color(100, 100, 100)
            self.cell(0, 10, f'Page {self.page_no()} | Systemic Health Audit | PP Health Tracker', align='C')

    pdf = PDFReport()
    pdf.add_page()
    pdf.ln(15)

    # Info Section
    pdf.set_font("Arial", 'B', 12)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(100, 10, f"Patient: {username}")
    pdf.cell(90, 10, f"Date: {data['timestamp']}", align='R', ln=1)
    pdf.line(10, 62, 200, 62)
    pdf.ln(10)

    # Main Result Box
    pdf.set_fill_color(245, 255, 245)
    pdf.rect(10, 75, 190, 35, 'F')
    pdf.set_y(80)
    pdf.set_font("Arial", 'B', 16)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(190, 10, f"DIAGNOSIS: {data['prediction']}", align='C', ln=1)
    
    pdf.set_font("Arial", 'B', 12)
    pdf.set_text_color(34, 197, 94)
    pdf.cell(190, 10, f"CLINICAL RISK SCORE: {data['risk_score']}/100", align='C', ln=1)
    pdf.ln(15)

    # Probabilities Matrix
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(190, 10, "PROBABILISTIC DISTRIBUTION MATRIX:", ln=1)
    
    pdf.set_fill_color(240, 240, 240)
    for label, prob in zip(data['class_labels'], data['probabilities']):
        pdf.set_font("Arial", 'B', 10)
        pdf.cell(95, 8, f"  {label}", border=1, fill=True)
        pdf.set_font("Arial", '', 10)
        pdf.cell(95, 8, f"  {prob:.2%}", border=1, ln=1)
    pdf.ln(10)

    # Input Audit Table
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(190, 10, "SYSTEMIC BIOMARKERS (RAW INPUTS):", ln=1)
    pdf.set_font("Arial", '', 9)
    sorted_inputs = sorted(data['inputs'].items())
    
    pdf.set_fill_color(250, 250, 250)
    for i, (key, val) in enumerate(sorted_inputs):
        display_key = key.replace('_', ' ').title()
        fill = (i % 2 == 0)
        pdf.cell(95, 7, f"  {display_key}:", border=1, fill=fill)
        pdf.cell(95, 7, f"  {val}", border=1, ln=1, fill=fill)

    # Recommendations
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(190, 10, "CLINICAL RECOMMENDATIONS:", ln=1)
    pdf.set_font("Arial", '', 10)
    
    if 'Chronic' in data['prediction']:
        tips = ["- Immediate consultation with a fatigue specialist is recommended.",
                "- Implement 'Pacing' strategies to avoid Post-Exertional Malaise (PEM).",
                "- Focus on restorative rest and nutrient-dense anti-inflammatory diet."]
    elif 'Chances' in data['prediction']:
        tips = ["- Monitor your symptoms daily and reduce high-impact stress.",
                "- Improve sleep hygiene (7-9 hours of dark-room rest).",
                "- Introduce gentle light movement but avoid over-exertion."]
    else:
        tips = ["- Maintain a balanced lifestyle with regular hydration.",
                "- Regular physical check-ups to monitor baseline metrics.",
                "- Practice mindfulness to manage daily cognitive loads."]
    
    for tip in tips:
        pdf.multi_cell(190, 8, tip)

    # Forecasting
    pdf.ln(10)
    pdf.set_fill_color(34, 197, 94)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(190, 10, " NEURO-PREDICTIVE OUTLOOK (7-DAY PROJECTION)", ln=1, fill=True)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", 'I', 10)
    pdf.ln(5)
    pdf.multi_cell(190, 8, forecast_text)
    pdf.ln(10)
    pdf.set_font("Arial", '', 8)
    pdf.set_text_color(150, 150, 150)
    pdf.multi_cell(190, 5, "NOTE: Projections are based on linear regression of historical markers and intended for clinical guidance only. Statistical confidence increases with audit frequency.")

    output = pdf.output(dest='S')
    response = make_response(output)
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = f'attachment; filename=Clinical_Dossier_{datetime.datetime.now().strftime("%Y%m%d")}.pdf'
    return response

def generate_historical_csv(user_predictions):
    """Generates a CSV string containing all historical predictions for GDPR export."""
    si = io.StringIO()
    cw = csv.writer(si)
    
    # Headers
    cw.writerow(['ID', 'Timestamp', 'Prediction_Result', 'Raw_Inputs_JSON'])
    
    for row in user_predictions:
        cw.writerow([row[0], row[1], row[2], row[3]])
        
    output = si.getvalue()
    response = make_response(output)
    response.headers["Content-Disposition"] = f"attachment; filename=Historical_Data_Export_{datetime.datetime.now().strftime('%Y%m%d')}.csv"
    response.headers["Content-type"] = "text/csv"
    return response
