# insightflow.py
# InsightFlow — Autonomous Data Analyst Agent
# Built by Fêranmi Olufemi (@Feranmilux)

import pandas as pd
from google import genai
import io
import matplotlib.pyplot as plt
import matplotlib
import os
import datetime

# ReportLab Imports for Phase 5
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, 
                                Image, Table, TableStyle, HRFlowable)
from reportlab.lib.enums import TA_CENTER, TA_LEFT

matplotlib.use('Agg')  # works without display interface

# ── CONFIG ──────────────────────────────────────
# ⚠️ PASTE YOUR API KEY HERE 
API_KEY = "YOUR_API_KEY_HERE" 
client = genai.Client(api_key=API_KEY)

# ── STEP 1: LOAD THE DATA ───────────────────────
def load_dataset(file_path):
    """
    Load the dataset from the given file path.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return None

# ── STEP 2: INSPECT WITH AI ─────────────────────
def inspect_dataset(df):
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()

    prompt = f"""
You are an expert data scientist.
Here is the dataset information:

HEAD (first 5 rows):
{df.head().to_string()}

INFO:
{info_str}

Summarize what this dataset is about in 3 sentences.
Then list the 5 most interesting questions someone 
could ask about this data.
"""
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        print("\n🤖 Agent says:")
        print(response.text)
    except Exception as e:
        print(f"\n⚠️ Could not inspect data. Google's servers are busy right now. (Error: {e})")

# ── STEP 3: TOOL — RUN DATA MATH ────────────────
def run_data_math(df, code_str):
    """
    Safely executes a Pandas/NumPy expression
    on the dataframe and returns the result.
    """
    try:
        local_vars = {"df": df, "pd": pd}
        exec(f"result = {code_str}", {}, local_vars)
        return local_vars["result"]
    except Exception as e:
        return f"Error: {e}"

# ── STEP 4 & 6: REASONING WITH CONVERSATION MEMORY ──
def ask_agent_with_memory(df, user_question, history, chart_number):
    """
    Processes the question using conversational history so the agent
    can resolve context-dependent follow-ups.
    """
    columns_info = str(df.dtypes.to_dict())
    sample = df.head(3).to_string()

    history_str = ""
    for i, (q, a) in enumerate(history[-3:], 1):
        history_str += f"Q{i}: {q}\nA{i}: {a}\n\n"

    prompt = f"""
You are an expert data analyst with access to a Pandas DataFrame called 'df'.

Dataset columns and types: {columns_info}
Sample data:
{sample}

Previous conversation history:
{history_str}

Current question from the user: "{user_question}"

Based on the current question and the context of the previous conversation, write ONE line of Pandas code to extract the answer.
Return ONLY the raw executable code, nothing else. Do not use markdown format.
Example format: df.groupby('City')['Total'].mean()
"""
    # --- PHASE 1 & 2: THINK & RUN ---
    try:
        code_response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        pandas_code = code_response.text.strip().replace("```python", "").replace("```", "").strip()
        print(f"\n🔧 Agent generated code: {pandas_code}")
        
        result = run_data_math(df, pandas_code)
        print(f"📊 Raw result:\n{result}")
        
    except Exception as e:
        print(f"\n⚠️ Google's servers are a bit busy right now. Please press Enter and ask your question again! (Error: {e})")
        return "Error: Request failed due to server load."

    # ── TRIGGER AUTOMATIC CHART GENERATION ──
    generate_chart(df, user_question, result, chart_number)

    # --- PHASE 3: EXPLAIN ANSWER ---
    explain_prompt = f"""
The user asked: "{user_question}"
Previous conversation context:
{history_str}
The Pandas code ran and returned this result:
{result}

Explain this current result in 2-3 clear sentences as if talking to a business owner. Be specific with numbers.
"""
    try:
        explanation = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=explain_prompt
        )
        answer = explanation.text
        print(f"\n🤖 Agent Answer:")
        print(answer)
        return answer
    except Exception as e:
        print(f"\n📊 Generated data result successfully, but explanation phase timed out: {result}")
        return str(result)

# ── STEP 5: AUTO CHART ──────────────────────────
def generate_chart(df, user_question, result, chart_number=1):
    """
    Asks Gemini to write matplotlib code to visualize the result, saving sequentially.
    """
    columns_info = str(df.dtypes.to_dict())

    prompt = f"""
You are a data visualization expert.
The user asked: "{user_question}"
The result of the query was: {str(result)[:500]}
DataFrame columns: {columns_info}

Write ONE block of matplotlib code to visualize this data clearly. Use 'df' as the dataframe.

CRITICAL RULE: If the 'result' provided above is a single value, short string, or a category, do NOT plot just that single value. Instead, write code that plots a comprehensive breakdown related to the user's question across the whole dataframe.

General Layout Rules:
- Use plt.figure(figsize=(10,6))
- Add a clear title and axis labels
- Use plt.tight_layout()
- End with plt.savefig('insightflow_chart_{chart_number}.png', dpi=150, bbox_inches='tight')
- Do NOT use plt.show()
Return ONLY executable python code, no markdown formatting, no explanations.
"""
    try:
        chart_code = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        code = chart_code.text.strip().replace("```python", "").replace("```", "").strip()

        print(f"\n📊 Generating chart {chart_number}...")
        local_vars = {"df": df, "plt": plt, "pd": pd}
        exec(code, {}, local_vars)
        print(f"✅ Chart {chart_number} saved as 'insightflow_chart_{chart_number}.png'")
    except Exception as e:
        print(f"⚠️  Chart error: {e}")

# ── STEP 7: PDF REPORT ──────────────────────────
def generate_pdf_report(df, history, charts_generated):
    """
    Exports a full professional PDF report containing:
    - Dataset summary
    - All Q&A pairs from the session
    - All charts generated
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename  = f"InsightFlow_Report_{timestamp}.pdf"
    doc       = SimpleDocTemplate(filename, pagesize=A4,
                                   rightMargin=50, leftMargin=50,
                                   topMargin=60, bottomMargin=50)

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Title'],
        fontSize=24,
        textColor=colors.HexColor('#1a1a2e'),
        spaceAfter=6,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    subtitle_style = ParagraphStyle(
        'Subtitle',
        parent=styles['Normal'],
        fontSize=11,
        textColor=colors.HexColor('#4a4a8a'),
        spaceAfter=4,
        alignment=TA_CENTER
    )
    section_style = ParagraphStyle(
        'Section',
        parent=styles['Heading1'],
        fontSize=14,
        textColor=colors.HexColor('#1a1a2e'),
        spaceBefore=16,
        spaceAfter=8,
        fontName='Helvetica-Bold'
    )
    question_style = ParagraphStyle(
        'Question',
        parent=styles['Normal'],
        fontSize=11,
        textColor=colors.HexColor('#2563eb'),
        spaceBefore=10,
        spaceAfter=4,
        fontName='Helvetica-Bold'
    )
    answer_style = ParagraphStyle(
        'Answer',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.HexColor('#374151'),
        spaceAfter=8,
        leading=14
    )
    meta_style = ParagraphStyle(
        'Meta',
        parent=styles['Normal'],
        fontSize=9,
        textColor=colors.HexColor('#9ca3af'),
        alignment=TA_CENTER
    )

    story = []

    # ── HEADER ──────────────────────────────────
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph("InsightFlow", title_style))
    story.append(Paragraph("Autonomous Data Analysis Report", subtitle_style))
    story.append(Paragraph(
        f"Generated: {datetime.datetime.now().strftime('%B %d, %Y at %H:%M')}",
        meta_style
    ))
    story.append(Spacer(1, 0.2*inch))
    story.append(HRFlowable(width="100%", thickness=2,
                             color=colors.HexColor('#2563eb')))
    story.append(Spacer(1, 0.2*inch))

    # ── DATASET SUMMARY ──────────────────────────
    story.append(Paragraph("Dataset Summary", section_style))

    summary_data = [
        ["Metric", "Value"],
        ["Total Rows", f"{df.shape[0]:,}"],
        ["Total Columns", str(df.shape[1])],
        ["Columns", ", ".join(df.columns.tolist())],
        ["Questions Asked", str(len(history))],
        ["Charts Generated", str(charts_generated)],
    ]

    summary_table = Table(summary_data, colWidths=[2*inch, 4*inch])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1a1a2e')),
        ('TEXTCOLOR',  (0,0), (-1,0), colors.white),
        ('FONTNAME',   (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE',   (0,0), (-1,0), 10),
        ('BACKGROUND', (0,1), (-1,-1), colors.HexColor('#f8fafc')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1),
         [colors.HexColor('#f8fafc'), colors.white]),
        ('TEXTCOLOR',  (0,1), (-1,-1), colors.HexColor('#374151')),
        ('FONTNAME',   (0,1), (-1,-1), 'Helvetica'),
        ('FONTSIZE',   (0,1), (-1,-1), 9),
        ('GRID',       (0,0), (-1,-1), 0.5, colors.HexColor('#e5e7eb')),
        ('ROWHEIGHT',  (0,0), (-1,-1), 20),
        ('VALIGN',     (0,0), (-1,-1), 'MIDDLE'),
        ('LEFTPADDING',(0,0), (-1,-1), 8),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 0.2*inch))
    story.append(HRFlowable(width="100%", thickness=0.5,
                             color=colors.HexColor('#e5e7eb')))

    # ── Q&A SECTION ──────────────────────────────
    story.append(Paragraph("Analysis & Insights", section_style))

    for i, (question, answer) in enumerate(history, 1):
        story.append(Paragraph(f"Q{i}: {question}", question_style))
        # Clean answer text for PDF
        clean_answer = answer.replace('**', '').replace('*', '').replace('#', '')
        story.append(Paragraph(clean_answer, answer_style))

        # Add chart if it exists for this question
        chart_file = f"insightflow_chart_{i}.png"
        if os.path.exists(chart_file):
            try:
                img = Image(chart_file, width=5*inch, height=3*inch)
                story.append(img)
                story.append(Spacer(1, 0.1*inch))
            except:
                pass

        story.append(HRFlowable(width="100%", thickness=0.3,
                                  color=colors.HexColor('#e5e7eb')))
        story.append(Spacer(1, 0.1*inch))

    # ── FOOTER ───────────────────────────────────
    story.append(Spacer(1, 0.3*inch))
    story.append(HRFlowable(width="100%", thickness=1,
                             color=colors.HexColor('#2563eb')))
    story.append(Spacer(1, 0.1*inch))
    story.append(Paragraph(
        "Generated by InsightFlow | Built by Fêranmi Olufemi (@Feranmilux)",
        meta_style
    ))
    story.append(Paragraph(
        "github.com/Feranmilux | linkedin.com/in/feranmiolufemi",
        meta_style
    ))

    doc.build(story)
    print(f"\n✅ PDF Report saved: {filename}")
    return filename

# ── MAIN ────────────────────────────────────────
def main():
    print("=" * 50)
    print("  INSIGHTFLOW — Autonomous Data Analyst")
    print("=" * 50)

    filepath = input("\nEnter CSV file path: ").strip().strip('"')
    df = load_dataset(filepath)

    if df is not None:
        print("\nInspecting your dataset...")
        inspect_dataset(df)

        print("\n" + "=" * 50)
        print("Ask me anything about your data.")
        print("Type 'report' to export PDF.")
        print("Type 'quit' to exit.")
        print("=" * 50)

        # Track conversation memory and charts
        history = []
        chart_count = 0

        while True:
            question = input("\n❓ Your question: ").strip()
            
            if question.lower() == "quit":
                print("\n✅ Session complete.")
                break
                
            elif question.lower() == "report":
                if not history:
                    print("⚠️  Ask at least one question first.")
                    continue
                generate_pdf_report(df, history, chart_count)
                continue

            if question:
                chart_count += 1
                answer = ask_agent_with_memory(df, question, history, chart_count)
                history.append((question, answer))
                print(f"📝 Memory: {len(history)} questions remembered")

        # Auto export report at end if questions were asked
        if history:
            export = input("\nExport PDF report? (yes/no): ").strip().lower()
            if export == "yes":
                generate_pdf_report(df, history, chart_count)
                print("Goodbye!")
            else:
                print("Goodbye!")

if __name__ == "__main__":
    main()