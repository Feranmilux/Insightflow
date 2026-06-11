"""
InsightFlow — Autonomous Data Analyst Agent
Streamlit Web Interface
Built by Fêranmi Olufemi (@Feranmilux)
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import google.generativeai as genai
import io
import datetime
import os
import json
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                 Image, Table, TableStyle, HRFlowable)
from reportlab.lib.enums import TA_CENTER
import tempfile

# ── PAGE CONFIG ─────────────────────────────────────────────────
st.set_page_config(
    page_title="InsightFlow",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CUSTOM CSS ───────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .main-title {
        color: #ffffff;
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: 2px;
    }
    .main-subtitle {
        color: #90caf9;
        font-size: 1rem;
        margin-top: 0.5rem;
    }
    .metric-card {
        background: #f8fafc;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .question-bubble {
        background: #eff6ff;
        border-left: 4px solid #2563eb;
        padding: 0.8rem 1rem;
        border-radius: 0 8px 8px 0;
        margin: 0.5rem 0;
        font-weight: 600;
        color: #1e40af;
    }
    .answer-bubble {
        background: #f0fdf4;
        border-left: 4px solid #16a34a;
        padding: 0.8rem 1rem;
        border-radius: 0 8px 8px 0;
        margin: 0.5rem 0;
        color: #166534;
    }
    .code-bubble {
        background: #1e1e2e;
        color: #cdd6f4;
        padding: 0.6rem 1rem;
        border-radius: 8px;
        font-family: monospace;
        font-size: 0.85rem;
        margin: 0.3rem 0;
    }
    .stButton > button {
        background: linear-gradient(135deg, #2563eb, #1d4ed8);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #1d4ed8, #1e40af);
        transform: translateY(-1px);
    }
</style>
""", unsafe_allow_html=True)

# ── SESSION STATE ────────────────────────────────────────────────
if "history"       not in st.session_state: st.session_state.history       = []
if "df"            not in st.session_state: st.session_state.df            = None
if "model"         not in st.session_state: st.session_state.model         = None
if "charts"        not in st.session_state: st.session_state.charts        = []
if "api_ready"     not in st.session_state: st.session_state.api_ready     = False
if "dataset_name"  not in st.session_state: st.session_state.dataset_name  = ""

# ── GEMINI HELPERS ───────────────────────────────────────────────
def init_gemini(api_key):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        # Quick test
        model.generate_content("Say OK")
        return model
    except Exception as e:
        return None

def run_data_math(df, code_str):
    try:
        local_vars = {"df": df, "pd": pd}
        exec(f"result = {code_str}", {}, local_vars)
        return local_vars["result"]
    except Exception as e:
        return f"Error: {e}"

def inspect_dataset(df, model):
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    prompt = f"""
You are an expert data scientist.
Dataset HEAD:
{df.head().to_string()}
INFO:
{info_str}

1. Summarize what this dataset is about in 2 sentences.
2. List 5 smart questions someone could ask about this data.
Format as:
SUMMARY: ...
QUESTIONS:
- ...
- ...
"""
    response = model.generate_content(prompt)
    return response.text

def ask_agent(df, question, history, model):
    columns_info = str(df.dtypes.to_dict())
    sample       = df.head(3).to_string()
    history_str  = ""
    for i, (q, a, _) in enumerate(history[-3:], 1):
        history_str += f"Q{i}: {q}\nA{i}: {a}\n\n"

    # Step 1 — Get Pandas code
    code_prompt = f"""
You are an expert data analyst with a Pandas DataFrame called 'df'.
Columns: {columns_info}
Sample:
{sample}
Previous conversation:
{history_str}
Question: "{question}"
Write ONE line of Pandas code to answer this.
Return ONLY the code, nothing else.
Example: df.groupby('City')['Total'].mean()
"""
    code_resp   = model.generate_content(code_prompt)
    pandas_code = code_resp.text.strip().strip("```python").strip("```").strip()

    # Step 2 — Run code
    result = run_data_math(df, pandas_code)

    # Step 3 — Explain result
    explain_prompt = f"""
User asked: "{question}"
Pandas result: {str(result)[:800]}
Explain in 2-3 clear sentences for a business owner.
Be specific with numbers. Start directly with the answer.
"""
    explanation = model.generate_content(explain_prompt)
    answer      = explanation.text

    return pandas_code, result, answer

def generate_chart(df, question, result, model):
    columns_info = str(df.dtypes.to_dict())
    prompt = f"""
You are a data visualization expert with matplotlib.
User asked: "{question}"
Result: {str(result)[:500]}
DataFrame columns: {columns_info}

Write matplotlib code to visualize this result.
Rules:
- Use plt.figure(figsize=(10,5))
- Use df as the dataframe
- Add title, axis labels
- Use plt.tight_layout()
- End with plt.savefig(output_path, dpi=150, bbox_inches='tight')
- output_path is a variable already defined
- Do NOT use plt.show()
Return ONLY the code.
"""
    chart_code = model.generate_content(prompt)
    code = chart_code.text.strip().strip("```python").strip("```").strip()

    try:
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp.close()
        local_vars = {"df": df, "plt": plt, "output_path": tmp.name, "pd": pd}
        exec(code, {}, local_vars)
        plt.close('all')
        return tmp.name
    except Exception as e:
        plt.close('all')
        return None

# ── PDF EXPORT ───────────────────────────────────────────────────
def generate_pdf(df, history, dataset_name):
    tmp     = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    tmp.close()
    doc     = SimpleDocTemplate(tmp.name, pagesize=A4,
                                 rightMargin=50, leftMargin=50,
                                 topMargin=60, bottomMargin=50)
    styles  = getSampleStyleSheet()

    title_s = ParagraphStyle('T', parent=styles['Title'],
                              fontSize=22, textColor=colors.HexColor('#1a1a2e'),
                              spaceAfter=4, alignment=TA_CENTER,
                              fontName='Helvetica-Bold')
    sub_s   = ParagraphStyle('S', parent=styles['Normal'],
                              fontSize=10, textColor=colors.HexColor('#4a4a8a'),
                              spaceAfter=4, alignment=TA_CENTER)
    sec_s   = ParagraphStyle('H', parent=styles['Heading1'],
                              fontSize=13, textColor=colors.HexColor('#1a1a2e'),
                              spaceBefore=14, spaceAfter=6,
                              fontName='Helvetica-Bold')
    q_s     = ParagraphStyle('Q', parent=styles['Normal'],
                              fontSize=11, textColor=colors.HexColor('#2563eb'),
                              spaceBefore=10, spaceAfter=3,
                              fontName='Helvetica-Bold')
    a_s     = ParagraphStyle('A', parent=styles['Normal'],
                              fontSize=10, textColor=colors.HexColor('#374151'),
                              spaceAfter=6, leading=14)
    m_s     = ParagraphStyle('M', parent=styles['Normal'],
                              fontSize=8, textColor=colors.HexColor('#9ca3af'),
                              alignment=TA_CENTER)

    story = []
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph("InsightFlow", title_s))
    story.append(Paragraph("Autonomous Data Analysis Report", sub_s))
    story.append(Paragraph(
        f"Dataset: {dataset_name}  |  "
        f"Generated: {datetime.datetime.now().strftime('%B %d, %Y %H:%M')}",
        m_s))
    story.append(Spacer(1, 0.15*inch))
    story.append(HRFlowable(width="100%", thickness=2,
                             color=colors.HexColor('#2563eb')))
    story.append(Spacer(1, 0.15*inch))

    # Dataset summary table
    story.append(Paragraph("Dataset Summary", sec_s))
    data = [
        ["Metric",       "Value"],
        ["Rows",         f"{df.shape[0]:,}"],
        ["Columns",      str(df.shape[1])],
        ["Column Names", ", ".join(df.columns.tolist())],
        ["Questions",    str(len(history))],
    ]
    t = Table(data, colWidths=[2*inch, 4.1*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND',  (0,0), (-1,0),  colors.HexColor('#1a1a2e')),
        ('TEXTCOLOR',   (0,0), (-1,0),  colors.white),
        ('FONTNAME',    (0,0), (-1,0),  'Helvetica-Bold'),
        ('FONTSIZE',    (0,0), (-1,-1), 9),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),
         [colors.HexColor('#f8fafc'), colors.white]),
        ('GRID',        (0,0), (-1,-1), 0.4, colors.HexColor('#e5e7eb')),
        ('ROWHEIGHT',   (0,0), (-1,-1), 18),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('VALIGN',      (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.15*inch))
    story.append(HRFlowable(width="100%", thickness=0.4,
                             color=colors.HexColor('#e5e7eb')))

    # Q&A
    story.append(Paragraph("Analysis & Insights", sec_s))
    for i, (question, answer, chart_path) in enumerate(history, 1):
        story.append(Paragraph(f"Q{i}: {question}", q_s))
        clean = answer.replace('**','').replace('*','').replace('#','')
        story.append(Paragraph(clean, a_s))
        if chart_path and os.path.exists(chart_path):
            try:
                story.append(Image(chart_path,
                                   width=5*inch, height=3*inch))
                story.append(Spacer(1, 0.1*inch))
            except:
                pass
        story.append(HRFlowable(width="100%", thickness=0.3,
                                  color=colors.HexColor('#e5e7eb')))
        story.append(Spacer(1, 0.05*inch))

    # Footer
    story.append(Spacer(1, 0.3*inch))
    story.append(HRFlowable(width="100%", thickness=1,
                             color=colors.HexColor('#2563eb')))
    story.append(Spacer(1, 0.1*inch))
    story.append(Paragraph(
        "Generated by InsightFlow  |  "
        "Built by Fêranmi Olufemi (@Feranmilux)  |  "
        "github.com/Feranmilux",
        m_s))

    doc.build(story)
    return tmp.name

# ════════════════════════════════════════════════════════════════
#  UI
# ════════════════════════════════════════════════════════════════

# Header
st.markdown("""
<div class="main-header">
    <p class="main-title">📊 InsightFlow</p>
    <p class="main-subtitle">
        Autonomous Data Analyst Agent · Powered by Google Gemini
    </p>
</div>
""", unsafe_allow_html=True)

# ── SIDEBAR ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Setup")

    api_key = st.text_input("Google Gemini API Key",
                             type="password",
                             placeholder="AIza...")
    if api_key and not st.session_state.api_ready:
        with st.spinner("Connecting to Gemini..."):
            model = init_gemini(api_key)
        if model:
            st.session_state.model     = model
            st.session_state.api_ready = True
            st.success("✅ Gemini connected!")
        else:
            st.error("❌ Invalid API key")

    st.divider()
    st.markdown("### 📁 Upload Dataset")
    uploaded = st.file_uploader("Upload CSV file",
                                 type=["csv"],
                                 label_visibility="collapsed")
    if uploaded and st.session_state.api_ready:
        df = pd.read_csv(uploaded)
        st.session_state.df           = df
        st.session_state.dataset_name = uploaded.name
        st.session_state.history      = []
        st.session_state.charts       = []
        st.success(f"✅ {uploaded.name} loaded!")
        st.caption(f"{df.shape[0]:,} rows × {df.shape[1]} columns")

    st.divider()

    # Stats
    if st.session_state.df is not None:
        st.markdown("### 📈 Session Stats")
        col1, col2 = st.columns(2)
        col1.metric("Questions", len(st.session_state.history))
        col2.metric("Charts", len([c for _,_,c in st.session_state.history if c]))

    st.divider()

    # Clear session
    if st.button("🗑️ Clear Session"):
        st.session_state.history = []
        st.session_state.charts  = []
        st.rerun()

    st.divider()
    st.markdown("""
    <div style='text-align:center; color:#9ca3af; font-size:0.75rem;'>
    Built by <b>Fêranmi Olufemi</b><br>
    @Feranmilux · LAUTECH CS<br>
    Stanford Code in Place 2026
    </div>
    """, unsafe_allow_html=True)

# ── MAIN CONTENT ─────────────────────────────────────────────────
if not st.session_state.api_ready:
    st.info("👈 Enter your Gemini API key in the sidebar to get started.")
    st.markdown("""
    **How to get a free API key:**
    1. Go to [aistudio.google.com](https://aistudio.google.com)
    2. Click **Get API Key**
    3. Click **Create API key**
    4. Paste it in the sidebar
    """)

elif st.session_state.df is None:
    st.info("👈 Upload a CSV file in the sidebar to begin analysis.")

else:
    df    = st.session_state.df
    model = st.session_state.model

    # ── DATASET OVERVIEW ────────────────────────────────────────
    with st.expander("📋 Dataset Overview", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Rows",    f"{df.shape[0]:,}")
        col2.metric("Columns", df.shape[1])
        col3.metric("Dataset", st.session_state.dataset_name)
        col4.metric("Questions Asked", len(st.session_state.history))

        tab1, tab2, tab3 = st.tabs(["Preview", "Data Types", "Statistics"])
        with tab1:
            st.dataframe(df.head(10), use_container_width=True)
        with tab2:
            st.dataframe(pd.DataFrame({
                "Column":   df.columns,
                "Type":     df.dtypes.astype(str).values,
                "Non-Null": df.notna().sum().values,
                "Nulls":    df.isna().sum().values
            }), use_container_width=True)
        with tab3:
            st.dataframe(df.describe(), use_container_width=True)

    # ── AI DATASET SUMMARY ──────────────────────────────────────
    if st.button("🤖 Analyse Dataset with AI"):
        with st.spinner("Gemini is reading your dataset..."):
            summary = inspect_dataset(df, model)
        st.markdown("### 🤖 AI Analysis")
        st.markdown(summary)

    st.divider()

    # ── QUESTION INPUT ──────────────────────────────────────────
    st.markdown("### ❓ Ask About Your Data")
    col1, col2 = st.columns([4, 1])
    with col1:
        question = st.text_input("",
                                  placeholder="e.g. Which product line generates the most revenue?",
                                  label_visibility="collapsed")
    with col2:
        ask_btn = st.button("Ask Agent", use_container_width=True)

    # Quick question suggestions
    st.markdown("**Quick questions:**")
    qcols = st.columns(3)
    suggestions = [
        "Which category has highest sales?",
        "What is the average transaction value?",
        "Which month had the most revenue?",
        "Show top 5 products by profit",
        "What is the sales trend over time?",
        "Which city performs best?"
    ]
    for i, suggestion in enumerate(suggestions):
        if qcols[i % 3].button(suggestion, key=f"sug_{i}",
                                use_container_width=True):
            question = suggestion
            ask_btn  = True

    # Process question
    if ask_btn and question:
        with st.spinner("🤔 Agent is thinking..."):
            try:
                pandas_code, result, answer = ask_agent(
                    df, question,
                    st.session_state.history,
                    model
                )
                with st.spinner("📊 Generating chart..."):
                    chart_path = generate_chart(df, question, result, model)

                st.session_state.history.append(
                    (question, answer, chart_path)
                )
            except Exception as e:
                st.error(f"Agent error: {e}")

    st.divider()

    # ── CONVERSATION HISTORY ────────────────────────────────────
    if st.session_state.history:
        st.markdown("### 💬 Analysis History")

        for i, (q, a, chart_path) in enumerate(
                reversed(st.session_state.history), 1):
            idx = len(st.session_state.history) - i + 1
            with st.container():
                st.markdown(f"""
                <div class="question-bubble">
                    Q{idx}: {q}
                </div>
                """, unsafe_allow_html=True)
                st.markdown(f"""
                <div class="answer-bubble">
                    {a}
                </div>
                """, unsafe_allow_html=True)

                if chart_path and os.path.exists(chart_path):
                    st.image(chart_path, use_container_width=True)

                st.markdown("---")

        # ── PDF EXPORT ───────────────────────────────────────────
        st.markdown("### 📄 Export Report")
        if st.button("📥 Download PDF Report", use_container_width=True):
            with st.spinner("Building PDF report..."):
                try:
                    pdf_path = generate_pdf(
                        df,
                        st.session_state.history,
                        st.session_state.dataset_name
                    )
                    with open(pdf_path, "rb") as f:
                        pdf_bytes = f.read()
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
                    st.download_button(
                        label="⬇️ Click to Download",
                        data=pdf_bytes,
                        file_name=f"InsightFlow_Report_{timestamp}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                    st.success("✅ Report ready!")
                except Exception as e:
                    st.error(f"PDF error: {e}")
    else:
        st.info("Ask your first question above to start the analysis.")