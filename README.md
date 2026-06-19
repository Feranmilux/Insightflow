# 📊 InsightFlow

InsightFlow is an Autonomous Data Analyst Agent built with Python and the Google Gemini API. 

Instead of writing code manually, you simply upload a dataset and ask questions in plain English. The agent autonomously writes Pandas code, executes it, explains the insights, generates Matplotlib charts, and exports everything into a clean PDF report.

### 🛠️ Tech Stack
* **Language:** Python
* **LLM:** Google Gemini 1.5 Flash
* **Data & Viz:** Pandas, Matplotlib
* **Reporting:** ReportLab
* **Frontend:** Streamlit

### 🚀 How to Run Locally
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `streamlit run app.py`
4. Enter your Gemini API key in the sidebar and upload your CSV!

#### 🛡️ Data Quality & Data Integrity Guardrails

To ensure high-fidelity entity resolution and reporting accuracy, InsightFlow handles incoming user datasets with built-in data hygiene steps:
1. **Schema Validation:** Automatically parses uploaded CSV files to ensure columns match structural expectations before passing queries to the LLM.
2. **Deterministic Processing:** Leverages strict system prompting inside the Google Gemini pipeline to force exact Pandas calculation execution, eliminating programmatic hallucinations.
3. **Data Sanitization:** Trims whitespace, standardizes casing, and handles missing or null data
