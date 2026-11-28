# AI Resume Screening Agent

[![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-TBD-lightgrey)](#license)

**An explainable, human-in-the-loop resume screening workspace for ranking candidates against a job description.**

Upload PDF or DOCX resumes, configure the scoring model, compare candidates, manage recruitment stages, generate interview questions, and export professional reports - all through a local Streamlit interface.

---

## Highlights

- **Multi-resume analysis** for PDF and DOCX files
- **Transparent suitability scoring** with configurable skill, semantic, and experience weights
- **Candidate profile extraction** for name, email, phone, education, experience, and skills
- **Side-by-side comparisons** with interactive Plotly visualizations
- **Candidate-specific interview prompts** based on matched skills and gaps
- **Local recruitment pipeline** with statuses and recruiter notes
- **Consolidated exports** as PDF, Excel, and CSV
- **Human oversight first**: scores support decisions but do not replace recruiter judgment

---

## How It Works

```text
PDF / DOCX Resume ──> Text Extraction ──> Candidate Profile
                                                |
                                                v
Job Description ────> Skills + Experience ---> Weighted Ranking
                                                |
                                                v
                      Candidate Dashboard + Reports + Shortlist
```

1. Enter a job title and paste the job description.
2. Upload one or more PDF or DOCX resumes.
3. Adjust the scoring weights or enable optional embeddings.
4. Analyze the resumes and review each candidate's score breakdown.
5. Compare candidates, prepare interview questions, update pipeline stages, and export reports.

---

## Scoring Model

The overall score is normalized to a maximum of 100.

| Component | Default weight | What it measures |
|---|---:|---|
| Skill coverage | 50 | Required tracked skills found in the resume |
| Semantic relevance | 35 | Similarity between the job description and resume |
| Experience evidence | 15 | Extracted experience relative to the requirement |

Weights are fully configurable. If the job description does not state a required number of years, the experience weight is redistributed to skill coverage.

### Semantic analysis

- **Sentence Transformers** can use `all-MiniLM-L6-v2` for embedding-based similarity.
- The first embeddings run downloads the model.
- **TF-IDF with cosine similarity** is used as the automatic fallback when embeddings are unavailable.

---

## Application Views

| View | Purpose |
|---|---|
| **Screen resumes** | Upload resumes, configure scoring, and review ranked results |
| **Compare candidates** | Compare shortlisted candidates and shared skills |
| **Interview assistant** | Generate candidate-specific technical, gap, and behavioral questions |
| **Candidate database** | Manage pipeline stages and private recruiter notes |
| **Reports & exports** | Download PDF, Excel, and CSV reports |
| **Methodology** | Review scoring logic, limitations, and responsible-use guidance |

---

## Tech Stack

- **Interface:** Streamlit
- **Data:** Pandas, SQLite
- **Visualization:** Plotly
- **Document parsing:** PyPDF, python-docx
- **Similarity:** scikit-learn TF-IDF; optional sentence-transformers
- **Exports:** ReportLab and openpyxl

---

## Getting Started

### 1. Clone the repository

```powershell
git clone https://github.com/GURUPRASAD178/Resume-Screening-AI-Agent.git
cd Resume-Screening-AI-Agent
```

### 2. Create a virtual environment

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
py -m pip install --upgrade pip
py -m pip install streamlit pandas plotly pypdf python-docx scikit-learn reportlab openpyxl
```

For optional embedding support:

```powershell
py -m pip install sentence-transformers
```

### 4. Run the application

```powershell
py -m streamlit run app.py
```

Open the local URL shown by Streamlit, usually `http://localhost:8501`.

---

## Project Structure

```text
.
├── app.py          # Streamlit interface and screening pipeline
├── screening.db    # Local SQLite candidate database
└── README.md       # Project documentation
```

---

## Data and Privacy

Candidate information is saved locally in `screening.db`, including names, contact details, resume text, extracted skills, recruitment status, and recruiter notes.

- Use only for authorized recruitment workflows.
- Use only job-related criteria.
- Do not infer protected characteristics or make proxies for them.
- Verify resume claims against the source documents.
- Secure the database and apply an appropriate retention and deletion policy.
- Keep qualified humans responsible for hiring decisions.
- Do not reject candidates solely because of an automated score.

---

## Known Limitations

- Resume parsing is heuristic and complex layouts may be parsed incorrectly.
- Scanned PDFs require OCR because text extraction alone may produce no usable content.
- Extracted experience is an estimate when it must be derived from date ranges.
- Interview questions are template-generated; they are not produced by a hosted LLM.
- Scores are decision-support indicators, not predictions of job performance.

---


