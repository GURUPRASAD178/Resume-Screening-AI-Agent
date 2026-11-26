import io
import os
import re
import json
import sqlite3
import hashlib
from datetime import datetime, date

import pandas as pd
import streamlit as st
import plotly.express as px
from pypdf import PdfReader
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill
from openpyxl.utils import get_column_letter


st.set_page_config(page_title="AI Resume Screening Agent", page_icon="🧠", layout="wide")
DB = "screening.db"

SKILLS = {
    "Programming": ["python", "java", "c++", "c#", "javascript", "typescript", "sql", "r", "html", "css"],
    "Data & AI": ["machine learning", "deep learning", "artificial intelligence", "data analysis",
                  "data analytics", "data science", "nlp", "natural language processing",
                  "computer vision", "power bi", "tableau", "excel", "pandas", "numpy",
                  "scikit-learn", "tensorflow", "pytorch", "keras", "statistics", "spark",
                  "pyspark", "generative ai", "llm"],
    "Cloud & DevOps": ["aws", "azure", "gcp", "docker", "kubernetes", "jenkins", "git",
                       "github", "linux", "terraform"],
    "Databases": ["mysql", "postgresql", "mongodb", "oracle", "sql server", "sqlite", "redis"],
    "Web & Software": ["react", "angular", "node.js", "django", "flask", "fastapi",
                       "spring boot", "rest api", "api development", "streamlit"],
    "Business & Other": ["communication", "problem solving", "project management", "agile",
                         "scrum", "stakeholder management", "leadership", "critical thinking", "teamwork"]
}
ALL_SKILLS = {skill for group in SKILLS.values() for skill in group}


def db():
    conn = sqlite3.connect(DB, check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS candidates(
            id INTEGER PRIMARY KEY,
            fingerprint TEXT UNIQUE,
            name TEXT, email TEXT, phone TEXT, filename TEXT,
            resume_text TEXT, education TEXT, experience REAL, skills TEXT,
            created TEXT, status TEXT DEFAULT 'New', notes TEXT DEFAULT ''
        )
    """)
    conn.commit()
    return conn


def extract(file):
    content = file.getvalue()
    if file.name.lower().endswith(".pdf"):
        return "\n".join(page.extract_text() or "" for page in PdfReader(io.BytesIO(content)).pages)
    doc = Document(io.BytesIO(content))
    chunks = [p.text for p in doc.paragraphs]
    chunks += [" ".join(cell.text for cell in row.cells) for table in doc.tables for row in table.rows]
    return "\n".join(chunks)


def find_skills(text):
    content = (text or "").lower()
    found = []
    for skill in ALL_SKILLS:
        pattern = r"(?<![a-z0-9+#])" + re.escape(skill) + r"(?![a-z0-9+#])"
        if re.search(pattern, content):
            found.append(skill)
    return sorted(found)


def _month_year(value):
    """Convert a month/year or year string to a (year, month) pair."""
    value = value.strip().lower()
    if value in {"present", "current", "now", "till date"}:
        today = date.today()
        return today.year, today.month
    formats = ["%b %Y", "%B %Y", "%m/%Y", "%m-%Y", "%Y"]
    for fmt in formats:
        try:
            parsed = datetime.strptime(value.title() if "%b" in fmt or "%B" in fmt else value, fmt)
            return parsed.year, parsed.month if "%Y" != fmt else 1
        except ValueError:
            continue
    return None


def estimate_experience(text):
    """
    Extract explicit experience statements first. If absent, estimate from
    employment date ranges. Date-range estimates are approximate and may
    include overlapping roles, so overlapping periods are merged.
    """
    content = (text or "").lower()
    explicit_patterns = [
        r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)\s*(?:of\s*)?(?:professional\s*)?(?:work\s*)?(?:experience|exp)\b",
        r"(?:experience|exp)\s*(?:of|:)?\s*(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)\b",
        r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)\s*(?:in|with)\s+(?:the\s+)?(?:industry|software|data|development|IT)\b",
    ]
    explicit = []
    for pattern in explicit_patterns:
        explicit.extend(float(x) for x in re.findall(pattern, content, flags=re.I))
    if explicit:
        return round(max(explicit), 1), "Explicitly stated"

    month = r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
    date_token = rf"(?:{month}\s+\d{{4}}|\d{{1,2}}[/-]\d{{4}}|\d{{4}})"
    end_token = rf"(?:{month}\s+\d{{4}}|\d{{1,2}}[/-]\d{{4}}|\d{{4}}|present|current|till date|now)"
    range_pattern = re.compile(
        rf"({date_token})\s*(?:-|–|—|to|until)\s*({end_token})", re.I
    )

    intervals = []
    for line in (text or "").splitlines():
        # Date ranges are considered only on likely employment/project lines.
        if not re.search(r"work|experience|employment|developer|engineer|analyst|intern|consultant|manager|company|technologies|solutions|pvt|ltd", line, re.I):
            continue
        for start_raw, end_raw in range_pattern.findall(line):
            start = _month_year(start_raw)
            end = _month_year(end_raw)
            if start and end:
                start_idx = start[0] * 12 + start[1]
                end_idx = end[0] * 12 + end[1]
                if 0 <= end_idx - start_idx <= 12 * 45:
                    intervals.append((start_idx, end_idx))

    if not intervals:
        return None, "Not detected"

    intervals.sort()
    merged = []
    for start, end in intervals:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    months = sum(end - start for start, end in merged)
    return round(months / 12, 1), "Estimated from employment dates"


def profile(filename, text):
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines() if line.strip()]
    name = next(
        (line for line in lines[:8]
         if len(line) < 55 and not re.search(r"resume|curriculum vitae|@|phone|mobile|linkedin|\d{3,}", line, re.I)),
        os.path.splitext(filename)[0].replace("_", " ").title()
    )
    email = re.search(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", text)
    phone = re.search(r"(\+?\d[\d\s().-]{8,}\d)", text)
    experience, experience_source = estimate_experience(text)
    education = [
        line[:160] for line in lines
        if re.search(r"\b(B\.?Tech|B\.?E\.|M\.?Tech|M\.?E\.|MCA|BCA|B\.?Sc|M\.?Sc|MBA|Ph\.?D|Bachelor|Master|Diploma)\b", line, re.I)
    ]
    return {
        "name": name,
        "email": email.group(0) if email else "",
        "phone": phone.group(1) if phone else "",
        "experience": experience,
        "experience_source": experience_source,
        "education": "; ".join(education[:5]),
        "skills": find_skills(text)
    }


def required_experience(jd):
    years, _ = estimate_experience(jd)
    return years


def similarities(jd, texts, use_embeddings=False):
    if use_embeddings:
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("all-MiniLM-L6-v2")
            vectors = model.encode([jd] + texts, normalize_embeddings=True)
            return [float(value) * 100 for value in vectors[1:] @ vectors[0]]
        except Exception:
            st.warning("Sentence Transformers unavailable; using TF-IDF fallback.")
    matrix = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=15000).fit_transform([jd] + texts)
    return [float(value) * 100 for value in cosine_similarity(matrix[1:], matrix[0]).flatten()]


def rank(candidates, jd, weights, extra, semantic_scores):
    required = sorted(set(find_skills(jd)) | set(extra))
    required_years = required_experience(jd)
    total_weight = sum(weights)
    normalized = [weight * 100 / total_weight for weight in weights]
    results = []

    for candidate, semantic in zip(candidates, semantic_scores):
        found = set(candidate["profile"]["skills"])
        matched = sorted(set(required) & found)
        missing = sorted(set(required) - found)
        skill_pct = len(matched) / len(required) * 100 if required else 0
        years = candidate["profile"]["experience"]

        if required_years:
            experience_pct = min(years / required_years, 1) * 100 if years is not None else 0
            parts = {
                "Skill coverage": skill_pct * normalized[0] / 100,
                "Semantic relevance": semantic * normalized[1] / 100,
                "Experience evidence": experience_pct * normalized[2] / 100
            }
        else:
            # If the JD does not specify required years, redistribute the experience weight
            # to skill coverage rather than penalizing candidates for missing experience data.
            parts = {
                "Skill coverage": skill_pct * (normalized[0] + normalized[2]) / 100,
                "Semantic relevance": semantic * normalized[1] / 100
            }

        candidate.update({
            "matched": matched,
            "missing": missing,
            "skill_pct": skill_pct,
            "semantic_pct": semantic,
            "parts": parts,
            "score": round(max(0, min(100, sum(parts.values()))), 1),
            "required": required,
            "required_experience": required_years
        })
        results.append(candidate)
    return sorted(results, key=lambda item: item["score"], reverse=True)


def save_candidate(candidate, conn):
    fingerprint = hashlib.sha256((candidate["filename"] + candidate["text"]).encode(errors="ignore")).hexdigest()
    p = candidate["profile"]
    conn.execute(
        """INSERT OR IGNORE INTO candidates
        (fingerprint,name,email,phone,filename,resume_text,education,experience,skills,created)
        VALUES(?,?,?,?,?,?,?,?,?,?)""",
        (fingerprint, p["name"], p["email"], p["phone"], candidate["filename"],
         candidate["text"], p["education"], p["experience"], json.dumps(p["skills"]),
         datetime.now().isoformat(timespec="seconds"))
    )
    conn.commit()


def pdf_bytes(results, role):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(A4))
    styles = getSampleStyleSheet()
    story = [
        Paragraph("AI Resume Screening — Consolidated Report", styles["Title"]),
        Paragraph("Role: " + (role or "Not specified"), styles["Heading2"]),
        Paragraph("Scores are decision-support indicators, not probabilities of job success. Human review is required.", styles["BodyText"]),
        Spacer(1, 12)
    ]
    if results:
        best = results[0]
        story.append(Paragraph(f"Best matching resume: <b>{best['profile']['name']}</b> — {best['score']}/100", styles["Heading2"]))
        story.append(Paragraph(f"{len(best['matched'])}/{len(best['required'])} tracked skills detected; semantic relevance {best['semantic_pct']:.1f}%.", styles["BodyText"]))
    data = [["Rank", "Candidate", "Score / 100", "Skills %", "Semantic %", "Experience", "Status"]]
    for index, result in enumerate(results, 1):
        exp = result["profile"]["experience"]
        data.append([index, result["profile"]["name"], f"{result['score']}/100",
                     f"{result['skill_pct']:.1f}", f"{result['semantic_pct']:.1f}",
                     f"{exp} years" if exp is not None else "Not stated", result.get("status", "New")])
    table = Table(data, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#203858")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), .4, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#edf2f7")]),
        ("PADDING", (0, 0), (-1, -1), 6)
    ]))
    story += [Spacer(1, 10), table, Spacer(1, 12), Paragraph("Candidate explanations", styles["Heading2"])]
    for index, result in enumerate(results, 1):
        contributions = "; ".join(f"{key}: {value:.1f} points" for key, value in result["parts"].items())
        story.append(Paragraph(f"{index}. {result['profile']['name']} — {result['score']}/100", styles["Heading3"]))
        story.append(Paragraph("Score contributions: " + contributions, styles["BodyText"]))
        story.append(Paragraph("Matched: " + (", ".join(result["matched"]) or "None"), styles["BodyText"]))
        story.append(Paragraph("Missing: " + (", ".join(result["missing"]) or "None"), styles["BodyText"]))
        story.append(Spacer(1, 6))
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


def xlsx_bytes(results):
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Candidate Ranking"
    sheet.append(["Rank", "Candidate", "Resume", "Suitability / 100", "Skill coverage %", "Semantic relevance %",
                  "Experience years", "Experience detection", "Email", "Phone", "Education",
                  "Matched skills", "Missing skills", "Status"])
    for index, result in enumerate(results, 1):
        p = result["profile"]
        sheet.append([index, p["name"], result["filename"], result["score"], round(result["skill_pct"], 1),
                      round(result["semantic_pct"], 1), p["experience"], p["experience_source"], p["email"],
                      p["phone"], p["education"], ", ".join(result["matched"]), ", ".join(result["missing"]),
                      result.get("status", "New")])
    header_fill = PatternFill("solid", fgColor="203858")
    for cell in sheet[1]:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = header_fill
    for column in sheet.columns:
        sheet.column_dimensions[get_column_letter(column[0].column)].width = min(
            max(max(len(str(cell.value or "")) for cell in column) + 2, 12), 42
        )
    sheet.freeze_panes = "A2"
    sheet.auto_filter.ref = sheet.dimensions
    explanation = workbook.create_sheet("Score Explanation")
    explanation.append(["Candidate", "Component", "Contribution points"])
    for result in results:
        for component, value in result["parts"].items():
            explanation.append([result["profile"]["name"], component, round(value, 2)])
    for cell in explanation[1]:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = header_fill
    buffer = io.BytesIO()
    workbook.save(buffer)
    buffer.seek(0)
    return buffer.getvalue()


def interview_questions(result):
    questions = [
        (f"Technical — {skill}", f"Describe a project where you used {skill}. What was your role, challenge, approach and measurable outcome?")
        for skill in result["matched"][:4]
    ]
    questions += [
        (f"Gap validation — {skill}", f"This role requires {skill}. What related experience or learning have you had?")
        for skill in result["missing"][:3]
    ]
    questions += [
        ("Project deep dive", "Explain your most relevant project, your contribution, tools, trade-offs and results."),
        ("Problem solving", "Describe a difficult job-related problem you solved and how you validated the result."),
        ("Collaboration", "Describe a time you worked with a stakeholder or teammate with different priorities.")
    ]
    return questions[:10]


conn = db()
st.title("🧠 AI Resume Screening Agent")
st.caption("Explainable candidate ranking • Recruitment workflow • Human-in-the-loop")
tabs = st.tabs(["Screen resumes", "Compare candidates", "Interview assistant", "Candidate database", "Reports & exports", "Methodology"])

with tabs[0]:
    role = st.text_input("Job title / role")
    jd = st.text_area("Paste job description", height=180)
    uploads = st.file_uploader("Upload multiple PDF/DOCX resumes", type=["pdf", "docx"], accept_multiple_files=True)

    with st.expander("Scoring and semantic settings"):
        weight_skills = st.slider("Required skill coverage weight", 0, 100, 50)
        weight_semantic = st.slider("Semantic relevance weight", 0, 100, 35)
        weight_experience = st.slider("Experience evidence weight", 0, 100, 15)
        use_embeddings = st.checkbox("Use Sentence Transformers embeddings (downloads model on first use)", False)
    extra_skills = st.text_input("Additional must-have skills, comma-separated")

    if st.button("Analyze and rank candidates", type="primary", use_container_width=True):
        if not jd.strip() or not uploads:
            st.error("Enter a job description and upload at least one resume.")
        elif weight_skills + weight_semantic + weight_experience == 0:
            st.error("Set at least one non-zero weight.")
        else:
            candidates = []
            for uploaded_file in uploads:
                try:
                    resume_text = extract(uploaded_file)
                    if resume_text.strip():
                        item = {
                            "filename": uploaded_file.name,
                            "text": resume_text,
                            "profile": profile(uploaded_file.name, resume_text)
                        }
                        candidates.append(item)
                        save_candidate(item, conn)
                    else:
                        st.warning(f"No text extracted from {uploaded_file.name}; scanned PDFs need OCR.")
                except Exception as error:
                    st.error(f"{uploaded_file.name}: {error}")

            if candidates:
                try:
                    semantic_scores = similarities(jd, [item["text"] for item in candidates], use_embeddings)
                except Exception as error:
                    st.error(f"Similarity calculation failed: {error}")
                    semantic_scores = [0] * len(candidates)
                extras = [value.strip().lower() for value in extra_skills.split(",") if value.strip()]
                st.session_state.results = rank(
                    candidates, jd, [weight_skills, weight_semantic, weight_experience], extras, semantic_scores
                )
                st.session_state.role = role
                st.session_state.jd = jd

    results = st.session_state.get("results", [])
    if results:
        top = results[0]
        metric_cols = st.columns(4)
        metric_cols[0].metric("Resumes analyzed", len(results))
        metric_cols[1].metric("Best match", top["profile"]["name"])
        metric_cols[2].metric("Top score", f"{top['score']:.1f}/100")
        metric_cols[3].metric("Tracked skills", len(top["required"]))
        st.success(f"Best matching resume: {top['profile']['name']} ({top['score']:.1f}/100).")

        st.subheader("Candidate suitability scores")
        chart_df = pd.DataFrame([
            {"Candidate": f"{item['profile']['name']} — {item['filename']}",
             "Score": item["score"], "Score label": f"{item['score']:.1f}/100"}
            for item in results
        ])
        chart_df = chart_df.sort_values("Score", ascending=True)
        score_chart = px.bar(
            chart_df, x="Score", y="Candidate", orientation="h", text="Score label",
            range_x=[0, 100], title="Resume suitability score for every candidate (out of 100)"
        )
        score_chart.update_traces(textposition="outside", cliponaxis=False)
        score_chart.update_layout(xaxis_title="Suitability score (0–100)", yaxis_title="", height=max(400, 48 * len(chart_df)))
        st.plotly_chart(score_chart, use_container_width=True)

        st.subheader("Candidate ranking and score details")
        ranking_df = pd.DataFrame([
            {
                "Rank": index,
                "Candidate": item["profile"]["name"],
                "Resume": item["filename"],
                "Resume score (/100)": item["score"],
                "Skill coverage %": round(item["skill_pct"], 1),
                "Semantic relevance %": round(item["semantic_pct"], 1),
                "Experience (years)": item["profile"]["experience"],
                "Experience detection": item["profile"]["experience_source"]
            }
            for index, item in enumerate(results, 1)
        ])
        st.dataframe(ranking_df, use_container_width=True, hide_index=True)

        st.plotly_chart(
            px.bar(ranking_df, x="Candidate", y="Resume score (/100)", text="Resume score (/100)",
                   range_y=[0, 110], title="All candidates — resume score out of 100"),
            use_container_width=True
        )

        contribution_rows = [
            {"Candidate": item["profile"]["name"], "Component": component, "Contribution points": value}
            for item in results for component, value in item["parts"].items()
        ]
        st.plotly_chart(
            px.bar(pd.DataFrame(contribution_rows), x="Candidate", y="Contribution points",
                   color="Component", barmode="stack", title="Explainable score contributions"),
            use_container_width=True
        )

        st.subheader("Extracted candidate profiles")
        for index, result in enumerate(results, 1):
            p = result["profile"]
            with st.expander(f"{index}. {p['name']} — {result['score']:.1f}/100"):
                st.write({
                    "Email": p["email"] or "Not detected",
                    "Phone": p["phone"] or "Not detected",
                    "Experience": f"{p['experience']} years" if p["experience"] is not None else "Not detected",
                    "Experience detection": p["experience_source"],
                    "Education": p["education"] or "Not detected",
                    "Detected skills": ", ".join(p["skills"])
                })
                st.write("Score breakdown: " + "; ".join(f"{key}: {value:.1f} points" for key, value in result["parts"].items()))
                st.write("Matched:", ", ".join(result["matched"]) or "None")
                st.write("Missing:", ", ".join(result["missing"]) or "None")

with tabs[1]:
    results = st.session_state.get("results", [])
    st.subheader("Side-by-side candidate comparison")
    if len(results) < 2:
        st.info("Analyze at least two resumes first.")
    else:
        names = [f"{item['profile']['name']} ({item['filename']})" for item in results]
        selected = st.multiselect("Candidates", names, default=names[:min(4, len(names))])
        selected_set = set(selected)
        subset = [item for item, label in zip(results, names) if label in selected_set]
        if subset:
            compare_df = pd.DataFrame([
                {"Candidate": item["profile"]["name"], "Overall score (/100)": item["score"],
                 "Skill coverage": item["skill_pct"], "Semantic relevance": item["semantic_pct"],
                 "Experience years": item["profile"]["experience"]}
                for item in subset
            ])
            st.dataframe(compare_df, use_container_width=True, hide_index=True)
            st.plotly_chart(
                px.bar(compare_df.melt(id_vars="Candidate", var_name="Metric", value_name="Value"),
                       x="Candidate", y="Value", color="Metric", barmode="group",
                       title="Candidate comparison"),
                use_container_width=True
            )
            shared = set.intersection(*(set(item["matched"]) for item in subset))
            union = set.union(*(set(item["matched"]) for item in subset))
            st.write("Shared skills:", ", ".join(sorted(shared)) or "None")
            st.write("Skills found among selected candidates:", ", ".join(sorted(union)) or "None")

with tabs[2]:
    results = st.session_state.get("results", [])
    st.subheader("Candidate-specific interview questions")
    if not results:
        st.info("Analyze resumes first.")
    else:
        labels = [f"{item['profile']['name']} ({item['filename']})" for item in results]
        chosen = st.selectbox("Candidate", labels, key="interview_candidate")
        result = results[labels.index(chosen)]
        questions = interview_questions(result)
        for category, question in questions:
            st.markdown(f"**{category}**")
            st.write(question)
        st.download_button(
            "Download interview questions",
            "\n\n".join(f"{category}: {question}" for category, question in questions),
            file_name="interview_questions.txt"
        )

with tabs[3]:
    st.subheader("Local candidate database and shortlist pipeline")
    rows = conn.execute(
        "SELECT id,name,email,phone,filename,education,experience,created,status,notes FROM candidates ORDER BY created DESC"
    ).fetchall()
    if not rows:
        st.info("Analyzed resumes are saved automatically in screening.db.")
    else:
        database_df = pd.DataFrame(rows, columns=["ID", "Name", "Email", "Phone", "Resume", "Education",
                                                  "Experience", "Added", "Status", "Notes"])
        st.dataframe(database_df.drop(columns=["Notes"]), use_container_width=True, hide_index=True)
        candidate_id = st.selectbox(
            "Candidate record", [row[0] for row in rows],
            format_func=lambda value: next(f"{row[1]} (#{row[0]})" for row in rows if row[0] == value)
        )
        old = next(row for row in rows if row[0] == candidate_id)
        stages = ["New", "Reviewing", "Shortlisted", "Interview", "On hold", "Not selected"]
        status = st.selectbox("Recruitment stage", stages, index=stages.index(old[8]) if old[8] in stages else 0)
        notes = st.text_area("Recruiter notes", old[9] or "")
        if st.button("Save status and notes"):
            conn.execute("UPDATE candidates SET status=?,notes=? WHERE id=?", (status, notes, candidate_id))
            conn.commit()
            st.success("Candidate record updated.")
        st.metric("Shortlisted candidates", sum(row[8] == "Shortlisted" for row in rows))
        st.caption("Records are stored locally in screening.db. Secure the database; it contains personal information.")

with tabs[4]:
    results = st.session_state.get("results", [])
    if not results:
        st.info("Analyze resumes first to generate reports.")
    else:
        st.download_button("Download consolidated PDF report", pdf_bytes(results, st.session_state.get("role", "")),
                           file_name="AI_Resume_Screening_Report.pdf", mime="application/pdf", type="primary")
        st.download_button("Download Excel analytics workbook", xlsx_bytes(results),
                           file_name="AI_Resume_Screening_Analytics.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        export_rows = [
            {
                "Rank": index, "Candidate": item["profile"]["name"], "Resume": item["filename"],
                "Suitability score / 100": item["score"], "Skill coverage %": round(item["skill_pct"], 1),
                "Semantic relevance %": round(item["semantic_pct"], 1),
                "Experience years": item["profile"]["experience"],
                "Experience detection": item["profile"]["experience_source"],
                "Email": item["profile"]["email"], "Phone": item["profile"]["phone"],
                "Education": item["profile"]["education"], "Matched skills": ", ".join(item["matched"]),
                "Missing skills": ", ".join(item["missing"])
            }
            for index, item in enumerate(results, 1)
        ]
        st.download_button("Download detailed CSV", pd.DataFrame(export_rows).to_csv(index=False).encode(),
                           file_name="candidate_details.csv", mime="text/csv")

with tabs[5]:
    st.markdown("""
    ### Scoring model
    - **Skill coverage:** tracked job skills found in the resume divided by tracked job skills.
    - **Semantic relevance:** optional Sentence Transformers embeddings (`all-MiniLM-L6-v2`) or TF-IDF fallback.
    - **Experience evidence:** included when the job description states a years-of-experience requirement.
    - **Score:** weighted suitability score normalized to a maximum of 100.

    ### Experience extraction
    The parser first checks explicit statements such as `2 years of experience`, `3+ years`, or `experience: 2.5 years`.
    If no explicit value is found, it attempts an approximate estimate from employment date ranges such as `Jan 2021 – Mar 2024`.
    Resume layouts vary, so extracted experience should be checked against the original resume.

    ### Limitations
    Resume parsing is heuristic and may miss content in complex layouts. Scanned PDFs need OCR. Interview questions are template-generated from detected skills and gaps, not generated by a hosted LLM. Resume claims are not independently verified.

    ### Fairness, privacy and human oversight
    Use only job-related criteria. Do not score protected characteristics or proxies. Apply the same criteria to every applicant, review original resumes, verify claims, and keep qualified humans responsible for hiring decisions. Do not automatically reject candidates based on scores. Secure and delete local candidate records according to your retention policy.
    """)
st.caption("AI Resume Screening Agent Pro • Local prototype • Human review required")
