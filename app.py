# app.py
from utils.interview_analysis import analyze_interview
import streamlit as st
from utils.job_parser import parse_and_extract_job_keywords
from utils.resume_parser import parse_and_extract_resume_keywords
from utils.profile_matcher import match_profiles_rag
from utils.vector_database import VectorDB

# Initialize vector database
vector_db = VectorDB()

st.sidebar.title("AI Recruitment Tool")
app_mode = st.sidebar.selectbox("Choose the app mode", ["Job Matching", "Interview Analysis"])

if app_mode == "Job Matching":
    st.title("Upload Job Descriptions")
    job_files = st.file_uploader("Upload job descriptions", type=["txt", "json"], accept_multiple_files=True)

    st.subheader("Upload Resumes")
    resume_files = st.file_uploader("Upload candidate resumes", type=["pdf", "docx"], accept_multiple_files=True)

    if st.button("Process"):
        if job_files and resume_files:
            # Parse and extract keywords from job descriptions
            job_keywords = parse_and_extract_job_keywords(job_files)
            # Parse and extract keywords from resumes
            resume_keywords = parse_and_extract_resume_keywords(resume_files)

            # Store job descriptions and resumes in vector database
            vector_db.store_jobs(job_keywords)
            vector_db.store_resumes(resume_keywords)

            # Perform profile matching using RAG
            matches = match_profiles_rag(vector_db, job_keywords, resume_keywords)

            for match in matches:
                st.write(f"**Resume:** {match}")
                for job_match in match["matches"]:
                    st.write(f"- **Job:** {job_match}")
                    st.write(f"  - Similarity Score: {job_match['score']}")

        else:
            st.error("Please upload both job descriptions and resumes.")

else:
    # st.title("Interview Analysis")
    analyze_interview()
