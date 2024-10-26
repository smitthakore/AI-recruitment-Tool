import fitz  # PyMuPDF for handling PDF files
from io import BytesIO
from docx import Document
from .ner_extractor import extract_information_huggingface

def parse_pdf(file):
    """
    Extracts text from a PDF file-like object.
    """
    
    text = ""
    pdf_data = file.read()  # Read file bytes
    pdf_stream = BytesIO(pdf_data)  # Convert bytes to a byte stream

    with fitz.open("pdf", pdf_stream) as pdf:  # Specify format as 'pdf'
        for page in pdf:
            text += page.get_text()
    return text

def parse_docx(file):
    """
    Extracts text from a DOCX file-like object.
    """
    
    doc = Document(file)
    return "\n".join([paragraph.text for paragraph in doc.paragraphs])

def parse_and_extract_resume_keywords(resume_files):
    """
    Processes resume files in PDF and DOCX formats to extract relevant keywords.
    :param resume_files: List of uploaded resume files (PDF or DOCX).
    :return: List of dictionaries containing extracted information from each resume.
    """
    
    resume_keywords = []

    for file in resume_files:
        # Determine file type and parse accordingly
        if file.name.endswith(".pdf"):
            resume_text = parse_pdf(file)
        elif file.name.endswith(".docx"):
            resume_text = parse_docx(file)
        else:
            continue  # Skip unsupported file types

        # Extract relevant information using the NER model
        extracted_info = extract_information_huggingface(resume_text)
        extracted_info["resume_text"] = resume_text  # Keep the full text for reference
        resume_keywords.append(extracted_info)

    return resume_keywords
