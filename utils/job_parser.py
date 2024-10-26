from .ner_extractor import extract_information_huggingface

def parse_txt(file):
    """
    Extracts text from a TXT file.
    """
    return file.read().decode("utf-8")

def parse_and_extract_job_keywords(job_files):
    """
    Processes job description files in TXT format to extract relevant keywords.
    
    :param job_files: List of uploaded job description files (TXT).
    :return: List of dictionaries containing extracted information from each job description.
    """
    job_keywords = []

    for file in job_files:
        if file.name.endswith(".txt"):
            job_description = parse_txt(file)
            extracted_info = extract_information_huggingface(job_description)
            extracted_info["description"] = job_description  # Keep the full text for reference
            job_keywords.append(extracted_info)
        else:
            continue  # Skip unsupported file types

    return job_keywords
