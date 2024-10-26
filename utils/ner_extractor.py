import requests
import re

def extract_information_huggingface(text):
    '''
    Extracts key factors from input text such as skills, years of exp, education, etc.
    '''
    
    url = "https://api-inference.huggingface.co/models/dbmdz/bert-large-cased-finetuned-conll03-english"
    headers = {"Authorization": "Bearer hf_xxxxxxxxxxxxxxxxxxxxx"}
    payload = {"inputs": text}
    
    response = requests.post(url, headers=headers, json=payload)
    response_data = response.json()

    extracted_info = {
        "skills": [],
        "years_of_experience": [],
        "education": [],
        "certifications": [],
        "technologies": []
    }
    # NER or token classification model has pre built entities to classify each input token.
    if isinstance(response_data, list):
        for entity in response_data:
            if 'entity_group' in entity:
                label = entity['entity_group']
                text = entity['word']
                if label == "MISC":  # Skills
                    extracted_info["skills"].append(text)
                elif label == "EDU":  # Education
                    extracted_info["education"].append(text)
                elif label == "CERT":  # Certifications
                    extracted_info["certifications"].append(text)
                elif label == "TECH":  # Technologies
                    extracted_info["technologies"].append(text)

    experience_pattern = r'(\d+)\s*(?:years?|yr|yrs?)'
    experience_matches = re.findall(experience_pattern, text, re.IGNORECASE)
    extracted_info["years_of_experience"] = list(set(experience_matches))

    return extracted_info
