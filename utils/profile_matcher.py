def match_profiles_rag(vector_db, job_keywords, resume_keywords):
    """
    RAG implementation: Matches job descriptions to resumes by finding the most relevant resumes for each job
    based on extracted keywords and similarity scores.
    
    :param vector_db: An instance of the VectorDB class for storing and searching vectors.
    :param job_keywords: List of job descriptions with extracted information and keywords.
    :param resume_keywords: List of resumes with extracted information and keywords.
    :return: A list of matches for each job description.
    """
    all_matches = []  # List to store matches for each job

    # For each job description, search for the most similar resumes
    for job in job_keywords:
        job_description = job["description"]
        
        # Find top-k matching resumes for this job description
        resume_matches = vector_db.search_similar(job_description, top_k=5)
        
        job_match = {"job": job, "matches": []}  # Store job and its top resume matches
        
        for resume_match in resume_matches:
            match_score = resume_match["score"]
            resume_data = resume_match["metadata"]

            job_match["matches"].append({
                "resume": resume_data,
                "score": match_score
            })
        
        all_matches.append(job_match)

    return all_matches
