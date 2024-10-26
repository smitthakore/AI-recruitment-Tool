import faiss
import numpy as np
import requests
import uuid

class VectorDB:
    def __init__(self):
        self.embedding_dim = 384  # Dimension for all-MiniLM-L6-v2 embeddings
        self.index = faiss.IndexFlatL2(self.embedding_dim)  # L2 distance index
        self.job_ids = []  # List to keep track of job IDs
        self.resume_ids = []  # List to keep track of resume IDs
        self.embeddings = {}  # Dictionary to map IDs to metadata

        # Hugging Face Inference API setup
        # Embeddings model
        self.hf_url = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"

        self.hf_headers = {"Authorization": "Bearer hf_xxxxxxxxxxxxxxxxxxxxx"}


    def get_embedding(self, text):
        """
        Fetches an embedding from Hugging Face API for a given text.
        """
        
        headers = {
            "Authorization": f"Bearer hf_xxxxxxxxxxxxxxxxxxxxxx",  
            "Content-Type": "application/json",
        }

        # Directly pass the text instead of using a dictionary with "inputs"
        response = requests.post(self.hf_url, headers=headers, json=text)

        # Log the response content on failure for debugging
        if response.status_code != 200:
            print("Error in Hugging Face response:", response.json())
            response.raise_for_status()  

        # Parse the response assuming it returns a list of embeddings
        response_data = response.json()
        if isinstance(response_data, list) and all(isinstance(val, float) for val in response_data):
                embedding = response_data
                return np.array(embedding, dtype="float32")
        else:
            raise ValueError("Unexpected response structure from Hugging Face API:", response_data)


    def store_jobs(self, job_keywords):
        for job in job_keywords:
            description = job["description"]
            embedding = self.get_embedding(description)
            job_id = str(uuid.uuid4())
            
            # Store in FAISS and keep track of metadata
            self.index.add(np.array([embedding]))
            self.job_ids.append(job_id)
            self.embeddings[job_id] = job

    def store_resumes(self, resume_keywords):
        for resume in resume_keywords:
            resume_text = resume["resume_text"]
            embedding = self.get_embedding(resume_text)
            resume_id = str(uuid.uuid4())

            # Store in FAISS and keep track of metadata
            self.index.add(np.array([embedding]))
            self.resume_ids.append(resume_id)
            self.embeddings[resume_id] = resume

    def search_similar(self, text, top_k=5):
        embedding = self.get_embedding(text).reshape(1, -1)
        distances, indices = self.index.search(embedding, top_k)  # Find top-k similar entries

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            matched_id = self.job_ids[idx] if idx < len(self.job_ids) else self.resume_ids[idx - len(self.job_ids)]
            results.append({
                "metadata": self.embeddings[matched_id],
                "score": float(dist)  # Convert distance to a similarity score
            })
        return results
