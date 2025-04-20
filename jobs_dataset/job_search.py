import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 1. Load and prepare data
def load_job_data(filepath):
    df = pd.read_excel(filepath, nrows=100000)
    
    # Explicitly cast columns to string dtype where applicable
    string_columns = [
        'id',
        'title',
        'company',
        'date_posted',
        'is_remote',
        'interval',
        'min_amount',
        'max_amount',
        'company_industry',
        'emails',
        'location', 'job_type', 'currency']
    df[string_columns] = df[string_columns].astype(str)
    
    df.fillna('', inplace=True)
    
    # Create a combined text field for semantic search
    df['search_text'] = df.apply(lambda row: (
        f"Title: {row['title']}. "
        f"Company: {row['company']}. "
        f"Location: {row['location']}. "
        f"Type: {row['job_type']}. "
        f"Salary: {row['min_amount']}-{row['max_amount']} {row['currency']}. "
        f"Remote: {'Yes' if row['is_remote'] else 'No'}. "
    ), axis=1)
    
    return df

# 2. Initialize models
class JobSearchAssistant:
    def __init__(self):
        # Semantic search model
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Phi-2 for natural language responses
        self.llm_tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            "microsoft/phi-2",
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
    def embed_text(self, text):
        return self.embedder.encode(text)
    
    def generate_response(self, prompt, context=""):
        full_prompt = f"""
        Context about available jobs:{context}
        Question: {prompt}
        Provide a helpful, detailed answer based on the job listings:
        """
        
        # Truncate the full_prompt to fit within the model's max length
        max_input_length = self.llm_tokenizer.model_max_length
        truncated_prompt = full_prompt[:max_input_length]
        print(f"Truncated prompt length: {len(truncated_prompt)}")
        inputs = self.llm_tokenizer(truncated_prompt, return_tensors="pt").to(self.llm_model.device)
        
        outputs = self.llm_model.generate(
            **inputs, 
            max_length=592,
            # max_new_tokens=1000, 
            temperature=0.7, 
            do_sample=True
        )
        
        return self.llm_tokenizer.decode(outputs[0], skip_special_tokens=True)

# 3. Search and response functions
def find_relevant_jobs(df, query, embedder, top_k=5):
    query_embed = embedder.embed_text(query)
    job_embeddings = embedder.embed_text(df['search_text'].tolist())
    
    # Calculate similarity scores
    scores = np.dot(query_embed, np.array(job_embeddings).T)
    top_indices = np.argsort(scores)[-top_k:][::-1]
    
    return df.iloc[top_indices]

def format_job_results(jobs_df):
    results = []
    for _, job in jobs_df.iterrows():
        results.append(
            f"• {job['title']} at {job['company']}\n"
            f"  - Location: {job['location']} {'(Remote)' if job['is_remote'] else ''}\n"
            f"  - Salary: {job['min_amount']}-{job['max_amount']} {job['currency']}\n"
            f"  - Type: {job['job_type']}\n"
            f"  - Posted: {job['date_posted']}\n"
        )
    return "\n\n".join(results)

# 4. Main interaction loop
def main():
    # Initialize
    df = load_job_data("./all_jobs.xlsx")
    assistant = JobSearchAssistant()
    
    print("Job Search Assistant initialized. Enter your questions about job openings.")
    
    while True:
        try:
            # Get user query
            query = input("\nWhat would you like to know? (or 'quit' to exit)\n> ")
            if query.lower() in ['quit', 'exit']:
                break
                
            # Find relevant jobs
            relevant_jobs = find_relevant_jobs(df, query, assistant)
            context = format_job_results(relevant_jobs)
            
            # Generate natural language response
            response = assistant.generate_response(query, context)
            
            # Display results
            print("\nHere are the most relevant job openings:")
            print(response.split("Answer:")[-1].strip())
            
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()