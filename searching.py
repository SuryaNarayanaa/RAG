#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_community.llms import Ollama

# Paths
MODEL = "mistral"  # Use the Mistral model
model = Ollama(model=MODEL)

text_chunks_folder = 'output/chunks/'
faiss_index_path = 'model_embeddings/faiss_index.index'
vector_db_path = 'model_embeddings/embeddings.pkl'

def load_text_chunks_from_folder(folder_path):
    text_chunks = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                text_chunks.append(file.read())
    return text_chunks

def embed_text_chunks(text_chunks, model):
    return model.encode(text_chunks)

def save_embeddings(embeddings, text_chunks, path):
    with open(path, 'wb') as file:
        pickle.dump((text_chunks, embeddings), file)

def load_embeddings(path):
    with open(path, 'rb') as file:
        return pickle.load(file)

def build_faiss_index(embeddings):
    dim = embeddings.shape[1]  # Dimension of embeddings
    index = faiss.IndexFlatL2(dim)  # L2 distance index
    index.add(embeddings)  # Add embeddings to index
    return index

from transformers import AutoTokenizer, AutoModelForCausalLM




def format_output(context, question):
    """
    Use Mistral model to generate formatted output.
    """
    # Define the template
    template = """
    Answer the question based on the context below. If you can't 
    answer the question, reply "I don't know".
    Only give me the answers based on the context below.
    Only answer the question asked. Do not provide additional information.
    Give a clear and concise answer.
    


    Context: {context}

    Question: {question}
    """

    # Format the template with context and question
    prompt_text = template.format(context=context, question=question)

    # Generate a response using the Mistral model
    response = model(prompt_text)
    # Return the formatted output
    return response


# Example usage





def search_faiss(query, index, model, k=5):
    query_embedding = model.encode([query])
    D, I = index.search(np.array(query_embedding), k)  # Search for top-k similar embeddings
    return I[0]  # Returns the indices of the most similar chunks

def retrieve_and_format_results(query, index, text_chunks, model):
    indices = search_faiss(query, index, model)
    
    # Handle case where no indices are returned
    if not indices.size:
        return "No relevant information found."

    # Check for valid indices
    valid_indices = [i for i in indices if 0 <= i < len(text_chunks)]
    results = " ".join([text_chunks[i] for i in valid_indices])  # Concatenate retrieved chunks
    
    formatted_results = format_output(results ,query)
    return formatted_results

# Initialize models
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Load or create embeddings and FAISS index
if os.path.exists(vector_db_path):
    text_chunks, embeddings = load_embeddings(vector_db_path)
else:
    text_chunks = load_text_chunks_from_folder(text_chunks_folder)
    embeddings = embed_text_chunks(text_chunks, embedding_model)
    save_embeddings(embeddings, text_chunks, vector_db_path)

faiss_index = build_faiss_index(embeddings)

# Example usage


# In[2]:


# query = "what are DNA made up of, explain in detail with help of flowchart" 
def return_formated_text(question):

    
    formatted_results = retrieve_and_format_results(question, faiss_index, text_chunks, embedding_model)
    return formatted_results
# print("RAG :(\n")
# print(formatted_results)


# In[ ]:




