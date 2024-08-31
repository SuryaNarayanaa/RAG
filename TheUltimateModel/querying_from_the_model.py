#!/usr/bin/env python
# coding: utf-8

# In[17]:


import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from langchain_community.llms import Ollama
from semantic_chunkers import StatisticalChunker


# In[18]:


embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


# In[19]:


embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
MODEL = "mistral"  
model = Ollama(model=MODEL)


# In[20]:


CHUNK_OP_DIRECTORY_TXT = ".\\output\\chunks"
FILE_PATH = ".\output\\text\\Metamorphosis by Franz Kafka.txt"


TEXT_FILE_NAME = os.path.splitext(os.path.basename(FILE_PATH))[0]
EMBEDDINGS_PATH  =f".\model_embeddings\{TEXT_FILE_NAME}_embeddings.pkl"

FAISS_INDEX  =f".\model_embeddings\{TEXT_FILE_NAME}_faiss.index"
FAISS_INDEX


# In[33]:


def format_output(context, question):
    """
    Use Mistral model to generate formatted output.
    """
    template = f"""
    >>> POINTS TO REMEMBER BEFORE GENERATING THE OUTPUT
        CONSIDER YOU ARE A CHATBOT WITH NO KNOWLEDGE.
        YOU WILL GAIN KNOWLEDGE ONLY WITH THE INFORMATION/CONTEXT I GIVE YOU.
        DON'T TRY TO ANSWER OUTSIDE OF THE INFORMATION I GIVE YOU.
        GENERATE THE OUTPUTS IN A STRUCTURED MANNER.
        IF THE ANSWER TO THE QUESTION IS OUT OF THE CONTEXT, THEN RETURN THAT "THE CONTEXT IS OUT OF THE KNOWLWDGE. NO RELEVANT INFORMATION FOUND"

    >>> INFORMATION/CONTEXT : {context}
    >>> QUERY : {question}

    
    

    
    """
    

    prompt_text = template.format(context=context, question=question)

    response = model(prompt_text)
    return response

def search_faiss(query, index, model, k=5):
    query_embedding = model.encode([query])
    D, I = index.search(np.array(query_embedding), k)  
    return I[0]  

def retrieve_and_format_results(query, index, text_chunks, model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')):
    indices = search_faiss(query, index, model)
    
    if not indices.size:
        return "No relevant information found."

    valid_indices = [i for i in indices if 0 <= i < len(text_chunks)]
    results = " ".join([text_chunks[i] for i in valid_indices]) 
    formatted_results = format_output(results ,query)
    return formatted_results




# In[34]:


def load_embeddings(path):
    with open(path, 'rb') as file:
        return pickle.load(file)


# In[35]:



def build_faiss_index(embeddings):
    dim = embeddings.shape[1]  # Dimension of embeddings
    index = faiss.IndexFlatL2(dim)  # L2 distance index
    index.add(embeddings)  # Add embeddings to index
    return index



# In[38]:





# In[ ]:




