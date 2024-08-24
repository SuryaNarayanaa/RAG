#!/usr/bin/env python
# coding: utf-8

# In[11]:


import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from semantic_chunkers import StatisticalChunker
from langchain_community.llms import Ollama


# In[12]:


CHUNK_OP_DIRECTORY_TXT = ".\\output\\chunks"
FILE_PATH = ".\output\\text\\Metamorphosis by Franz Kafka.txt"


TEXT_FILE_NAME = os.path.splitext(os.path.basename(FILE_PATH))[0]


# In[13]:


faiss_index_path = f'model_embeddings/{TEXT_FILE_NAME}_faiss.index'
vector_db_path =f'model_embeddings/{TEXT_FILE_NAME}_embeddings.pkl'


def load_text_chunks_from_folder(folder_path):
    text_chunks = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                text_chunks.append(file.read())

    print("TEXT LOADED SUCCESSFULLY")

    return text_chunks


def embed_text_chunks(text_chunks, model):

    return model.encode(text_chunks)


def save_embeddings(embeddings, text_chunks, path):
    with open(path, 'wb') as file:
        pickle.dump((text_chunks, embeddings), file)




def build_faiss_index(embeddings):
    dim = embeddings.shape[1]  
    index = faiss.IndexFlatL2(dim)  
    index.add(embeddings)  
    print("FAISS BUILT  SUCCESSFULLY")

    return index









def saving_the_model(CHUNK_OP_DIRECTORY_TXT):
    text_chunks = load_text_chunks_from_folder(CHUNK_OP_DIRECTORY_TXT)
    embeddings = embed_text_chunks(text_chunks, embedding_model)
    save_embeddings(embeddings, text_chunks, vector_db_path)

    faiss_index = build_faiss_index(embeddings)

    faiss.write_index(faiss_index, faiss_index_path)
    print("EMBEDDINGS SAVED SUCCESSFULLY")


# In[14]:


saving_the_model(CHUNK_OP_DIRECTORY_TXT)


# In[1]:







