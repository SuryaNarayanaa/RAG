#!/usr/bin/env python
# coding: utf-8

# In[31]:


import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from semantic_chunkers import StatisticalChunker
from semantic_router.encoders import HuggingFaceEncoder



# In[59]:


CHUNK_OP_DIRECTORY_TXT = ".\\output\\chunks"
FILE_PATH = ".\output\\text\\Metamorphosis by Franz Kafka.txt"


TEXT_FILE_NAME = os.path.splitext(os.path.basename(FILE_PATH))[0]


# In[60]:


embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


# In[35]:


embedding_model_chunks =HuggingFaceEncoder()
chunker= StatisticalChunker(encoder=embedding_model_chunks)


# In[52]:


def read_entire_txt_file(FILE_PATH):

    
    with open(FILE_PATH , 'r') as f:
        text = f.read()
        f.close()
    print("TEXT READ SUCCESSFUL")
    return text



# In[53]:


def chunk_and_save_as_txt(text):

    chunks= chunker([text])
    splits = [chunk.splits for sublist in chunks for chunk in sublist]
    for i , indv_chunk in enumerate(splits):
        CHUNK_PATH = os.path.join(CHUNK_OP_DIRECTORY_TXT , f"chunk_{i+1}")
        with open(CHUNK_PATH , 'w') as f:
            f.write(str(indv_chunk))
            f.close()
    
    print("CHUNK SAVE SUCCESSFUL")
    
    

        



# In[54]:


def generate_chunks(path):
    text = read_entire_txt_file(path)

    chunk_and_save_as_txt(text)
    print("CHUNKS GENERATED SUCCESSFULLY")
    return


# In[55]:


generate_chunks(FILE_PATH)


# In[61]:





# In[ ]:




