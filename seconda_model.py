from sentence_transformers import SentenceTransformer
import numpy as np
import faiss  
import fitz  
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
import streamlit as st




def extract_text_with_references(pdf_path):
    text_chunks = []
    with fitz.open(stream=pdf_path.read(), filetype="pdf") as doc:

        for page_num, page in enumerate(doc):
            text = page.get_text()
            text_chunks.append({
                'text': text,
                'page': page_num
            })
    return text_chunks

model = SentenceTransformer('all-MiniLM-L6-v2')


def create_index(embeddings):
     index = faiss.IndexFlatL2(embeddings.shape[1])
     index.add(embeddings)
     return index

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceInstructEmbeddings(model_name = "hkunlp/instructor-xl")
    vetorstore = FAISS.from_texts(texts = text_chunks , embeddings = embeddings)
    return vetorstore



def main():
    st.title("PDF Question Answering System")

    # Upload PDF
    pdf_file = st.file_uploader("Upload a PDF", type="pdf")
    if pdf_file:
        text_chunks = extract_text_with_references(pdf_file)
    vectorstore = get_vectorstore(text_chunks)
        
if __name__ == "__main__":
    main()
