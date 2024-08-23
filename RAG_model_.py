import streamlit as st
from dotenv import load_dotenv
import fitz  
from langchain.text_splitter import SentenceTransformersTokenTextSplitter
import spacy
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        with fitz.open(stream=pdf.read(), filetype="pdf") as doc:
            for i, page in enumerate(doc):
                text += page.get_text()

    return text

def semantic_chunking(text, chunk_size=1500, chunk_overlap=150):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sent in doc.sents:
        sent_length = len(sent.text.split())
        if current_length + sent_length > chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = current_chunk[-chunk_overlap:]  # Apply overlap
            current_length = sum(len(s.split()) for s in current_chunk)
        
        current_chunk.append(sent.text)
        current_length += sent_length
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

def get_text_cunks(raw_txt):

    text_splitter = SentenceTransformersTokenTextSplitter()

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceInstructEmbeddings(model_name = "hkunlp/instructor-xl")
    vetorstore = FAISS.from_texts(texts = text_chunks , embeddings = embeddings)
    return vetorstore



def main():

    load_dotenv()






    st.set_page_config(page_title="RAG MODEl", page_icon= ":books:")

    st.header("Chat with PDF's :books:")

    st.text_input("Ask a Question ? ")
    
    with st.sidebar:
        st.subheader("Your Documents :")
        pdf_docs =st.file_uploader("Upload pdf and click on process",accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):

                if pdf_docs:
                    raw_text = get_pdf_text(pdf_docs)
                    st.success("Processing complete!")
                    st.write("Extracted Text:", raw_text)
                else:
                    st.error("Please upload at least one PDF file.")

                text_chunks = get_text_cunks(raw_text)
                vectorstore = get_vectorstore(text_chunks)












if __name__ == "__main__":
    main()