#!/usr/bin/env python
# coding: utf-8

# In[30]:


import pdf2image
from pdf2image import convert_from_path
import pytesseract
import os

from concurrent.futures import ThreadPoolExecutor



# In[31]:


DATA_FOLDER_DIRECTORY = "..\data"
OUTPUT_DIRECTORY = ".\output"
EXTRACTED_TEXT_DIRECTORY = "./output/text"



# In[32]:


import pytesseract
from PIL import Image
import cv2
import numpy as np
import os
import fitz 
import camelot
import pdf2image
import io


def capture_images_from_pdf(pdf_directory):
    images = pdf2image.convert_from_path(pdf_path=pdf_directory)
    return images

def extract_images_from_page(image_path):
   
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1=50, threshold2=150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    extracted_images = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 50 and h > 50:
            cropped_img = img[y:y+h, x:x+w]
            _, img_encoded = cv2.imencode('.jpg', cropped_img)
            img_bytes = img_encoded.tobytes()
            extracted_images.append((x, y, w, h, img_bytes))
    return extracted_images






def text_extractor(image_path):
   
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image, lang='eng')
    return text


# In[33]:


from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

def describe_image(image_path):
    """
    Generate a textual description of an image using a pre-trained image captioning model.

    :param image_path: Path to the image file.
    :return: A textual description of the image.
    """
    # Load the pre-trained model and processor
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    # Load and process the image
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    # Generate caption
    out = model.generate(**inputs)
    description = processor.decode(out[0], skip_special_tokens=True)

    return description

# Example usage



# In[34]:


import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import string

# Download NLTK data (if not already installed)
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('punkt')


def preprocess_text(text):
    """
    Preprocess the text by removing stop words, punctuation, and applying stemming.
    
    :param text: Raw text to be processed.
    :return: Cleaned and preprocessed text.
    """
    # Initialize the stop words and stemmer
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()

    # Tokenize the text
    words = word_tokenize(text)

    # Convert to lowercase and remove punctuation
    words = [word.lower() for word in words if word.isalnum()]

    # Remove stop words and apply stemming
    words = [stemmer.stem(word) for word in words if word not in stop_words]

    # Reassemble the text
    clean_text = ' '.join(words)

    return clean_text


# In[35]:


def extract_data_from_image(image_path):
 
    text = text_extractor(image_path)
    extracted_images = extract_images_from_page(image_path)
    
    
    combined_text = preprocess_text(text)
    for (x, y, w, h, img_bytes) in extracted_images:
        img_text = pytesseract.image_to_string(Image.open(io.BytesIO(img_bytes)), lang='eng')
        img_description = describe_image(io.BytesIO(img_bytes))
        combined_text += f"\n\n[Image at ({x},{y}) with size ({w}x{h})]: {img_text}\nDescription: {img_description}"
    
    
    
    return combined_text



def save_text_to_file(book , pgno,chunkno , text):
    file_name = f"{book}_{pgno}_{chunkno}.txt"
    file_path = os.path.join(OUTPUT_DIRECTORY, "text", file_name)
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(text)

def process_book_page(book_path, book_name, page_num, chunk_number):
    image_path = f"temp_image_{chunk_number}_{page_num}.jpg"
    images_from_book = capture_images_from_pdf(book_path)
    image = images_from_book[page_num]
    image.save(image_path, 'JPEG')
    
    combined_text = extract_data_from_image(image_path)
    save_text_to_file(book_name, page_num + 1, chunk_number, combined_text)
    os.remove(image_path)


def extract_data_from_directory(data_directory , chat_id):
    data_root_directory = data_directory
    data_root_directory = os.path.abspath(data_root_directory)
    books_directory = os.listdir(data_root_directory)
    chunk_number = 1
    for book in books_directory:
        print(f"{book}----------")
        print(data_root_directory)
        book_path = os.path.join(data_root_directory, book)
        if os.path.isfile(book_path) and book_path.endswith(".pdf"):
            text_of_the_entire_book = ""
            images_from_book = capture_images_from_pdf(book_path)
            for page_num, image in enumerate(images_from_book):

                image_path = f"temp_image_{chunk_number}_{page_num}.jpg"
                image.save(image_path, 'JPEG')
                text_of_the_entire_book += extract_data_from_image(image_path) 
                text_of_the_entire_book+= "\n\n"
                os.remove(image_path) 
            print(f"{book}---------completed")


            book_file_name = os.path.basename(book_path) 
            book_name = os.path.splitext(book_file_name)[0]
            DIRECTORY_FOR_BOOK_TEX_CHATID = os.path.join(EXTRACTED_TEXT_DIRECTORY, chat_id)
            os.makedirs(DIRECTORY_FOR_BOOK_TEX_CHATID ,exist_ok=True)
            DIRECTORY_FOR_BOOK_TEXT = os.path.join(DIRECTORY_FOR_BOOK_TEX_CHATID, book_name + ".txt")
            if not os.path.exists(DIRECTORY_FOR_BOOK_TEX_CHATID):
                os.makedirs(DIRECTORY_FOR_BOOK_TEX_CHATID)
            with open(DIRECTORY_FOR_BOOK_TEXT, 'w', encoding='utf-8') as file:
                file.write(text_of_the_entire_book)
            print(f"SAVED SUCCESSFUL IN THE PATH {DIRECTORY_FOR_BOOK_TEXT}")
                # save_text_to_file(os.path.splitext(book)[0], page_num + 1, chunk_number, combined_text)
                # chunk_number+=1
    return DIRECTORY_FOR_BOOK_TEX_CHATID
                


# In[36]:




# In[ ]:





# In[ ]:




