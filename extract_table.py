import os
import pytesseract
import pdf2image
import csv
from pytesseract import Output
from pdf2image import convert_from_path

# Define paths
pdf_directory = 'data/'
table_output_folder = 'output/tables/'

# Ensure the output directory exists
os.makedirs(table_output_folder, exist_ok=True)

# Define the path to Tesseract executable if it's not in your PATH
# Uncomment and modify the following line if needed
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def extract_table_from_image(image):
    # Use pytesseract to extract text and layout information
    data = pytesseract.image_to_data(image, output_type=Output.DICT, lang='eng')
    
    # Extract text from data and construct table rows
    num_boxes = len(data['text'])
    table_data = []
    current_row = []
    for i in range(num_boxes):
        if int(data['conf'][i]) > 0:  # Only consider positive confidence values
            current_row.append(data['text'][i])
            if data['line_num'][i] != data['line_num'][i - 1] if i > 0 else True:
                table_data.append(current_row)
                current_row = []
    if current_row:
        table_data.append(current_row)
    
    return table_data

def process_pdf(pdf_path):
    images = convert_from_path(pdf_path)
    table_count = 0
    for page_num, image in enumerate(images):
        # Extract tables from the image
        table_data = extract_table_from_image(image)
        
        # Save tables to CSV
        if table_data:
            csv_filename = os.path.join(table_output_folder, f"{os.path.splitext(os.path.basename(pdf_path))[0]}_page{page_num + 1}_table.csv")
            with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(table_data)
       
