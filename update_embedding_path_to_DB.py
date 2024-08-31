import psycopg2

def update_document_paths(chat_id, pdf_folder_path, embedding_pkl_path, embedding_index_path):
    conn = psycopg2.connect(database="BookGPT", user="postgres", password="Ntsn03062005@")
    cursor = conn.cursor()

    # SQL statement to update the document paths
    update_query = """
    INSERT INTO embedding_paths (chat_id, pdf_folder, embedding_pkl, embedding_index)
    VALUES (%s, %s, %s, %s);
    """
    
    cursor.execute(update_query, (chat_id,pdf_folder_path, embedding_pkl_path, embedding_index_path ))
    conn.commit()
    
    cursor.close()
    conn.close()

def get_document_paths(chat_id):
    conn = psycopg2.connect(database="BookGPT", user="postgres", password="Ntsn03062005@")
    cursor = conn.cursor()

    # SQL statement to retrieve the document paths
    select_query = """
    SELECT pdf_folder, embedding_pkl, embedding_index
    FROM embedding_paths
    WHERE chat_id = %s;
    """
    
    cursor.execute(select_query, (chat_id,))
    result = cursor.fetchone()
    
    cursor.close()
    conn.close()
    
    if result:
        pdf_folder, embedding_pkl, embedding_index = result
        return {
            'pdf_folder_path': pdf_folder,
            'embedding_pkl_path': embedding_pkl,
            'embedding_index_path': embedding_index
        }
    else:
        return None
