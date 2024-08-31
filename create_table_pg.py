import psycopg2

# Connect to the PostgreSQL database
conn = psycopg2.connect(
    database="BookGPT",
    user="postgres",
    password="Ntsn03062005@"
)

# Create a cursor object
cursor = conn.cursor()

# SQL statement to create the table
create_table_query = """
CREATE TABLE IF NOT EXISTS embedding_paths (
    chat_id TEXT PRIMARY KEY,
    pdf_folder TEXT,
    embedding_pkl TEXT,
    embedding_index TEXT
);
"""

# Execute the SQL statement
cursor.execute(create_table_query)

# Commit the changes
conn.commit()

# Close the cursor and connection
cursor.close()
conn.close()

print("Table 'documents' created successfully in the 'BookGPT' database.")
