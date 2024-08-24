import os
from nbconvert import PythonExporter
import nbformat

def convert_ipynb_to_py(filename):
    # Iterate through all files in the directory
    
        # Check if the file is a Jupyter notebook
        if filename.endswith('.ipynb'):
            ipynb_path = filename
            py_path = os.path.join( os.path.splitext(filename)[0] + '.py')
            
            # Read the notebook content
            with open(ipynb_path, 'r', encoding='utf-8') as f:
                notebook = nbformat.read(f, as_version=4)
            
            # Convert notebook to Python script
            exporter = PythonExporter()
            script, _ = exporter.from_notebook_node(notebook)
            
            # Save the Python script
            with open(py_path, 'w', encoding='utf-8') as f:
                f.write(script)

            print(f"Converted {filename} to {os.path.basename(py_path)}")

# Usage
directory_path = "searching.ipynb"  # Change this to your directory path
convert_ipynb_to_py(directory_path)
