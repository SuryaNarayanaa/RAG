from flask import Flask, render_template, request, jsonify
from flask_cors import CORS  # Import CORS
from searching import return_formated_text

app = Flask(__name__)

# Initialize CORS
CORS(app)  # Enable CORS for all routes


@app.route('/', methods=['POST'])
def handle_question():
    # Extract the question from the POST request
    data = request.get_json()  # Assuming the request data is in JSON format
    question = data.get('question', '')  # Get the 'question' key from the JSON data

    # Process the question (this is where your logic would go)
    # For now, let's just send back a simple response
    response_text = return_formated_text(question)

    # Send the response back as JSON
    return jsonify({'response': response_text})

if __name__ == '__main__':
    app.run(debug=True)