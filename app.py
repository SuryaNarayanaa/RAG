from flask import Flask, render_template, request, jsonify
from TheUltimateModel.pdf_scanners import 
app = Flask(__name__)

# Route for the home page
@app.route('/' ,methods = ['POST'] )
def home():
    return "HOME PAGE"


# Route with a dynamic URL
@app.route('/hello/<name>')
def hello(name):
    return f"Hello, {name}!"

# Route with POST method
@app.route('/submit', methods=['POST'])
def submit():
    data = request.get_json()
    name = data.get('name', 'Unknown')
    return jsonify({"message": f"Data received from {name}!"})

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
