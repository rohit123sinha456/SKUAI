from flask import Flask, render_template, request
import json

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('form.html')

@app.route('/submit', methods=['POST'])
def submit():
    project_name = request.form.get("project_name")
    labels = request.form.get("labels")
    labels = json.loads(labels)
    print(type(labels))
    return f"Form Data: {project_name}<br>Data Types: {labels}"

if __name__ == '__main__':
    app.run(debug=True)
