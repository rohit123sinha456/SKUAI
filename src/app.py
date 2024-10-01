from flask import Flask, jsonify,request
from flask_cors import CORS
from upload_pipeline import project_creation,convert_pdf_to_jpg,upload_images_from_folder,preprocess_inputs,logging
from model_training import train_and_store_model,load_and_infer_vanilla,get_training_status_of_model,store_model
from export_pipeline import export_annotation
import os
import sqlite3
from flask import g
import tempfile
import shutil
from dotenv import load_dotenv
import threading
from threadcommunication import lock,shared_dict
from constants import folder_path,latest_model_path
import json
from vectordb import insert_categories,get_collection,classify_input_text
load_dotenv()


path = os.environ['PATH']
APIMODE = os.getenv("APIMODE", "development")  
poppler_path = ""
if APIMODE == "production":
    poppler_path = "/usr/local/bin"  # Path where Poppler is installed in the container
else:
    poppler_path = "D:/Rohit/GarmentsSKU/SKUAI/src/poppler-24.07.0/Library/bin" # Arbitrary path for non-production mode

os.environ["PATH"] =  poppler_path + ';' + path

DATABASE = 'application.db'
PROJECT_PATH = os.getenv("PROJECTPATH","D:/Rohit/GarmentsSKU/SKU-Garments-")
milvuscollection = get_collection()
app = Flask(__name__)
CORS(app)

def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
    return db

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

@app.route('/')
def home():
    return jsonify({"message": "Welcome to the Flask app with CORS enabled!"})

@app.route('/create_project', methods=['POST'])
def createProject():
    data = request.get_json()

    title = data.get('title')
    description = data.get('description')
    items = data.get('items')
    project_title,project_desc,project_labels = preprocess_inputs(title,description,items)
    logging.info(project_title)
    logging.info(project_desc)
    logging.info(project_labels)
    project_id = project_creation( project_title,project_desc,project_labels)
    # Process the data as needed
    return jsonify({"projectID":project_id})

@app.route('/create_categories', methods=['POST'])
def create_categories():
    categories = request.get_json()
    result = insert_categories(milvuscollection,categories)
    if result == None:
        return jsonify({"message": "Successfully Inserted Data"})
    else:
        return jsonify({"error":result}), 400


@app.route('/get_word_categories', methods=['POST'])
def get_word_categories():
    data = request.get_json()
    text = data.get('text')
    result = classify_input_text(milvuscollection,text)
    return jsonify({"message" : result})


@app.route('/upload_data', methods=['POST'])
def uploadData():
    project_name = request.form.get("project_name")
    project_id = request.form.get("project_id")
    output_dir = os.path.join(PROJECT_PATH,"raw_data",project_name)
    print(output_dir,project_id)
    print(request.files)
    if 'files' not in request.files or project_id is None or output_dir is None :
        return 'No file part or project ID or Output Directory'
    

    files = request.files.getlist('files')
    if len(files) > 200:
        return 'Too many files. Maximum 10 allowed.'

    for file in files:
        extension = file.filename.split(".")[1].lower()
        if(extension != "pdf"):
            return jsonify({"message": "Files not all pdf"})
    
    try:
        temp_dir = tempfile.mkdtemp()
        for file in files:
            file_path = os.path.join(temp_dir, file.filename)
            file.save(file_path)

        print("Converting Images to Images")
        convert_pdf_to_jpg(temp_dir, output_dir)
        print("Uploading images to Label Studio")
        upload_images_from_folder(output_dir, project_id)
        print("Cleaning temporary upload folder")
        # print(os.environ["PATH"])
        print(temp_dir)
        shutil.rmtree(temp_dir)
    except Exception as e:
        return jsonify({"error": e}) , 400

    return jsonify({"message": "Files uploaded successfully"})

@app.route('/upload_weights', methods=['POST'])
def uploadWeights():
    data = request.get_json()
    project_name = data.get('project_name')
    try:
        store_model(project_name)
    except Exception as e:
        return jsonify({"error": str(e)})
    return jsonify({"message": "Model Loaded successfully"})

@app.route('/export_data', methods=['POST'])
def exportData():
    data = request.get_json()
    project_name = data.get('project_name')
    project_id = data.get('project_id')
    try:
        export_annotation(project_name,project_id)
    except Exception as e:
        return jsonify({"Error": str(e)}) , 400
    return jsonify({"message": "Annotations Exported successfully"})

@app.route('/train_model', methods=['POST'])
def trainModel():
    data = request.get_json()
    project_name = data.get('project_name')
    if not os.path.exists(os.path.join(".",folder_path,project_name)):
        return jsonify({"message": "Export data before training"})
    with lock:
        shared_dict[project_name] = False
    thread = threading.Thread(target=train_and_store_model, args=(project_name,))
    thread.start()
    # train_and_store_model(project_name)
    return jsonify({"message": "Model is training"})

@app.route('/get_training_status/<project_name>', methods=['GET'])
def get_training_status(project_name):
    try:
        status = get_training_status_of_model(str(project_name))
    except Exception as e:
        return jsonify({"Error": str(e)}) , 400
    return jsonify({"Training complete status": status})

@app.route('/infer', methods=['POST'])
def inferModel():
    key = None
    project_name = request.form.get("project_name")
    labels = request.form.get("labels")
    labels = json.loads(labels)
    if 'file' not in request.files:
        return 'No file provided'
    
    file = request.files['file']
    filename = file.filename.split(".")[1].lower()
    if(filename == "pdf"):
        key = "PDF"
    elif(filename in ["jpeg","jpg","png"]):
        key = "IMG"
    else:
        return "Not valid file"
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, file.filename)
    file.save(file_path)
    try:
        result = load_and_infer_vanilla(project_name,file_path,key,labels)
        print(type(labels))
        print(labels)
    except Exception as e:
        return jsonify({"Error": str(e)}) , 400
    return jsonify({"message": result})


if __name__ == '__main__':
    folders = ["exported_data","latest_models","raw_data"]
    for folder in folders:
        if not os.path.exists(os.path.join(".",folder)):
            os.makedirs(os.path.join(".",folder))
    
    app.run(debug=True,host='0.0.0.0',port=12001)
