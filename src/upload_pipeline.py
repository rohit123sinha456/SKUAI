'''
1. Make sure to download poppler from here: https://github.com/oschwartz10612/poppler-windows/releases/
2. Extract the downloaded folder
3. Place the folder in whichever directory you wish to
4. Navigate to the "bin" folder
5. Copy the path of the "bin" folder and add to PATH variables
'''

import os
import base64
import requests
from pdf2image import convert_from_path
import logging
from utils import get_random_color
from dotenv import load_dotenv

logging.basicConfig(filename="upload-log.log", level=logging.DEBUG, 
                    filemode="w+",format="%(name)s → %(levelname)s: %(message)s",
                    datefmt="%Y-%m-%d %H:%M")
load_dotenv()

api_key = 'Token ' + os.getenv('LABEL_STUDIO_USER_TOKEN')  # Set accordingly
label_studio_url = os.getenv('LABELSTUDIOURL') # Set accordingly | can be found in the url bar of the browser


def project_creation(project_title,project_description,label_config):
    global api_key,label_studio_url
    headers = {
    "Authorization": api_key,
    "Content-Type": "application/json"
    }
    payload = {
        "title": project_title,
        "description": project_description,
        "label_config": label_config
    }
    response = requests.post(label_studio_url, headers=headers, json=payload)
    logging.info(f"Project status code: {response.status_code}")

    if response.status_code == 201:
        print("Creating Project")
        logging.info(f"Project:{project_title} created succesfully")
    else:
        logging.error("Error in project creation")
    res = response.json()
    #print(details)
    id = res['id']
    logging.info(f"Project ID:{id} created succesfully")
    # print(id)
    return(id)



# Function to convert PDF to JPG
def convert_pdf_to_jpg(input_dir, output_dir):
    """
    Converts all .pdf files in the input directory to .jpg images and saves them in the output directory.

    Arguments:
        input_dir (str): Path to the input directory containing PDF files.
        output_dir (str): Path to the output directory where converted JPG images will be saved.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  # Create the output directory if it doesn't exist
    print("Converting Images")

    for filename in os.listdir(input_dir):
        filename = filename.lower()
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(input_dir, filename)  # Get the full path of the PDF file
            output_path = os.path.join(output_dir, f'{os.path.splitext(filename)[0]}.jpg')  # Construct the output image path

            try:
                images = convert_from_path(pdf_path)
                for i, img in enumerate(images):
                    img_path = f'{output_path[:-4]}_{i}.jpg' if i > 0 else output_path  # Add index to output image path if multiple pages
                    img.save(img_path, 'JPEG')

            except Exception as e:
                logging.error(f'Error converting {pdf_path}: {e}')

            else:
                logging.info(f'Successfully converted {pdf_path} to {output_path}')



def upload_image_to_label_studio(image_path, filename, project_id, api_key, label_studio_url):
    print(image_path)
    files=[('filename',(filename,open(image_path,'rb'),'image/jpeg'))]
    payload = {}
    
    # Set up the headers
    headers = {
        "Authorization": api_key
    }

    # Make the API request
    url = f"{label_studio_url}/{project_id}/import?commit_to_project=true" #7/import?commit_to_project=true
    response = requests.request("POST", url, headers=headers, data=payload, files=files)

    # Check if the request was successful
    if response.status_code == 201:
        logging.info(f"Successfully uploaded {os.path.basename(image_path)}")
    else:
        logging.error(f"Failed to upload {os.path.basename(image_path)}. Status code: {response.status_code}")
        logging.error(f"Response: {response.text}")



# Function to iterate through the output directory to upload each individual image
def upload_images_from_folder(folder_path, project_id):
    global api_key,label_studio_url
    print("Uploading Images")
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, filename)
            upload_image_to_label_studio(image_path,filename, project_id, api_key, label_studio_url)


def preprocess_inputs(project_title,project_description,dlabel):
    dlabstr = """"""

    view_open = """ <View>
        <Image name="image" value="$image"/>
        <RectangleLabels name="label" toName="image">"""

    view_close = """ </RectangleLabels>
        </View>"""

    for label in dlabel:
        color = get_random_color()
        dlabstr = dlabstr + f"""<Label value="{label}" background="{color}"/>""" + "\n"

    label_config = view_open + dlabstr + view_close

    return project_title, project_description, label_config

def user_input():
    project_title = input("Enter project title")
    project_description = input("Enter project description")
    
    #getting label details from the user

    dlabstr = """"""
    dlabel = []
    count = int(input("Enter the number of labels"))
    for i in range(count):
        label,color = input("Enter the label and color").split(",")
        dlabel.append((label,color))

    view_open = """ <View>
        <Image name="image" value="$image"/>
        <RectangleLabels name="label" toName="image">"""

    view_close = """ </RectangleLabels>
        </View>"""

    for label,color in dlabel:
        dlabstr = dlabstr + f"""<Label value="{label}" background="{color}"/>""" + "\n"

    label_config = view_open + dlabstr + view_close

    return project_title, project_description, label_config



# if __name__ == "__main__":

    
#     folder_path = r'Purchase_Orders_JPG' # Set accordingly

#     input_dir = r"Purchase_Orders_PDF"
#     output_dir = r"Purchase_Orders_JPG"

#     project_title,project_description,label_config = user_input()

#     project_id = project_creation( project_title,project_description,label_config) #derived from project creation function
#     convert_pdf_to_jpg(input_dir, output_dir)
#     upload_images_from_folder(folder_path, project_id)