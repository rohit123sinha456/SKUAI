import zipfile
import requests
import os
from dotenv import load_dotenv
load_dotenv()

folder_path = 'exported_data'
params = {
    "exportType": "COCO"
}
headers = {
    "Authorization": "Token " + os.getenv('LABEL_STUDIO_USER_TOKEN')
}

def export_annotation(client_name,client_id):
    url = f"{os.getenv('LABELSTUDIOURL')}/{client_id}/export"

    if not os.path.exists(os.path.join(".",folder_path,client_name)):
        os.makedirs(os.path.join(".",folder_path,client_name))

    response = requests.get(url, headers=headers, params=params)

    if response.status_code == 200:
        try:
            with open(os.path.join(".",folder_path,client_name,client_name+".zip"), "wb") as f:
                f.write(response.content)
            print("Download completed successfully.")
            try:
                zip_file_path = os.path.join(".",folder_path,client_name,client_name+".zip")
                extract_to = os.path.join(".",folder_path,client_name)
                with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_to)
                    print("Successfully Extracted the .zip file")
                os.remove(zip_file_path)
                print("Deleted the zip File")
            except Exception as e:
                print("Error in zip file extraction"+ e)
                return
        except Exception as e:
            print("Error in downloading file" + e)
            return
    else:
        print(f"Failed to download the file. Status code: {response.status_code}")

if __name__ == "__main__":
    export_annotation("Client1",1)