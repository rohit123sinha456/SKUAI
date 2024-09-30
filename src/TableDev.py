import os
import cv2
import tempfile
import pandas as pd
import layoutparser as lp
from img2table.ocr import PaddleOCR
from img2table.document import Image

class TableExtractor:
    def __init__(self,  ocr_lang='en', score_threshold=0.6):
        """
        Initializes the TableExtractor class.
        
        Parameters:
        model_path (str): Path to the pre-trained layout model.
        config_path (str): Path to the configuration file.
        ocr_lang (str): Language for the OCR engine. Default is English.
        score_threshold (float): Threshold score for layout detection. Default is 0.6.
        """
        # self.model = model
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        self.ocr = PaddleOCR(lang=ocr_lang)
    
    def detect_layout(self, img_path):
        """
        Detects layout blocks in the given image.
        
        Parameters:
        img_path (str): Path to the input image.
        
        Returns:
        layout (Layout): Detected layout from the input image.
        img (ndarray): Loaded image.
        """
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found at path: {img_path}")
        layout = self.model.detect(img)
        return layout, img

    def extract_table_from_block(self, img_path, layout, label_map, table_list):
        """
        Extracts the table from the detected layout blocks in the image.
        
        Parameters:
        img_path (str): Path to the input image.
        label_map (dict): Mapping of layout block types to their corresponding labels.
        
        Returns:
        table_data (pd.DataFrame): Extracted table as a pandas DataFrame.
        """
        # layout, img = self.detect_layout(img_path)
        img = cv2.imread(img_path)
        
        # Create metadata
        metadata = []
        for idx, block in enumerate(layout):
            label_info = {
                "id": idx,
                "coordinates": {
                    "x1": block.coordinates[0],
                    "y1": block.coordinates[1],
                    "x2": block.coordinates[2],
                    "y2": block.coordinates[3]
                },
                "type": block.type,
                "score": block.score
            }
            metadata.append(label_info)

        extracted_table_list = []

        for data in metadata:
            if data['type'] in table_list:  # label '5' corresponds to 'TableContents'
                x1 = int(data['coordinates']['x1'])
                x2 = int(data['coordinates']['x2'])
                y1 = int(data['coordinates']['y1'])
                y2 = int(data['coordinates']['y2'])

                # Crop the detected table region
                crop = img[y1:y2, x1:x2]
                crop_height, crop_width = crop.shape[:2]
                target_h, target_w = 600, 800
                resized = cv2.resize(crop, (target_w, target_h), interpolation=cv2.INTER_AREA)
                cv2.imwrite(r"D:/Rohit/GarmentsSKU/SKUAI/src/aiapp/table.jpg", crop)
                # Extract tabular data using img2table
                cropped_image = Image(r"D:/Rohit/GarmentsSKU/SKUAI/src/aiapp/table.jpg", detect_rotation=True)
                
                # # Extract the table into a DataFrame
                # extracted_table = cropped_image.extract_tables(
                #     ocr=self.ocr,
                #     implicit_rows=True,
                #     implicit_columns=False,
                #     borderless_tables=True,
                #     min_confidence=50
                # )

                jsonDf = None

                with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=True) as temp_table:
                    
                    cropped_image.to_xlsx(dest=temp_table,
                        ocr=self.ocr,
                        implicit_rows=True,
                        implicit_columns=False,
                        borderless_tables=True,
                        min_confidence=50)

                    df = pd.read_excel(temp_table, header=None)
                    jsonDf = df.to_json(orient='records')

                extracted_table_list.append({
                    "label": label_map.get(data['type'], "Unknown"),
                    "coordinates": data['coordinates'],
                    "table": jsonDf
                })
            
        return extracted_table_list  # Return empty DataFrame if no 'TableContents' block is detected
    
        #         cropped_image.to_xlsx(dest=r"C:/Users/datacore/Downloads/STRAUSS/table.xlsx",
        #             ocr=self.ocr,
        #             implicit_rows=True,
        #             implicit_columns=False,
        #             borderless_tables=True,
        #             min_confidence=50)
                
        #         if extracted_table:
        #             return extracted_table
        #         else:
        #             return pd.DataFrame()  # Return an empty DataFrame if no table is found
        # return pd.DataFrame()  # Return empty DataFrame if no 'TableContents' block is detected
