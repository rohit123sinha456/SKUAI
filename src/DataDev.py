import os
import cv2
import layoutparser as lp
from paddleocr import PaddleOCR

class LayoutTextExtractor:
    def __init__(self,  ocr_lang='en', use_ocr=True):
        """
        Initializes the LayoutTextExtractor class.
        
        Parameters:
        model (Detectron2LayoutModel): The layout detection model.
        ocr_lang (str): Language to be used by PaddleOCR. Default is English.
        use_ocr (bool): Whether to use OCR for text extraction. Default is True.
        """
        # self.model = model
        self.use_ocr = use_ocr
        if use_ocr:
            os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
            self.ocr = PaddleOCR(use_angle_cls=True, lang=ocr_lang, show_log=False)
        else:
            self.ocr = None

    def detect_layout(self, img_path):
        """
        Detects layout blocks in the given image.
        
        Parameters:
        img_path (str): Path to the input image.
        
        Returns:
        layout (Layout): Detected layout from the input image.
        """
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found at path: {img_path}")
        layout = self.model.detect(img)
        return layout, img

    def extract_text_from_block(self, img_path, layout, label_map, table_list):
        """
        Extracts text from a specific image block using OCR.
        
        Parameters:
        img (ndarray): The image from which to extract text.
        block (dict): A dictionary containing coordinates of the block.
        
        Returns:
        extracted_data (list): List of extracted text and their coordinates.
        """
        # layout, img = self.detect_layout(img_path)
        img = cv2.imread(img_path)
        
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
        
        extracted_data = []
        
        for data in metadata:
            if data['type'] not in table_list:  # Exclude TableContents (label 5)
                x1, x2, y1, y2 = data['coordinates']['x1'], data['coordinates']['x2'], data['coordinates']['y1'], data['coordinates']['y2']
                x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)

                crop = img[y1:y2, x1:x2]
                crop_height, crop_width = crop.shape[:2]
                target_h, target_w = 600, 800
                resized = cv2.resize(crop, (target_w, target_h), interpolation=cv2.INTER_AREA)

                # Extract text using OCR
                result = self.ocr.ocr(crop)
                words = []
                for line in result:
                    for word_info in line:
                        bbox, (text, confidence) = word_info
                        if confidence > 0.5:  # Only consider text with high confidence
                            x_min, y_min = int(bbox[0][0]), int(bbox[0][1])
                            x_max, y_max = int(bbox[2][0]), int(bbox[2][1])
                            word = {
                                'text': text,
                                'x': x_min,
                                'y': y_min,
                                'w': x_max - x_min,
                                'h': y_max - y_min
                            }
                            words.append(word)

                # Add the extracted words along with their label
                extracted_data.append({
                    "label": label_map.get(data['type'], "Unknown"),
                    "coordinates": data['coordinates'],
                    "words": words
                })
        
        return extracted_data
