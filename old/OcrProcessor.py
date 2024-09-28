# OcrProcessor.py

import pytesseract
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class OcrProcessor:
    def __init__(self):
        pass

    def apply_ocr(self, _license_plate_region):
        whitelist = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
        try:
            # Apply OCR to recognize characters in the license plate region
            ocr_text = pytesseract.image_to_string(_license_plate_region, config='--psm 7')

            # Filter OCR result based on the whitelist
            filtered_text = ''.join(char for char in ocr_text if char.upper() in whitelist)

            # Print the filtered OCR result (optional)
            print("Filtered OCR Result:", filtered_text)

            return filtered_text
        except Exception:
            logging.exception("OCR FAILURE", exc_info=True)
            return None

    def post_process_ocr(self, ocr_text):
        try:
            # Filter OCR result to include only alphanumeric characters
            filtered_text = ''.join(char for char in ocr_text if char.isalnum())

            # Print the filtered OCR result (optional)
            print("Post-Processed OCR Result:", filtered_text)

            return filtered_text
        except Exception:
            logging.exception("POST-PROCESSING FAILURE", exc_info=True)
            return None
