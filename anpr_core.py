import os
import cv2
import time
import logging
import pytesseract
import xml.etree.ElementTree as ET

OUTPUT_FOLDER = "C:\\Users\\Brandon\\personal_code\\CARL\\OUTPUT\\"
INPUT_IMAGES = "C:\\Users\\Brandon\\personal_code\\CARL\\test\\test_group_1\\"
SHOW_IMAGES = True

def load_image(file2load):
    try:
        # Load the image
        image_path = f"{INPUT_IMAGES}{file2load}"
        image = cv2.imread(image_path)

        # Check if the image is loaded successfully and has the correct number of channels (BGR with 3 channels)
        if image is not None and image.shape[-1] == 3:  # Check for 3 channels (BGR)
            print("Image loaded successfully!")
        else:
            print("Failed to load image or invalid number of channels.")
            return None
    except Exception:
        logging.exception("FAILED TO LOAD IMAGE", exc_info=True)
        return None
    else:
        return image


def preprocess_image(_image):
    try:
        # Convert the image to grayscale
        gray_image = cv2.cvtColor(_image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

        if SHOW_IMAGES:
            # Display the original and preprocessed images (optional)
            cv2.imshow("Original Image", _image)
            cv2.imshow("Gray Scaled Image", gray_image)
            cv2.imshow("Blurred Image (Noise Reduction Technique)", blurred_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    except Exception:
        logging.exception("IMAGE PREPROCESSING FAILURE", exc_info=True)
        return None
    else:
        return blurred_image


def draw_roi(image):
    """
    Allows the user to draw a rectangle on the image to specify the region of interest (ROI).
    Returns the ROI rectangle coordinates (x, y, width, height).
    """
    # Display the image and allow the user to draw a rectangle
    clone = image.copy()
    roi_rectangle = cv2.selectROI("Select ROI", clone, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select ROI")  # Close the window after selection

    # Return the ROI rectangle coordinates
    return roi_rectangle


def detect_and_localize(_preproc_image):
    """
    This code snippet applies Canny edge detection to detect edges in the preprocessed image and then
    finds contours around those edges.
    """
    try:
        # Apply Canny edge detection to detect edges
        edges = cv2.Canny(_preproc_image, threshold1=50, threshold2=150)

        # Find contours in the edge-detected image
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw contours on the original image (optional)
        image_with_contours = cv2.drawContours(_preproc_image.copy(), contours, -1, (0, 255, 0), 2)

        if SHOW_IMAGES:
            # Display the original image with contours (optional)
            cv2.imshow("Image with Contours", image_with_contours)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    except Exception:
        logging.exception("DETECT & LOCALIZE FAILURE", exc_info=True)
        return None
    else:
        return contours


def extract_lp_region(_contours, _preprocessed_image, epsilon=0.01):
    try:
        # Sort contours by area in descending order
        contours = sorted(_contours, key=cv2.contourArea, reverse=True)

        # Extract the largest contour as the license plate region
        license_plate_contour = None
        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon * perimeter, True)  # Adjust epsilon value here
            if len(approx) == 4:  # Assuming license plate contours are rectangular
                license_plate_contour = approx
                break

        # Draw the license plate contour on the original image (optional)
        image_with_license_plate = cv2.drawContours(_preprocessed_image.copy(), [license_plate_contour], -1, (0, 255, 0),
                                                    2)

        if SHOW_IMAGES:
            # Display the original image with license plate contour (optional)
            cv2.imshow("Image with License Plate Contour", image_with_license_plate)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    except Exception:
        logging.exception("FAILURE TO EXTRACT LICENSE PLATE REGION", exc_info=True)
        return None
    else:
        return license_plate_contour


def further_processing(_license_plate_contour, _preprocessed_image):
    try:
        # Extract the license plate region based on the contour
        x, y, w, h = cv2.boundingRect(_license_plate_contour)
        license_plate_region = _preprocessed_image[y:y + h, x:x + w]

        # Check license plate region properties (debug)
        print("License Plate Region Shape:", license_plate_region.shape)
        print("License Plate Region Channels:", license_plate_region.shape[-1])

        # Convert the license plate region to BGR format if it's not already
        if license_plate_region.shape[-1] == 1:  # Check if it's a single-channel image
            license_plate_region = cv2.cvtColor(license_plate_region, cv2.COLOR_GRAY2BGR)

        # Convert the license plate region to grayscale if it's still in BGR format
        if license_plate_region.shape[-1] == 3:  # Check if it's a BGR image
            gray_license_plate = cv2.cvtColor(license_plate_region, cv2.COLOR_BGR2GRAY)
        else:
            gray_license_plate = license_plate_region  # Use as is

        # Apply binarization to the license plate region
        _, binary_license_plate = cv2.threshold(gray_license_plate, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        if SHOW_IMAGES:
            # Display the segmented license plate (optional)
            cv2.imshow("Segmented License Plate", binary_license_plate)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # Return the segmented license plate for further use or analysis
        return binary_license_plate
    except Exception:
        logging.exception("FAILURE TO PROCESS IMAGE FURTHER", exc_info=True)
        return None


def apply_ocr_v1(_license_plate_region):
    try:
        # Apply OCR to recognize characters in the license plate region
        ocr_text = pytesseract.image_to_string(_license_plate_region, config='--psm 8')

        # Print the OCR result (optional)
        print("OCR Result:", ocr_text)

        return ocr_text
    except Exception:
        logging.exception("OCR FAILURE", exc_info=True)
        return None


def apply_ocr_v2(_license_plate_region):
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


def post_process_ocr(ocr_text):
    try:
        # Filter OCR result to include only alphanumeric characters
        filtered_text = ''.join(char for char in ocr_text if char.isalnum())

        # Print the filtered OCR result (optional)
        print("Post-Processed OCR Result:", filtered_text)

        return filtered_text
    except Exception:
        logging.exception("POST-PROCESSING FAILURE", exc_info=True)
        return None


def annotate_image(f_name, original_image, license_plate_contour, recognized_text):
    try:
        # Draw bounding box around the license plate region
        x, y, w, h = cv2.boundingRect(license_plate_contour)
        annotated_image = cv2.rectangle(original_image.copy(), (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Overlay recognized text on the annotated image
        cv2.putText(annotated_image, recognized_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        if SHOW_IMAGES:
            # Display the annotated image (optional)
            cv2.imshow("Annotated Image", annotated_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # Save the annotated image to a specified folder location (optional)
        cv2.imwrite(f"OUTPUT/{f_name}_annotated_image.jpg", annotated_image)
        return [x, y, w, h]
    except Exception:
        logging.exception("ANNOTATION FAILURE", exc_info=True)
        return None


def save_annotation_xml(OUTPUT_FOLDER, file_name, annotation_data):
    try:
        # Generate the XML file path
        xml_file_path = os.path.join(OUTPUT_FOLDER, f"{file_name.split('.')[0]}.xml")

        # Create the XML structure
        root = ET.Element("annotation")
        object_elem = ET.SubElement(root, "object")
        for key, value in annotation_data.items():
            if key == "name":
                name_elem = ET.SubElement(object_elem, key)
                name_elem.text = value
            elif key == "license_plate":
                license_plate_elem = ET.SubElement(object_elem, key)
                license_plate_elem.text = value
            elif key in ("xmin", "ymin", "width", "height"):
                bbox_elem = ET.SubElement(object_elem, "bndbox")
                coord_elem = ET.SubElement(bbox_elem, key)
                coord_elem.text = str(value)

        # Create the XML tree and write to the XML file
        tree = ET.ElementTree(root)
        tree.write(xml_file_path)

        logging.info(f"Annotation saved successfully in XML format: {xml_file_path}")
    except Exception:
        logging.exception("FAILED TO SAVE ANNOTATION IN XML FORMAT", exc_info=True)


if __name__ == "__main__":
    file_list = os.listdir(INPUT_IMAGES)
    print(file_list)
    for f in file_list:
        stripped_name = f.split(".png")[0]

        img = load_image(
            file2load=f
        )
        preprocessed_img = preprocess_image(
            _image=img
        )
        contrs = detect_and_localize(
            _preproc_image=preprocessed_img
        )

        # to help with debugging
        # epsilon_values = [0.07, 0.08, 0.09]  # Adjust epsilon values to experiment
        # for epsilon in epsilon_values:
        #     lp_contour = extract_lp_region(contrs, preprocessed_img, epsilon=epsilon)
        # break

        lp_contour = extract_lp_region(
            _contours=contrs,
            _preprocessed_image=preprocessed_img
        )
        segmented_lp = further_processing(
            _license_plate_contour=lp_contour,
            _preprocessed_image=preprocessed_img
        )
        ocr_result = apply_ocr_v2(
            _license_plate_region=segmented_lp
        )
        filtered_result = post_process_ocr(
            ocr_text=ocr_result
        )
        bound_box = annotate_image(
            f_name=stripped_name,
            original_image=img,
            license_plate_contour=lp_contour,
            recognized_text=filtered_result
        )
        annotation_data = {
            "name": "Cars111.png",
            "license_plate": filtered_result,
            "xmin": bound_box[0],
            "ymin": bound_box[1],
            "width": bound_box[2],
            "height": bound_box[3],
        }

        file_name = f"{stripped_name}_annotated_result.jpg"
        save_annotation_xml(OUTPUT_FOLDER, file_name, annotation_data)


