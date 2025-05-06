# AnnotationGenerator.py

import os
import cv2
import logging
import xml.etree.ElementTree as ET

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class AnnotationGenerator:
    def __init__(self, _show_image=False):
        self.SHOW_IMAGES = _show_image

    def annotate_image(self, f_name, original_image, license_plate_contour, recognized_text):
        try:
            # Draw bounding box around the license plate region
            x, y, w, h = cv2.boundingRect(license_plate_contour)
            annotated_image = cv2.rectangle(original_image.copy(), (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Overlay recognized text on the annotated image
            cv2.putText(annotated_image, recognized_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            if self.SHOW_IMAGES:
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

    def save_annotation_xml(self, output_folder, file_name, annotation_data):
        try:
            # Generate the XML file path
            xml_file_path = os.path.join(output_folder, f"{file_name.split('.')[0]}.xml")

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

            print(f"Annotation saved successfully in XML format: {xml_file_path}")
        except Exception as e:
            print(f"Error saving annotation in XML format: {e}")
