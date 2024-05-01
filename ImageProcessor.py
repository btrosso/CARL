# ImageProcessor.py

import cv2
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class ImageProcessor:
    def __init__(self, _show_image=False):
        self.SHOW_IMAGES = _show_image

    def load_image(self, image_path):
        try:
            # Load the image
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

    def preprocess_image(self, _image):
        try:
            # Convert the image to grayscale
            gray_image = cv2.cvtColor(_image, cv2.COLOR_BGR2GRAY)

            # Apply Gaussian blur to reduce noise
            blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

            if self.SHOW_IMAGES:
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

    def detect_and_localize(self, _preproc_image):
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

            if self.SHOW_IMAGES:
                # Display the original image with contours (optional)
                cv2.imshow("Image with Contours", image_with_contours)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        except Exception:
            logging.exception("DETECT & LOCALIZE FAILURE", exc_info=True)
            return None
        else:
            return contours

    def extract_lp_region(self, _contours, _preprocessed_image, epsilon=0.02):
        """
            extract license plate region
            """
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
            image_with_license_plate = cv2.drawContours(_preprocessed_image.copy(), [license_plate_contour], -1,
                                                        (0, 255, 0),
                                                        2)

            if self.SHOW_IMAGES:
                # Display the original image with license plate contour (optional)
                cv2.imshow("Image with License Plate Contour", image_with_license_plate)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        except Exception:
            logging.exception("FAILURE TO EXTRACT LICENSE PLATE REGION", exc_info=True)
            return None
        else:
            return license_plate_contour

    def further_processing(self, _license_plate_contour, _preprocessed_image):
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

            if self.SHOW_IMAGES:
                # Display the segmented license plate (optional)
                cv2.imshow("Segmented License Plate", binary_license_plate)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            # Return the segmented license plate for further use or analysis
            return binary_license_plate
        except Exception:
            logging.exception("FAILURE TO PROCESS IMAGE FURTHER", exc_info=True)
            return None