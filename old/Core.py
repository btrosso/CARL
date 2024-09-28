# Core.py
import os
import sys
import glob
import logging

from ImageProcessor import ImageProcessor
from OcrProcessor import OcrProcessor
from AnnotationGenerator import AnnotationGenerator

# Set up logging to a file and optionally to the console
logging.basicConfig(
    filename='.\\OUTPUT\\core.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('core.log'), logging.StreamHandler()]
)


class MainApp:
    def __init__(self, _show_image=False):
        self.image_processor = ImageProcessor(_show_image)
        self.ocr_processor = OcrProcessor()
        self.annotation_generator = AnnotationGenerator(_show_image)

    def process_image(self, file_path):
        stripped_name = file_path.split("\\")[-1].split(".png")[0]

        img = self.image_processor.load_image(
            image_path=file_path
        )
        preprocessed_img = self.image_processor.preprocess_image(
            _image=img
        )
        contrs = self.image_processor.detect_and_localize(
            _preproc_image=preprocessed_img
        )

        # to help with debugging
        # epsilon_values = [0.07, 0.08, 0.09]  # Adjust epsilon values to experiment
        # for epsilon in epsilon_values:
        #     lp_contour = extract_lp_region(contrs, preprocessed_img, epsilon=epsilon)
        # break

        lp_contour = self.image_processor.extract_lp_region(
            _contours=contrs,
            _preprocessed_image=preprocessed_img
        )
        segmented_lp = self.image_processor.further_processing(
            _license_plate_contour=lp_contour,
            _preprocessed_image=preprocessed_img
        )
        ocr_result = self.ocr_processor.apply_ocr(
            _license_plate_region=segmented_lp
        )
        filtered_result = self.ocr_processor.post_process_ocr(
            ocr_text=ocr_result
        )
        bound_box = self.annotation_generator.annotate_image(
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
        self.annotation_generator.save_annotation_xml(
            output_folder=".\\OUTPUT\\",
            file_name=file_name,
            annotation_data=annotation_data
        )

    def run_anpr(self, input_folder):
        # file_list = os.listdir(input_folder)
        file_list = glob.glob(input_folder + "*.png")
        print(file_list)
        for f in file_list:
            self.process_image(f)



if __name__ == "__main__":
    try:
        # show_images = sys.argv[1]
        show_images = False
    except Exception:
        show_images = False

    app = MainApp(
        _show_image=show_images
    )
    app.run_anpr(".\\IMAGES\\")
