import cv2
from PIL import Image
import pytesseract


pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
# Open the image file directly
image_path = 'C:/Users/Brandon/personal_code/CARL/practice/test_group1/Cars4.png'
image = Image.open(image_path)

# Convert the image to RGB mode if needed
image = image.convert('RGB')
cv2.imshow("TEST", image)
# Perform OCR on the image to extract text
text = pytesseract.image_to_string(image, lang='eng')

# Print or use the extracted text as needed
print(text)
