import cv2
import os
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom import minidom

# Load the image
image_path = '/practice1/test_group1/Cars27.png'
image = cv2.imread(image_path)

# Display the image
cv2.imshow('Image', image)

# Define the initial bounding box
bbox = cv2.selectROI('Image', image, fromCenter=False, showCrosshair=True)

# Create a list to store bounding box coordinates
bounding_boxes = [bbox]

# Create an XML structure for annotations
annotation = Element('annotation')
folder = SubElement(annotation, 'folder')
folder.text = 'images'
filename = SubElement(annotation, 'filename')
filename.text = os.path.basename(image_path)
size = SubElement(annotation, 'size')
width = SubElement(size, 'width')
width.text = str(image.shape[1])
height = SubElement(size, 'height')
height.text = str(image.shape[0])
depth = SubElement(size, 'depth')
depth.text = str(image.shape[2])
segmented = SubElement(annotation, 'segmented')
segmented.text = '0'

while True:
    key = cv2.waitKey(1) & 0xFF

    # Move the bounding box with arrow keys
    if key == ord('w'):
        bbox = (bbox[0], bbox[1] - 1, bbox[2], bbox[3])
    elif key == ord('s'):
        bbox = (bbox[0], bbox[1] + 1, bbox[2], bbox[3])
    elif key == ord('a'):
        bbox = (bbox[0] - 1, bbox[1], bbox[2], bbox[3])
    elif key == ord('d'):
        bbox = (bbox[0] + 1, bbox[1], bbox[2], bbox[3])

    # Resize the bounding box with "+" and "-"
    elif key == ord('+'):
        bbox = (bbox[0], bbox[1], bbox[2] + 1, bbox[3] + 1)
    elif key == ord('-'):
        bbox = (bbox[0], bbox[1], bbox[2] - 1, bbox[3] - 1)

    # Update the image with the adjusted bounding box
    img_with_bbox = image.copy()
    cv2.rectangle(img_with_bbox, (int(bbox[0]), int(bbox[1])), (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])), (0, 255, 0), 2)
    cv2.imshow('Image with Bounding Box', img_with_bbox)

    # Save the adjusted bounding box coordinates if "s" is pressed
    if key == ord('s'):
        bounding_boxes.append(bbox)

    # Exit the loop with "q"
    if key == ord('q'):
        break

# Add each bounding box to the XML structure
for bbox in bounding_boxes:
    object = SubElement(annotation, 'object')
    name = SubElement(object, 'name')
    name.text = 'licence'
    pose = SubElement(object, 'pose')
    pose.text = 'Unspecified'
    truncated = SubElement(object, 'truncated')
    truncated.text = '0'
    occluded = SubElement(object, 'occluded')
    occluded.text = '0'
    difficult = SubElement(object, 'difficult')
    difficult.text = '0'
    bndbox = SubElement(object, 'bndbox')
    xmin = SubElement(bndbox, 'xmin')
    xmin.text = str(int(bbox[0]))
    ymin = SubElement(bndbox, 'ymin')
    ymin.text = str(int(bbox[1]))
    xmax = SubElement(bndbox, 'xmax')
    xmax.text = str(int(bbox[0] + bbox[2]))
    ymax = SubElement(bndbox, 'ymax')
    ymax.text = str(int(bbox[1] + bbox[3]))

# Save the XML structure to a file
xml_string = minidom.parseString(tostring(annotation)).toprettyxml(indent="  ")
with open('annotation.xml', 'w') as f:
    f.write(xml_string)

cv2.destroyAllWindows()
