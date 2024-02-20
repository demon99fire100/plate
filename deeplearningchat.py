import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import easyocr
from tensorflow.keras.models import load_model

model = load_model('/home/deb/code/static/models/object_detection.h5')

def object_detection(path, filename):
    # Read image
    image = load_img(path, target_size=(224, 224))
    image_arr_224 = img_to_array(image) / 255.0
    h, w, d = image_arr_224.shape
    test_arr = image_arr_224.reshape(1, 224, 224, 3)

    # Make predictions
    coords = model.predict(test_arr)
    denorm = np.array([w, w, h, h])
    coords = (coords * denorm).astype(np.int32)

    # Draw bounding box
    xmin, xmax, ymin, ymax = coords[0]
    pt1 = (xmin, ymin)
    pt2 = (xmax, ymax)
    print(pt1, pt2)

    # Draw bounding box on the original image
    image = cv2.cvtColor(img_to_array(load_img(path)), cv2.COLOR_RGB2BGR)
    cv2.rectangle(image, pt1, pt2, (0, 255, 0), 3)
    cv2.imwrite('/home/deb/code/static/predict/{}'.format(filename), image)
    return coords

def OCR(path, filename):
    img = cv2.imread(path)
    coords = object_detection(path, filename)
    xmin, xmax, ymin, ymax = coords[0]
    roi = img[ymin:ymax, xmin:xmax]

    # Convert ROI to grayscale
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Apply additional preprocessing steps if needed
    # Example: roi_gray = cv2.threshold(roi_gray, 128, 255, cv2.THRESH_BINARY)[1]

    reader = easyocr.Reader(['en'])
    text = reader.readtext(roi_gray, detail=0)
    print(text)
    return text
