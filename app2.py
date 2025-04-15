import streamlit as st
import numpy as np
import cv2
from PIL import Image
from pdf2image import convert_from_bytes
import pytesseract


st.set_page_config(page_title="W-2 Reader", layout="centered")
st.title("📄 W-2 Form Reader")



st.subheader("Upload your W-2 document")
left, center, right = st.columns([1, 5, 1])
with center:
    uploaded_file = st.file_uploader("Support image file (PNG, JPG, JPEG) and PDF file", type=["png", "jpg", "jpeg", "pdf"])


# option 1:
def preprocess_image(image):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    thresh = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return thresh



def find_main_content_box(image):
    rgb_image = image.convert("RGB")
    img = np.array(rgb_image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Binarize and invert
    _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

    # Dilation to connect border lines, but not overkill
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    dilated = cv2.dilate(binary, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours by area (descending)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / h

        # Try to find a box that looks like a form (exclude outer image)
        if 0.9 < aspect_ratio < 2.5 and w > 600 and h > 300:
            cropped = rgb_image.crop((x, y, x + w, y + h))
            return cropped

    return image  # fallback if no good box found



#
def preprocess_for_boxes(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

    # Detect horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    detect_horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)

    # Detect vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    detect_vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=1)

    # Combine both
    grid = cv2.add(detect_horizontal, detect_vertical)

    return grid



# Find boxes in the image
def find_boxes(thresh_img):
    dilated = cv2.dilate(thresh_img, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    st.write(f"✅ Total contours: {len(contours)}")

    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # Avoid outer box and extremely small noise
        if 20 < w < 1000 and 20 < h < 300:
            boxes.append((x, y, w, h))

    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
    return boxes



# 
def extract_text_from_boxes(image, boxes):
    # Convert to NumPy array if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
        
    results = []
    for i, (x, y, w, h) in enumerate(boxes):
        if w > 50 and h > 20:  # Filter out small/noisy boxes
            roi = image[y:y+h, x:x+w]
            text = pytesseract.image_to_string(roi, config="--psm 6")
            results.append({
                "box": (x, y, w, h),
                "text": text.strip()
            })
    return results



def draw_boxes(image, boxes):
    image_np = np.array(image.convert("RGB"))
    for (x, y, w, h) in boxes:
        cv2.rectangle(image_np, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return image_np




# --------- Main App ---------
if uploaded_file:
    # Check if the uploaded file is a PDF
    if uploaded_file.type == "application/pdf":
        # Convert PDF to images
        images = convert_from_bytes(uploaded_file.read())
        st.image(images[0], caption=f'Uploaded PDF "{uploaded_file.name}"', use_container_width=True)
        image = images[0]
    else: # it's an image
        # Read image file
        image = Image.open(uploaded_file)
        st.image(image, caption=f'Uploaded Image "{uploaded_file.name}"', use_container_width=True)

    cropped_img = find_main_content_box(image)
    st.image(cropped_img, caption="Main Content", use_container_width=True)
    
    # Preprocess the image
    grid = preprocess_for_boxes(cropped_img)
    st.image(grid, caption="Grid", use_container_width=True)

    boxes = find_boxes(grid)

    boxed_image = draw_boxes(cropped_img, boxes)
    st.image(boxed_image, caption="Detected Grid Boxes", use_container_width=True)

    ocr_results = extract_text_from_boxes(cropped_img, boxes)
    for i, res in enumerate(ocr_results):
        st.text(f"Box {i+1} [{res['box']}]:\n{res['text']}\n{'-'*40}")

    '''
    # Perform OCR
    # ocr_result = ocr_image(preprocessed_image)
    # st.text_area("OCR Result", ocr_result, height=300)
    '''