import streamlit as st
import pytesseract
import cv2
from PIL import Image
import numpy as np
from pdf2image import convert_from_bytes
import re


st.set_page_config(page_title="Document OCR Demo", layout="centered")
st.title("📄 Document OCR - W-2 Form Reader")


uploaded_file = st.file_uploader("Upload a W-2 document (image or PDF)", type=["png", "jpg", "jpeg", "pdf"])


def preprocess_image(image):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    thresh = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return thresh


def ocr_image(image):
    custom_config = r'--oem 3 --psm 6'
    return pytesseract.image_to_string(image, config=custom_config)


def visualize_boxes(image):
    data = pytesseract.image_to_data(image, config='--psm 6', output_type=pytesseract.Output.DICT)
    image_copy = image.copy()

    for i in range(len(data['text'])):
        if int(data['conf'][i]) > 60 and data['text'][i].strip():
            (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
            cv2.rectangle(image_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image_copy, data['text'][i], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 1)

    return image_copy


def extract_w2_fields(text):
    lines = text.split('\n')
    fields = {}
    formatted_lines = []

    def find_line_value(keyword, num_format=r'\d{1,6}\.\d{2}', include_label=True):
        for i, line in enumerate(lines):
            if keyword.lower() in line.lower():
                # Get value from the same or next line
                match = re.findall(num_format, line)
                if not match and i + 1 < len(lines):
                    match = re.findall(num_format, lines[i + 1])
                if match:
                    if include_label:
                        formatted_lines.append(f"{keyword.strip()}:")
                    formatted_lines.append(match[0])
                    return match[0]
        return None

    fields["a. Employee's SSN"] = find_line_value("employee's social security number", r'\d{5,9}') or "Not found"
    fields["b. EIN"] = find_line_value("employer identification number (ein)", r'\d{5,9}') or "Not found"
    fields["c. Employer's name, address, and ZIP code"] = find_line_value("employer's name, address, and zip code", r'[A-Za-z ]+') or "Not found"
    fields["d. Control number"] = find_line_value("control number", r'\d{1,6}') or "Not found"
    fields["e. Employee's first name and initial"] = find_line_value("employee's first name and initial", r'[A-Za-z ]+') or "Not found"
    fields["e. Employee's last name"] = find_line_value("last name", r'[A-Za-z ]+') or "Not found"
    fields["f. Employee's address and ZIP code"] = find_line_value("employee's address and zip code", r'[A-Za-z0-9 ]+') or "Not found"
    fields["1. Wages, tips, other compensation"] = find_line_value("wages, tips, other compensation") or "Not found"
    fields["2. Federal income tax withheld"] = find_line_value("federal income tax witheld") or "Not found"
    fields["3. Social security wages"] = find_line_value("social security wages") or "Not found"
    fields["4. Social security tax withheld"] = find_line_value("social security tax withheld") or "Not found"
    fields["5. Medicare wages and tips"] = find_line_value("medicare wages and tips") or "Not found"
    fields["6. Medicare tax withheld"] = find_line_value("medicare tax withheld") or "Not found"
    fields["7. Social security tips"] = find_line_value("social security tips") or "Not found"
    fields["8. Allocated tips"] = find_line_value("allocated tips") or "Not found"
    fields["9. "] = "N/A"
    fields["10. Dependent care benefits"] = find_line_value("dependent care benefits") or "Not found"
    fields["11. Nonqualified Plans"] = find_line_value("nonqualified plans") or "Not found"
    fields["State Wages"] = find_line_value("state wages") or "Not found"
    fields["State Income Tax"] = find_line_value("state income tax") or "Not found"
    fields["Local Wages"] = find_line_value("local wages") or "Not found"
    fields["Local Income Tax"] = find_line_value("local income tax") or "Not found"
    fields["Locality Name"] = find_line_value("locality name", r'[A-Za-z]+', include_label=False) or "Not found"

    return fields, '\n'.join(formatted_lines)


def display_w2_table(fields: dict):
    st.subheader("📊 W-2 Key Information")
    field_labels = {
        "SSN": "Employee’s Social Security Number",
        "EIN": "Employer Identification Number (EIN)",
        "Wages": "Wages, Tips, Other Compensation"
    }

    table_data = {
        "Field": [field_labels.get(k, k) for k in fields.keys()],
        "Value": list(fields.values())
    }

    st.table(table_data)


# ---------- Main App Logic ----------
if uploaded_file:
    file_type = uploaded_file.type

    if file_type == "application/pdf":
        pages = convert_from_bytes(uploaded_file.read())
        st.write(f"PDF with {len(pages)} page(s) detected.")

        for i, page in enumerate(pages):
            st.subheader(f"Page {i+1}")
            processed = preprocess_image(page)
            text = ocr_image(processed)
            boxes = visualize_boxes(cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR))
            fields, formatted_text = extract_w2_fields(text)

            st.image(page, caption=f"Original Page {i+1}", use_container_width=True)
            st.image(boxes, caption="🟩 OCR Bounding Box", use_container_width=True)
            st.subheader("📜 Formatted Extracted Text")
            st.code(formatted_text)
            st.subheader("🔍 Extracted Fields (JSON)")
            st.json(fields)
            display_w2_table(fields)

    else:
        image = Image.open(uploaded_file)
        processed = preprocess_image(image)
        text = ocr_image(processed)
        boxes = visualize_boxes(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))
        fields, formatted_text = extract_w2_fields(text)

        st.image(image, caption=f"Original File: {uploaded_file.name}", use_container_width=True)
        st.image(boxes, caption="🟩 OCR Bounding Box", use_container_width=True)
        st.subheader("📜 Formatted Extracted Text")
        st.code(formatted_text)
        st.subheader("🔍 Extracted Fields (JSON)")
        st.json(fields)
        display_w2_table(fields)