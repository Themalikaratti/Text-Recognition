import streamlit as st
import cv2
import numpy as np
import pytesseract

# Set Tesseract path
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

def perform_ocr(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Perform OCR on the grayscale image
    image_char = pytesseract.image_to_string(gray_image)
    image_boxes = pytesseract.image_to_boxes(gray_image)
    return image_char, image_boxes

def main():
    st.title("Text Recognition")

    # Add option to choose between uploading an image or using the camera
    option = st.radio("Select Input Option", ("Upload Image", "Use Camera"))

    if option == "Upload Image":
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

        if uploaded_file is not None:
            # Read the image from the uploaded file
            file_bytes = uploaded_file.getvalue()
            nparr = np.frombuffer(file_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # Perform OCR on the uploaded image
            image_char, image_boxes = perform_ocr(image)

            # Display the original image and the detected text
            st.image(image, caption='Uploaded Image', use_column_width=True)
            st.write("Detected Text:", image_char)
    else:  # Use Camera option
        st.write("Camera feed")
        # Initialize the camera
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            st.error("Error: Unable to open camera.")
            return

        # Read a frame from the camera
        ret, frame = cap.read()

        if not ret:
            st.error("Error: Failed to capture frame.")
            return

        # Perform OCR on the camera frame
        image_char, image_boxes = perform_ocr(frame)

        # Display the camera frame with detected text
        st.image(frame, channels="BGR", caption="Camera Feed")
        st.write("Detected Text:", image_char)

        # Check if the user has pressed the "Stop" button
        if st.button("Start Camera_" + str(id(cap))):
            cap.release()
            return

if __name__ == "__main__":
    main()
