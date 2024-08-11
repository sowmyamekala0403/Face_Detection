
import streamlit as st
from detection import process_file, process_folder
from recognition import recognize_faces
import os

st.title("Visage Vault - The Face Recognizer")

# Option selection
option = st.selectbox("Choose an option", ["Process a single video file", "Process a folder containing videos"])

if option == "Process a single video file":
    file_path = st.file_uploader("Upload a video file", type=["mp4", "avi"])
    if file_path is not None:
        total_detected_images = process_file(file_path)
        st.write(f"Faces detected and saved successfully from video: {file_path.name}")
        
elif option == "Process a folder containing videos":
    folder_path = st.text_input("Enter the path to the folder containing videos")
    if st.button("Process Folder") and folder_path:
        total_detected_images = process_folder(folder_path)
        st.write("All videos processed successfully.")
        st.write("Total detected images:", total_detected_images)

st.title("Face Recognition")

testing_image = st.file_uploader("Upload a testing image", type=["jpg", "jpeg", "png"])
if testing_image is not None:
    output_folder = st.cache_data(recognize_faces)(testing_image)
    output_folder = os.path.abspath(output_folder)  # Get the absolute path
    st.success("Face recognition completed. Images copied successfully.")
    st.write("Output folder: " + output_folder.replace("\\", "/"))
