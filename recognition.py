# recognition.py

import os
from deepface import DeepFace
import shutil

def recognize_faces(uploaded_image):
    output_root_folder = "rec_Images"
    if not os.path.exists(output_root_folder):
        os.makedirs(output_root_folder)

    db_path = "d_faces"

    image_name = os.path.splitext(uploaded_image.name)[0]

    output_folder = os.path.join(output_root_folder, image_name)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    with open(os.path.join(output_folder, uploaded_image.name), "wb") as f:
        f.write(uploaded_image.getbuffer())

    img_path = os.path.join(output_folder, uploaded_image.name)

    for model in ["VGG-Face"]:
        result = DeepFace.find(img_path=img_path, db_path=db_path, model_name=model)
        for item in result:
            for image_path in item['identity']:
                image_name = os.path.basename(image_path)
                shutil.copy(image_path, os.path.join(output_folder, image_name))

    output_folder = output_folder.replace("\\", "/")

    return output_folder
