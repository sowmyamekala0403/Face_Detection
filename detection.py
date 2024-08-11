import os
import cv2
import datetime
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.model_zoo import model_zoo
import tempfile
import streamlit as st
import ntpath

class FaceDetector:
    def __init__(self):
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        self.cfg.MODEL.DEVICE = 'cpu'  # Set device to GPU
        self.predictor = DefaultPredictor(self.cfg)

    def detect_faces(self, frame):
        outputs = self.predictor(frame)
        instances = outputs["instances"]
        pred_classes = instances.pred_classes if instances.has("pred_classes") else None
        boxes = instances.pred_boxes if instances.has("pred_boxes") else None

        detected_faces = []
        if pred_classes is not None and boxes is not None:
            for class_idx, box in zip(pred_classes, boxes):
                if class_idx == 0:  # 0 corresponds to 'person' class in COCO dataset
                    box = box.cpu().numpy().astype(int)  # Convert box tensor to numpy array
                    x0, y0, x1, y1 = box[0], box[1], box[2], box[3]
                    detected_faces.append(frame[y0:y1, x0:x1])
        return detected_faces

    def format_timestamp(self, timestamp_ms):
        utc_datetime = datetime.datetime.utcfromtimestamp(timestamp_ms / 1000.0)
        timestamp = utc_datetime.strftime("%H%M%S")
        return timestamp

    def save_faces(self, detected_faces, output_dir, video_name, timestamp_ms, frame_number):
        video_name = os.path.splitext(video_name)[0]
        timestamp = self.format_timestamp(timestamp_ms)

        for i, face in enumerate(detected_faces):
            face_name = f"{video_name}_{timestamp}_{frame_number}_{i}.jpg"
            save_path = os.path.join(output_dir, face_name)
            cv2.imwrite(save_path, face)
            print(f"Saved image: {face_name}")
        return len(detected_faces)

    def add_frame_number(frame, frame_number):
        font = cv2.FONT_HERSHEY_SIMPLEX
        position = (10, 30)
        font_scale = 1
        font_color = (0, 255, 0)  # Green color
        line_type = 2

        cv2.putText(frame, f"Frame: {frame_number}", position, font, font_scale, font_color, line_type)

    def process_video(self, video_path):  
        output_dir = "d_faces"
        os.makedirs(output_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)

        frame_number = 0
        total_detected_images = 0
        frame_skip = 5  

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)

            if frame_number % 6 == 0:
                detected_faces = self.detect_faces(frame)

                count = self.save_faces(detected_faces, output_dir, ntpath.basename(video_path), timestamp_ms, frame_number)
                total_detected_images += count

            frame_number += 1
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number * frame_skip)

        cap.release()
        cv2.destroyAllWindows()

        return total_detected_images

def process_file(file_path):
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_file_path = os.path.join(tmp_dir, os.path.basename(file_path.name))
        with open(tmp_file_path, 'wb') as f:
            f.write(file_path.getvalue())
        face_detector = FaceDetector()
        total_detected_images = face_detector.process_video(tmp_file_path)
        st.write(f"{total_detected_images} images added to the d_faces folder.")

def process_folder(folder_path):
    total_detected_images = 0
    for filename in os.listdir(folder_path):
        if filename.endswith(".mp4") or filename.endswith(".avi"):
            file_path = os.path.join(folder_path, filename)
            face_detector = FaceDetector()
            total_detected_images += face_detector.process_video(file_path)

    st.write(f"{total_detected_images} images added to the d_faces folder.")
    return total_detected_images