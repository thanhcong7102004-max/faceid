from flask import Flask, render_template, request, Response
import tensorflow as tf
import numpy as np
import cv2
import os
import pickle
from deepface import DeepFace
from tensorflow.keras.layers import InputLayer
import dlib
from threading import Thread, Lock
import time
import logging

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load mô hình nhận diện khuôn mặt
model = tf.keras.models.load_model("face_recognition_model.h5", custom_objects={'InputLayer': InputLayer})
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Map cảm xúc từ tiếng Anh sang tiếng Việt
EMOTION_MAP = {
    "angry": "Giận dữ",
    "disgust": "Ghê tởm",
    "fear": "Sợ hãi",
    "happy": "Vui vẻ",
    "sad": "Buồn bã",
    "surprise": "Ngạc nhiên",
    "neutral": "Bình thường"
}

# Hàm tiền xử lý hình ảnh: điều chỉnh độ sáng và tương phản
def preprocess_image(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.equalizeHist(v)
    hsv = cv2.merge((h, s, v))
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    alpha = 1.3
    beta = 10
    image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return image

# Hàm nhận diện danh tính
def recognize_face(image, threshold=0.3):
    image = preprocess_image(image)
    img = cv2.resize(image, (100, 100))
    img = np.expand_dims(img, axis=0) / 255.0
    predictions = model.predict(img, verbose=0)
    predicted_label = np.argmax(predictions)
    confidence = np.max(predictions)
    
    logging.info(f"Confidence: {confidence:.4f}")
    
    if confidence < threshold:
        return "Không xác định", confidence
    
    sorted_probs = np.sort(predictions[0])[::-1]
    if len(sorted_probs) > 1 and (sorted_probs[0] - sorted_probs[1]) < 0.01:
        return "Không xác định", confidence
    
    return label_encoder.inverse_transform([predicted_label])[0], confidence

# Hàm nhận diện cảm xúc, giới tính, tuổi
def recognize_deepface_features(image):
    try:
        result = DeepFace.analyze(image, actions=['emotion', 'age', 'gender'], enforce_detection=False)
        emotion = EMOTION_MAP.get(result[0]['dominant_emotion'], "Không xác định")
        age = result[0]['age']
        gender = "Nam" if result[0]['dominant_gender'] == "Man" else "Nữ"
        return emotion, gender, age
    except:
        return "Không xác định", "Không xác định", "Không xác định"

# Nhận diện hướng khuôn mặt
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def recognize_head_pose(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    if len(faces) == 0:
        return "Không xác định"

    for face in faces:
        landmarks = predictor(gray, face)
        nose = (landmarks.part(30).x, landmarks.part(30).y)
        chin = (landmarks.part(8).x, landmarks.part(8).y)
        left_eye = (landmarks.part(36).x, landmarks.part(36).y)
        right_eye = (landmarks.part(45).x, landmarks.part(45).y)

        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        angle = np.arctan2(dy, dx) * 180 / np.pi

        if angle < -10:
            return "Nhìn trái"
        elif angle > 10:
            return "Nhìn phải"
        else:
            return "Nhìn thẳng"

    return "Không xác định"

# Route chính (Trang chủ - Đầy đủ thông tin cho ảnh)
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "image" in request.files:
            image = request.files["image"]
            image_path = os.path.join(UPLOAD_FOLDER, image.filename)
            image.save(image_path)

            img = cv2.imread(image_path)

            name, confidence = recognize_face(img)
            emotion, gender, age = recognize_deepface_features(img)
            head_pose = recognize_head_pose(img)

            return render_template(
                "index.html",
                name=name,
                confidence=f"{confidence:.2%}",
                emotion=emotion,
                gender=gender,
                age=age,
                head_pose=head_pose,
                image_path=image_path,
            )

    return render_template("index.html")

# Nhận diện từ Webcam (Chỉ danh tính)
def webcam_stream():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    cap.set(cv2.CAP_PROP_FPS, 15)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    
    last_result = {"name": "Không xác định", "confidence": 0.0}
    lock = Lock()
    last_update = 0
    update_interval = 5.0

    def process_face(face_roi):
        nonlocal last_result, last_update
        name, confidence = recognize_face(face_roi)
        with lock:
            last_result = {"name": name, "confidence": confidence}
        last_update = time.time()

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.01, minNeighbors=3, minSize=(20, 20))

        current_time = time.time()
        if current_time - last_update > update_interval and len(faces) > 0:
            for (x, y, w, h) in faces:
                logging.info(f"Face size (webcam): {w}x{h}")
                face_roi = frame[y:y+h, x:x+w]
                Thread(target=process_face, args=(face_roi,)).start()
                break

        with lock:
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                text = f"{last_result['name']} ({last_result['confidence']:.2%})"
                cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        _, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.route("/webcam")
def webcam():
    return Response(webcam_stream(), mimetype="multipart/x-mixed-replace; boundary=frame")

# Nhận diện từ Video (Chỉ danh tính, giữ danh tính 3 giây)
@app.route("/video", methods=["POST"])
def video():
    if "video" not in request.files:
        return "No file uploaded", 400
    
    video = request.files["video"]
    video_path = os.path.join(UPLOAD_FOLDER, video.filename)
    video.save(video_path)

    cap = cv2.VideoCapture(video_path)
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    if fps == 0:
        fps = 20  

    output_path = os.path.join(UPLOAD_FOLDER, "output_" + video.filename)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    frame_count = 0
    skip_frames = 2
    last_recognition = {"name": "Không xác định", "confidence": 0.0, "last_updated": 0}

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1
        if frame_count % skip_frames != 0:
            out.write(frame)
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.01, minNeighbors=1, minSize=(20, 20))

        current_time = time.time()
        for (x, y, w, h) in faces:
            logging.info(f"Face size (video): {w}x{h}")
            face_roi = frame[y:y+h, x:x+w]
            new_name, new_confidence = recognize_face(face_roi)
            
            # Cập nhật danh tính nếu đã qua 3 giây
            if current_time - last_recognition["last_updated"] >= 3.0:
                last_recognition = {"name": new_name, "confidence": new_confidence, "last_updated": current_time}
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            text = f"{last_recognition['name']} ({last_recognition['confidence']:.2%})"
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            break  # Chỉ xử lý khuôn mặt đầu tiên

        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    if os.path.getsize(output_path) == 0:
        return "Lỗi: Video đầu ra bị trống", 500

    return render_template("index.html", video_processed=True)

# Chạy Flask Server
if __name__ == "__main__":
    app.run(debug=True)