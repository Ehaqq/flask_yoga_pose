from flask import Flask, request, jsonify
import mediapipe as mp
import cv2
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)

# Load Mediapipe model
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# Load your classification model
with open('assets/model_cobadataset.pkl', 'rb') as f:
    model = pickle.load(f)

def decode_image(image_data):
    nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def process_image(image):
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = holistic.process(image_rgb)
        return results

@app.route('/detect', methods=['POST'])
def detect():
    try:
        data = request.json
        image_data = data['image']
        image = decode_image(image_data)

        results = process_image(image)
        
        response = {
            'class': None,
            'probability': None
        }

        if results.pose_landmarks:
            pose = results.pose_landmarks.landmark
            pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())

            # Make Detections
            X = pd.DataFrame([pose_row])
            body_language_class = model.predict(X)[0]
            body_language_prob = model.predict_proba(X)[0]

            response['class'] = body_language_class
            response['probability'] = body_language_prob.tolist() # Convert to list for JSON serialization

        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port= 5000)
