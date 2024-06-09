from flask import Flask, session, render_template, request, send_file, Response, current_app, make_response, jsonify
from flask_socketio import SocketIO, emit

import cv2
import numpy as np
import pandas as pd
import mediapipe as mp 
import pickle
import base64

# SocketIo
app = Flask(__name__)
socketio = SocketIO(app)

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('message')
def handle_message(message):
    print('Received message:', message)


# Load Model
with open('assets\model_cobadataset.pkl', 'rb') as f:
    model = pickle.load(f)

# Mediapipe    
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

@socketio.on('image')
def handle_image(image_data):
    try:
        body_language_prob = 0.0
        body_language_class = "none"
        image_data_bytes = base64.b64decode(image_data)
        image_array = np.frombuffer(image_data_bytes, dtype=np.uint8)
        decoded_image = cv2.imdecode(image_array, cv2.IMREAD_UNCHANGED)
        
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            image = cv2.cvtColor(decoded_image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (340, 180), interpolation=cv2.INTER_LINEAR)
            
            # Make Detections
            results = holistic.process(image)
            print('Detection results obtained')

            # Recolor image back to BGR for rendering
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                      mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))

            try:
                if results.pose_landmarks is not None:
                    # Extract Pose landmarks
                    pose = results.pose_landmarks.landmark
                    pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
                    
                    # Concatenate rows
                    row = pose_row
                    
                    # Make Detections
                    X = pd.DataFrame([row])
                    body_language_class = model.predict(X)[0]
                    body_language_prob = model.predict_proba(X)[0]
                    
                    print(f'class: {body_language_class}, prob: {body_language_prob}')

                else:
                    print('No pose landmarks detected')
                
            except Exception as e:
                print('Error during prediction:', e)
    
        processed_image_bytes = cv2.imencode('.jpg', image)[1].tobytes()
        processed_image_data = base64.b64encode(processed_image_bytes).decode('utf-8')
        prob_float = float(np.max(body_language_prob))
        prob = str(prob_float)
        print(prob)

        emit('response', {"imageData": processed_image_data,"pose_class": body_language_class, "prob": prob})
        # emit('response', processed_image_data, 'class': body_langua)

    except Exception as e:
        print('Error processing image:', e)




if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0')
