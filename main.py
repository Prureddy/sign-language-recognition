import pickle
import numpy as np
import cv2
import mediapipe as mp
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# Load model and label dict
model_dict = pickle.load(open('model.p', 'rb'))
model = model_dict['model']
labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T',
               20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: 'ZERO', 27: '1', 28: '2', 29: '3', 30: '4', 31: '5', 32: '6', 33: '7', 34: '8', 35: '9', 36: 'OK', 37: 'hiii', 38: 'CHECK /PLEASE', 39: 'BANG BANG',
               40: 'CALL ME', 41: 'GOOD JOB/LUCK', 42: 'SHOCKER', 43: 'YOU', 44: 'DISLIKE', 45: 'LOSER', 46: 'HIGH-FIVE', 47: 'HANG LOOSE', 48: 'ROCK', 49: 'LOVE YOU', 50: 'PUNCH YOU', 51: 'SUPER', 52: 'LITTLE'}

app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if 'frame' not in request.files:
        return jsonify({'error': 'No frame provided.'}), 400
    # Read image from POST
    file = request.files['frame'].read()
    npimg = np.frombuffer(file, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    data_aux = []
    x_ = []
    y_ = []

    results = hands.process(frame_rgb)
    predicted_character = ''
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]  # Only first hand
        for lm in hand_landmarks.landmark:
            x_.append(lm.x)
            y_.append(lm.y)
        for lm in hand_landmarks.landmark:
            data_aux.append(lm.x - min(x_))
            data_aux.append(lm.y - min(y_))
        if len(data_aux) == 42:  # Only if features match!
            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict[int(prediction[0])]
    return jsonify({'prediction': predicted_character})

if __name__ == '__main__':
    app.run(debug=True)
