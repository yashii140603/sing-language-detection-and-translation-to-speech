import pickle
import warnings
import cv2
import mediapipe as mp
import numpy as np

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf.symbol_database')

# Load the model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize video capture (use 0 for default webcam)
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)

# Define labels
labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25:'Z'}

# Define the expected number of features
expected_features = 84

# Open the text file in append mode
with open('detected_letters.txt', 'a') as file:
    while True:
        data_aux = []
        x_ = []
        y_ = []

        ret, frame = cap.read()
        if not ret:
            break

        H, W, _ = frame.shape

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            # Process only the first hand's landmarks
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x)
                data_aux.append(y)
                x_.append(x)
                y_.append(y)

            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            x2 = int(max(x_) * W) + 10
            y2 = int(max(y_) * H) + 10

            # Ensure the feature vector has the correct length
            if len(data_aux) < expected_features:
                # Pad the data_aux with zeros to reach the expected number of features
                data_aux.extend([0] * (expected_features - len(data_aux)))
            elif len(data_aux) > expected_features:
                # Truncate the data_aux to the expected number of features
                data_aux = data_aux[:expected_features]

            prediction = model.predict([np.asarray(data_aux)])

            predicted_character = labels_dict[int(prediction[0])]

            # Write the detected letter to the file
            file.write(predicted_character + '\n')

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
