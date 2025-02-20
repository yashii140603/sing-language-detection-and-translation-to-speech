import os
import pickle
import cv2
import mediapipe as mp

import matplotlib.pyplot as plt

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'

data = []
labels = []

# Check if DATA_DIR exists and is not empty
if not os.path.exists(DATA_DIR) or not os.listdir(DATA_DIR):
    print(f"Error: DATA_DIR '{DATA_DIR}' does not exist or is empty.")
else:
    for dir_ in os.listdir(DATA_DIR):
        class_dir = os.path.join(DATA_DIR, dir_)
        if not os.path.isdir(class_dir):
            continue

        for img_path in os.listdir(class_dir):
            data_aux = []
            img_full_path = os.path.join(class_dir, img_path)
            img = cv2.imread(img_full_path)

            # Check if image is read correctly
            if img is None:
                print(f"Warning: Could not read image '{img_full_path}'. Skipping.")
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            results = hands.process(img_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        data_aux.append(x)
                        data_aux.append(y)

                data.append(data_aux)
                labels.append(dir_)

    # Save data and labels using pickle
    with open('data.pickle', 'wb') as f:
        pickle.dump({'data': data, 'labels': labels}, f)

    print("Data collection and saving completed successfully.")

# Optional: Visualization code (commented out)
'''
plt.figure()
plt.imshow(img_rgb)
plt.show()
'''
