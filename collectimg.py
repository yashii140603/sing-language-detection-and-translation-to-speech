import os
import cv2
'''this file is for collect the data'''
DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 26  # this will create 3 folders : folder numbers 0,1 and 2 for storing the dataset
dataset_size = 100

# Default camera index is 0
camera_index = 0
cap = cv2.VideoCapture(camera_index)

#if camera index is not found we display this error message
if not cap.isOpened():
    print(f"Error: Camera with index {camera_index} could not be opened.")
    exit()

for j in range(number_of_classes):
    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print('Collecting data for class {}'.format(j))

    done = False
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        cv2.putText(frame, 'Ready? Press "Q" to start collecting!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    if not ret:
        print(f"Skipping class {j} due to camera read error.")
        continue

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        cv2.imshow('frame', frame)
        cv2.imwrite(os.path.join(class_dir, '{}.jpg'.format(counter)), frame)

        counter += 1
        if cv2.waitKey(25) & 0xFF == ord('q'):
            print("Interrupted by user.")
            break

cap.release()
cv2.destroyAllWindows()

