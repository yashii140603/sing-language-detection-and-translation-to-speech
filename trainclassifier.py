import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Load the data dictionary
data_dict = pickle.load(open('./data.pickle', 'rb'))

# Inspect the data to understand its structure
for i, item in enumerate(data_dict['data']):
    print(f"Item {i}: {item}")
    if i >= 10:  # Print only the first 10 items for inspection
        break

# Find the maximum length of the sequences in data
max_length = max(len(item) for item in data_dict['data'])

# Pad sequences to the maximum length with zeros or other appropriate values
padded_data = [np.pad(item, (0, max_length - len(item)), 'constant') for item in data_dict['data']]

# Convert to NumPy array
data = np.asarray(padded_data)

# Convert labels to NumPy array
labels = np.asarray(data_dict['labels'])

# Check class distribution
(unique, counts) = np.unique(labels, return_counts=True)
class_distribution = dict(zip(unique, counts))
print("Class distribution before handling:", class_distribution)

# Ensure each class has at least two samples
for label in unique:
    if class_distribution[label] < 2:
        indices = np.where(labels == label)[0]
        data = np.delete(data, indices, axis=0)
        labels = np.delete(labels, indices, axis=0)

# Check class distribution again
(unique, counts) = np.unique(labels, return_counts=True)
class_distribution = dict(zip(unique, counts))
print("Class distribution after handling:", class_distribution)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Initialize and train the RandomForestClassifier
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Predict the labels for the test set
y_predict = model.predict(x_test)

# Calculate the accuracy score
score = accuracy_score(y_predict, y_test)
print('{}% of samples were classified correctly!'.format(score * 100))

# Save the trained model using pickle
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
