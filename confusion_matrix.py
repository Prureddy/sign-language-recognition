import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

data_dict = pickle.load(open('./data.pickle', 'rb'))

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier()
model.fit(x_train, y_train)
y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)

y_predict = model.predict(x_test)

confusion_matrix = pd.crosstab(y_test, y_predict, rownames=['Actual'], colnames=['Predicted'])

# Plot the confusion matrix as a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_matrix, annot=True, cmap='Reds')
plt.title('Confusion Matrix')
plt.show()

plt.figure(figsize=(8, 6))
plt.bar(['Accuracy'], [score])
plt.ylim(0, 1)
plt.title('Model Accuracy')
plt.show()
y_predict_proba = model.predict_proba(x_test)
scores = []
thresholds = np.linspace(0, 1, num=100)  # Create thresholds from 0 to 1

for threshold in thresholds:
    t1 = threshold * 100 
    y_predict_thresholded = np.where(y_predict_proba[:, 1] >= t1, 1, 0)
    score = accuracy_score(y_predict_thresholded, y_test)
    scores.append(score)

plt.plot(thresholds, scores)
plt.xlabel('Threshold')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Threshold')
plt.show()






