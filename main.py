import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split

# Load the dataset from the CSV file
# what we want:
# X = np.array([[size1, shape1, texture1], [size2, shape2, texture2], ...])
# y = np.array([0, 1, 0, ...])

data = np.genfromtxt('Breast_cancer_data.csv', delimiter=',', skip_header=1)

# Split the data into the input features (X) and the labels (y)
X = data[:, :-1]
y = data[:, -1]

# Split the dataset into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Define the model
model = Sequential()
model.add(Dense(10, input_dim=5, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model to the training data
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Evaluate the model on the test data
loss, accuracy = model.evaluate(X_test, y_test)
print('Test loss:', loss)
print('Test accuracy:', accuracy)
