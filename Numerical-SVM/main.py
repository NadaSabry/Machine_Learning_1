import tensorflow.keras as ks
from keras.models import Sequential
from keras.layers import Dense, Dropout
import numpy as np  # library that contains mathematical tools
import matplotlib.pyplot as plt  # library that help us plot nice charts
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd  # best library to import and manage datasets
from sklearn.preprocessing import LabelEncoder  # encoding categorical data -> class from sklearn
from sklearn.model_selection import train_test_split  # splitting the dataset into training and testing
from mlxtend.plotting import plot_decision_regions
import seaborn as sn

# just show all Columns in print
##########################################################
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
dataset = pd.read_csv('TravelInsurancePrediction.csv')  # importing the dataset
###########################################################


X = dataset.iloc[:, 1:9].values  # our input(features) equals all rows and columns from 1 to 9
y = dataset.iloc[:, 9].values  # our output equals all rows and the last column only

# we fit to the second column
labelEncoder_X1 = LabelEncoder()
X[:, 1] = labelEncoder_X1.fit_transform(X[:, 1])

labelEncoder_X2 = LabelEncoder()
X[:, 2] = labelEncoder_X2.fit_transform(X[:, 2])
X[:, 6] = labelEncoder_X2.fit_transform(X[:, 6])
X[:, 7] = labelEncoder_X2.fit_transform(X[:, 7])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
# splitting the training dataset into training and validating = 0.34*0.3 = 0.102
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.34, random_state=0)

# features scaling

sc_X = StandardScaler()  # object from the class to scale x matrix
X = sc_X.fit_transform(X)
X_train = sc_X.fit_transform(X_train)
# fit the object and then transform the training set
X_test = sc_X.transform(X_test)
X_val = sc_X.transform(X_val)

sc_X = StandardScaler()  # object from the class to scale x matrix
X_train = sc_X.fit_transform(X_train)  # fit the object and then transform the training set
X_test = sc_X.transform(X_test)
X_val = sc_X.transform(X_val)

# end preprocessing

# model  WX+b
model = Sequential()
model.add(Dense(8, activation='relu'))  # hidden layer_1
model.add(Dense(16, activation='relu'))  # hidden layer  0 or 1
model.add(Dense(8, activation='relu'))  # hidden layer  0 or 1
model.add(Dense(1, activation='sigmoid'))
# compile update: W and b
model.compile(loss="mse", optimizer="adam", metrics=['accuracy'])  # mse:'main square error'

checkpoint = ks.callbacks.ModelCheckpoint(filepath="bestWight", verbose=1, save_best_only=True)

# batch_size update after 32 examples with epochs
# verbose display epochs everyTime
# Training
history = model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1, validation_split=0.2,
                    callbacks=[checkpoint])
model.summary()

# Confusion Matrix
# prediction = model.predict(y_test)
# confusion_matrix = pd.crosstab(dataset['y_test'], dataset['prediction'], rownames=['Actual'], colnames=['Predicted'])
# sn.heatmap(confusion_matrix, annot=True)
# plt.show()



print(history.history.keys())
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()

'''# # # Evaluate

print(model.evaluate(X_test, y_test))
print(model.metrics_names)

# # # Predict
# print(model.predict(X_test), Y_test)
# print(Y_test)

# # for i in range(len(labels)):
# #  print(model.predict(X_test)[i][0], Y_test[i])

# # model.predict(np.array([[1.2,2.3]]))

# #round(model.predict(np.array([[1, 1]]))[0][0])

# plot_decision_regions(X_train, np.array(y_train, dtype='int64'), clf=model, zoom_factor=1.)
# plt.show()'''

# print(dataset.info())
# print(X)
# print(X_train[0])
# print(y_train)