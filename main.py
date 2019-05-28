# ----- libraries & functions-------------------------------------------------------
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib 
import numpy
import matplotlib.pyplot as plt
import os
import pandas as pd

# ----- load pima indians dataset --------------------------------------------------
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")

# ----- split into input (X) and output (Y) variables ------------------------------
X = dataset[:,0:8]
Y = dataset[:,8]



# ----- scale features -------------------------------------------------------------
exists = os.path.isfile('scaler.pkl')
if exists:
	scaler = joblib.load('scaler.pkl')
else:
	scaler = StandardScaler().fit(X)
	joblib.dump(scaler, 'scaler.pkl') 

X = scaler.transform(X)


# ----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------
# MLP with automatic validation set
# ----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------

# ----- create model ---------------------------------------------------------------
model1 = Sequential()
model1.add(Dense(12, input_dim=8, activation='relu'))
model1.add(Dense(20, activation='relu'))
model1.add(Dense(1, activation='sigmoid'))

# ----- Compile model --------------------------------------------------------------
model1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# ----- print model ----------------------------------------------------------------
print(model1.summary())

# ----- Fit the model --------------------------------------------------------------
history1 = model1.fit(X, Y, validation_split=0.25, epochs=20, batch_size=10, verbose=0)

# ----- Plot training accuracy and loss --------------------------------------------
f1 = plt.figure(1, figsize=(8, 9))

plt.subplot(211)
plt.plot(history1.history['acc'])
plt.plot(history1.history['val_acc'])
plt.title('Model Accuracy', fontsize=20)
plt.ylabel('Accuracy', fontsize=15)
plt.xlabel('Epoch', fontsize=15)
plt.legend(['Train', 'Test'], loc='lower right')
plt.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)

# summarize history for loss
plt.subplot(212)
plt.plot(history1.history['loss'])
plt.plot(history1.history['val_loss'])
plt.title('Model Loss', fontsize=20)
plt.ylabel('Loss', fontsize=15)
plt.xlabel('Epoch', fontsize=15)
plt.legend(['Train', 'Test'], loc='upper right')
plt.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)

plt.subplots_adjust(
	#top=0,
	#bottom=0.1,
	hspace=0.3,
	top=0.95,
	bottom=0.08,
	right=0.95,
	left=0.1)


#################################################################
#################################################################
#################################################################


# ----- split into 67% for train and 33% for test ----------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)

# create model
model2 = Sequential()
model2.add(Dense(12, input_dim=8, activation='relu'))
model2.add(Dense(20, activation='relu'))
model2.add(Dense(1, activation='sigmoid'))

# Compile model
model2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
history2 = model2.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=20, batch_size=10, verbose=0)

#print result (to be continued...)
#scores2 = model2.evaluate(X_train, y_train, verbose=0)
#print(scores2)

# plot training
f2 = plt.figure(2, figsize=(8, 9))

plt.subplot(211)
plt.plot(history2.history['acc'])
plt.plot(history2.history['val_acc'])
plt.title('Model Accuracy', fontsize=20)
plt.ylabel('Accuracy', fontsize=15)
plt.xlabel('Epoch', fontsize=15)
plt.legend(['Train', 'Test'], loc='lower right')
plt.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
#plt.plot()

# summarize history for loss
plt.subplot(212)
plt.plot(history2.history['loss'])
plt.plot(history2.history['val_loss'])
plt.title('Model Loss', fontsize=20)
plt.ylabel('Loss', fontsize=15)
plt.xlabel('Epoch', fontsize=15)
plt.legend(['Train', 'Test'], loc='upper right')
plt.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)

plt.subplots_adjust(
	#top=0,
	#bottom=0.1,
	hspace=0.3,
	top=0.95,
	bottom=0.08,
	right=0.95,
	left=0.1)



#################################################################
#################################################################
#################################################################
# MLP for Pima Indians Dataset with 10-fold cross validation

# define 10-fold cross validation test harness
kfold = StratifiedKFold(n_splits=10, shuffle=True)

cvscores = []

for train, test in kfold.split(X, Y):
    # create model
	model = Sequential()
	model.add(Dense(12, input_dim=8, activation='relu'))
	model.add(Dense(8, activation='relu'))
	model.add(Dense(1, activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	# Fit the model
	model.fit(X[train], Y[train], epochs=20, batch_size=10, verbose=0)
	# evaluate the model
	scores = model.evaluate(X[test], Y[test], verbose=0)
	print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
	cvscores.append(scores[1] * 100)
print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))


plt.show()


