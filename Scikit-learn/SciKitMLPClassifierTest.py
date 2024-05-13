print(__doc__)


from sklearn import datasets, metrics

#Using same dataset from tensorflow
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
from sklearn.neural_network import MLPClassifier
import skimage
import datetime

# The digits dataset
scikit = datasets.load_digits()
digits = keras.datasets.mnist.load_data()#datasets.load_digits()

ds = datetime.datetime.now()

#Digits[0] is a array contains 60000 28*28 image and the result
images = digits[0][0]
labels = digits[0][1]

#Digits[1] is a array contains 10000 28*28 image and the result
testimg = digits[1][0]
testlabel = digits[1][1]



images_and_labels = list(zip(images, labels))#list(zip(digits.images, digits.target))
for index in range(4):
    plt.subplot(2, 4, index + 1)
    plt.axis('off')
    plt.imshow(images_and_labels[index][0], cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Training: %i' % images_and_labels[index][1])


# To apply a classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
n_samples = len(images_and_labels)
data = images.reshape((n_samples, -1))


# Create a classifier: a Multi-layer Perceptron classifier
classifier = MLPClassifier(solver='adam', learning_rate_init = 0.001,  random_state=1, verbose=True)

# We learn the digits on the first half of the digits
classifier.fit(data, labels)

# Now predict the value of the digit on the second half:
expected = testlabel
predictedImg = testimg.reshape((len(testimg), -1))
predicted = classifier.predict(predictedImg)

print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

images_and_predictions = list(zip(testimg, testlabel))
for index in range(4):
    plt.subplot(2, 4, index + 5)
    plt.axis('off')
    plt.imshow(images_and_predictions[index][0], cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Prediction: %i' % images_and_predictions[index][1])


de = datetime.datetime.now()

print('Total Second:%i' % (de-ds).total_seconds())

plt.show()