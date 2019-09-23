import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Load observations from the mnist dataset. The observations are divided into a training set and a test set
# Load data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

image_resolution = 28
image_size = image_resolution*image_resolution
number_of_labels = 10

# Flatten images
x_train = x_train.reshape(x_train.shape[0], image_size)
x_test = x_test.reshape(x_test.shape[0], image_size)

# Convert labels to one-hot
y_train = tf.keras.utils.to_categorical(y_train, number_of_labels)
y_test = tf.keras.utils.to_categorical(y_test, number_of_labels)


class LinearRegressionModel:
    def __init__(self):
        # Model input
        self.x = tf.placeholder(tf.float32, [None, image_resolution * image_resolution])
        self.y = tf.placeholder(tf.float32)
        self.labels = tf.placeholder(tf.float32, [None, number_of_labels])

        # Model variables
        self.W = tf.Variable(tf.zeros([image_resolution * image_resolution, number_of_labels]))
        self.b = tf.Variable(tf.zeros([number_of_labels]))

        # Logits
        logits = tf.matmul(self.x, self.W) + self.b

        # Predictor
        f = tf.math.softmax(logits)

        # Mean Squared Error
        self.loss = tf.losses.softmax_cross_entropy(self.y, logits)

        # Accuracy
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(f, 1), tf.argmax(self.y, 1)), tf.float32))

model = LinearRegressionModel()

# Training: adjust the model so that its loss is minimized
minimize_operation = tf.train.GradientDescentOptimizer(0.01).minimize(model.loss)

# Create session object for running TensorFlow operations
session = tf.Session()

# Initialize tf.Variable objects
session.run(tf.global_variables_initializer())

for epoch in range(500):
    session.run(minimize_operation, {model.x: x_train, model.y: y_train})
    accuracy = session.run([model.accuracy], {model.x: x_train, model.y: y_train})
    if epoch % 100 == 0:
        print(accuracy[0])

    if accuracy[0] > 0.9:
        print("Epochs done: ", epoch)
        break

# Evaluate training accuracy
W, b, accuracy, loss = session.run([model.W, model.b, model.accuracy, model.loss], {model.x: x_train, model.y: y_train})
print("W = %s, b = %s, accuracy = %s, loss = %s" % (W, b, accuracy*100, loss))

W = np.transpose(W)
W = W.reshape(W.shape[0], 28, 28)

for i in range(0, 10):
    plt.imsave("W%s.png" % (i), W[i, :])

plt.show()
