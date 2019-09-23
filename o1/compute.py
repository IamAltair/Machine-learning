import numpy as np

import tensorflow as tf

data = np.loadtxt(open("length_weight.csv", "rb"), delimiter=",", skiprows=1)

print(data)

x_data = [[row[0]] for row in data]
y_data = [[row[1]] for row in data]

x_train = np.mat(x_data)
y_train = np.mat(y_data)

class LinearRegressionModel:
    def __init__(self):
        # Model input
        self.x = tf.placeholder(tf.float32)
        self.y = tf.placeholder(tf.float32)

        # Model variables
        self.W = tf.Variable([[720.0]])
        self.b = tf.Variable([[0.0]])

        # Predictor
        f = tf.matmul(self.x, self.W) + self.b

        # Mean Squared Error
        self.loss = tf.reduce_mean(tf.square(f - self.y))


model = LinearRegressionModel()

# Training: adjust the model so that its loss is minimized
minimize_operation = tf.train.GradientDescentOptimizer(0.0001).minimize(model.loss)

# Create session object for running TensorFlow operations
session = tf.Session()

# Initialize tf.Variable objects
session.run(tf.global_variables_initializer())

for epoch in range(1000):
    session.run(minimize_operation, {model.x: x_train, model.y: y_train})

# Evaluate training accuracy
W, b, loss = session.run([model.W, model.b, model.loss], {model.x: x_train, model.y: y_train})
print("W = %s, b = %s, loss = %s" % (W, b, loss))

session.close()
