import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from mpl_toolkits.mplot3d import Axes3D



x_train = np.mat([[0.0,0.0], [0.0,1.0], [1.0,0.0], [1.0,1.0]])
y_train = np.mat([[1.0], [1.0], [1.0], [0.0]])

def o(x):
    return 1 / (1 + np.exp(-x))

def gresk(x):
    return 1 / (1 + tf.exp(-x))

class NonLinearRegressionModel:
    def __init__(self):
        # Model input
        self.x = tf.placeholder(tf.float32)
        self.y = tf.placeholder(tf.float32)

        # Model variables
        self.W = tf.Variable([[0.0], [0.0]])
        self.b = tf.Variable([[0.0]])

        logits = (tf.matmul(self.x, self.W)) + self.b

        # Predictor
        f = tf.sigmoid(logits)

        # Mean Squared Error
        self.loss = tf.losses.sigmoid_cross_entropy(self.y, logits)


model = NonLinearRegressionModel()

# Training: adjust the model so that its loss is minimized
minimize_operation = tf.train.GradientDescentOptimizer(0.1).minimize(model.loss)

# Create session object for running TensorFlow operations
session = tf.Session()

# Initialize tf.Variable objects
session.run(tf.global_variables_initializer())

for epoch in range(3000):
    session.run(minimize_operation, {model.x: x_train, model.y: y_train})

# Evaluate training accuracy
W, b, loss = session.run([model.W, model.b, model.loss], {model.x: x_train, model.y: y_train})
print("W = %s, b = %s, loss = %s" % (W, b, loss))

fig = plt.figure("Logistic regression: logical operator")

ax = fig.gca(projection='3d')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

x = y = np.arange(0, 1, 0.05)
X, Y = np.meshgrid(x, y)
zs = o(X*W[0][0] + Y*W[1][0] + b)
surface = ax.plot_surface(X, Y, zs)

plt.show()