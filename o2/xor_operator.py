import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from mpl_toolkits.mplot3d import Axes3D
import random



x_train = np.mat([[0.0,0.0], [0.0,1.0], [1.0,0.0], [1.0,1.0]])
y_train = np.mat([[0.0], [1.0], [1.0], [0.0]])

def o(x):
    return 1 / (1 + np.exp(-x))

def gresk(x):
    return 1 / (1 + tf.exp(-x))

def rand():
    return random.uniform(-1, 1)

rand1 = random.uniform(-1, 1)
rand2 = random.uniform(-1, 1)
rand3 = random.uniform(-1, 1)
rand4 = random.uniform(-1, 1)

class NonLinearRegressionModel:
    def __init__(self):
        # Model input
        self.x = tf.placeholder(tf.float32)
        self.y = tf.placeholder(tf.float32)

        # Model variables
        self.W1 = tf.Variable([[rand(), rand()], [rand(), rand()]])
        self.W2 = tf.Variable([[rand()], [rand()]])
        self.b1 = tf.Variable([[rand(), rand()]])
        self.b2 = tf.Variable([[rand()]])

        logits = (tf.matmul(self.x, self.W1)) + self.b1

        h = tf.sigmoid(logits)

        f = tf.matmul(h, self.W2) + self.b2

        # Mean Squared Error
        self.loss = tf.losses.sigmoid_cross_entropy(self.y, f)


model = NonLinearRegressionModel()

# Training: adjust the model so that its loss is minimized
minimize_operation = tf.train.GradientDescentOptimizer(0.5).minimize(model.loss)

# Create session object for running TensorFlow operations
session = tf.Session()

# Initialize tf.Variable objects
session.run(tf.global_variables_initializer())

for epoch in range(3000):
    session.run(minimize_operation, {model.x: x_train, model.y: y_train})

# Evaluate training accuracy
W1, W2, b1, b2, loss = session.run([model.W1, model.W2, model.b1, model.b2, model.loss], {model.x: x_train, model.y: y_train})
print("W1 = %s, W2 = %s, b1 = %s, b2 = %s, loss = %s" % (W1, W2, b1, b2, loss))

# print(rand1)
# print(rand2)
# print(rand3)
# print(rand4)

fig = plt.figure("Logistic regression: logical operator")

ax = fig.gca(projection='3d')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

x = y = np.arange(0, 1, 0.05)
X, Y = np.meshgrid(x, y)

zero = np.zeros(shape=(len(x), len(y)))
for i in range(len(x)):
    for j in range(len(y)):
        h = o(np.matmul([x[i], y[j]], W1) + b1)
        zero[i][j] = o(np.matmul(h, W2) + b2)
surface = ax.plot_surface(X, Y, zero)

plt.show()