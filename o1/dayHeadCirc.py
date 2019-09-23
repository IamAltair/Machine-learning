import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

fig, ax = plt.subplots()

data = np.loadtxt(open("day_head_circumference.csv", "rb"), delimiter=",", skiprows=1)

print(data)

x_data = [[row[0]] for row in data]
y_data = [[row[1]] for row in data]

x_train = np.mat(x_data)
y_train = np.mat(y_data)

ax.plot(x_train, y_train, 'o', label='$(\\hat x^{(i)},\\hat y^{(i)})$')
ax.set_xlabel('x')
ax.set_ylabel('y')

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
        self.W = tf.Variable([[0.0]])
        self.b = tf.Variable([[0.0]])

        # Predictor
        f = 20 * gresk((self.x*self.W) + self.b) + 31

        # Mean Squared Error
        self.loss = tf.reduce_mean(tf.square(f - self.y))


model = NonLinearRegressionModel()

# Training: adjust the model so that its loss is minimized
minimize_operation = tf.train.GradientDescentOptimizer(0.0000001).minimize(model.loss)

# Create session object for running TensorFlow operations
session = tf.Session()

# Initialize tf.Variable objects
session.run(tf.global_variables_initializer())

for epoch in range(3000):
    session.run(minimize_operation, {model.x: x_train, model.y: y_train})

# Evaluate training accuracy
W, b, loss = session.run([model.W, model.b, model.loss], {model.x: x_train, model.y: y_train})
print("W = %s, b = %s, loss = %s" % (W, b, loss))

x = np.mat([[np.min(x_train)], [np.max(x_train)]])
y = np.mat([[np.min(y_train)], [np.max(y_train)]])
data_x = np.arange(1, 2000)
data_y = []
for i in range(1, 2000):
    data_y.append((20*o(i * W[0][0] + b[0][0])) + 31)

print(data_y)

plt.plot(data_x, data_y)

plt.show()

session.close()
