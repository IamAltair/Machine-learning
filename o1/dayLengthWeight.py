from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# fig, ax = plt.subplots()

data = np.loadtxt(open("day_length_weight.csv", "rb"), delimiter=",", skiprows=1)

print(data)

x_data = [[row[1]] for row in data]
y_data = [[row[2]] for row in data]
z_data = [[row[0]] for row in data]

x_train = np.mat(x_data)
y_train = np.mat(y_data)
z_train = np.mat(z_data)



# ax.plot(x_train, y_train, 'o', label='$(\\hat x^{(i)},\\hat y^{(i)})$')
# ax.set_xlabel('x')
# ax.set_ylabel('y')


class LinearRegressionModel:
    def __init__(self):
        # Model input
        self.x = tf.placeholder(tf.float32)
        self.y = tf.placeholder(tf.float32)
        self.z = tf.placeholder(tf.float32)

        # Model variables
        self.W = tf.Variable([[0.0]])
        self.b = tf.Variable([[0.0]])
        self.M = tf.Variable([[0.0]])

        # Predictor
        f = tf.matmul(self.x, self.W) + tf.matmul(self.y, self.M) + self.b

        # Mean Squared Error
        self.loss = tf.reduce_mean(tf.square(f - self.z))


model = LinearRegressionModel()

# Training: adjust the model so that its loss is minimized
minimize_operation = tf.train.GradientDescentOptimizer(0.0001).minimize(model.loss)

# Create session object for running TensorFlow operations
session = tf.Session()

# Initialize tf.Variable objects
session.run(tf.global_variables_initializer())

for epoch in range(3000):
    session.run(minimize_operation, {model.x: x_train, model.y: y_train, model.z: z_train})

# Evaluate training accuracy
W, b, M, loss = session.run([model.W, model.b, model.M, model.loss], {model.x: x_train, model.y: y_train, model.z: z_train})
print("W = %s, b = %s, M = %s, loss = %s" % (W, b, M, loss))

model_x = np.arange(30, 130, 1)
model_y = np.arange(0, 50, 1)
model_x, model_y = np.meshgrid(model_x, model_y)
model_z = model_x*W[0][0] + model_y*M[0][0] + b

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x_train, y_train, z_train, color='red')

ax1 = fig.gca(projection='3d')
surface = ax1.plot_surface(model_x, model_y, model_z)


# model = LinearRegressionModel(np.mat([[W]]), np.mat([[b]]), np.mat([[V]]))

# x = np.mat([[np.min(x_train)], [np.max(x_train)]])
# ax.plot(x, model.f(x), label='$y = f(x) = xW+b$')

plt.show()
session.close()