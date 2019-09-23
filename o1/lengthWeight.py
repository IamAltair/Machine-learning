import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

data = np.loadtxt(open("length_weight.csv", "rb"), delimiter=",", skiprows=1)

print(data)

x_data = [[row[0]] for row in data]
y_data = [[row[1]] for row in data]

x_train = np.mat(x_data)
y_train = np.mat(y_data)

ax.plot(x_train, y_train, 'o', label='$(\\hat x^{(i)},\\hat y^{(i)})$')
ax.set_xlabel('x')
ax.set_ylabel('y')


class LinearRegressionModel:
    def __init__(self, W, b):
        self.W = W
        self.b = b

    # Predictor
    def f(self, x):
        return x * self.W + self.b

    # Mean Squared Error
    def loss(self, x, y):
        return np.mean(np.square(self.f(x) - y))


model = LinearRegressionModel(np.mat([[0.23956688]]), np.mat([[-8.609315]]))

x = np.mat([[np.min(x_train)], [np.max(x_train)]])
ax.plot(x, model.f(x), label='$y = f(x) = xW+b$')

print('loss:', model.loss(x_train, y_train))

ax.legend()
plt.show()
