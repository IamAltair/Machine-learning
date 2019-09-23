import numpy as np
import tensorflow as tf

batch_size = 1

enc = [
    [1, 0, 0, 0, 0, 0, 0, 0],  # ' '
    [0, 1, 0, 0, 0, 0, 0, 0],  # 'h'
    [0, 0, 1, 0, 0, 0, 0, 0],  # 'e'
    [0, 0, 0, 1, 0, 0, 0, 0],  # 'l'
    [0, 0, 0, 0, 1, 0, 0, 0],  # 'o'
    [0, 0, 0, 0, 0, 1, 0, 0],  # 'w'
    [0, 0, 0, 0, 0, 0, 1, 0],  # 'r'
    [0, 0, 0, 0, 0, 0, 0, 1],  # 'd'
]
encoding_size = np.shape(enc)[1]

index_to_char = [' ', 'h', 'e', 'l', 'o', 'w', 'r', 'd']
char_to_index = dict((char, i) for i, char in enumerate(index_to_char))

x_train = [[[enc[0], enc[1], enc[2], enc[3], enc[3], enc[4], enc[0], enc[5], enc[4], enc[6], enc[3], enc[7], enc[0]]]]  # ' hello world '
y_train = [[[enc[1], enc[2], enc[3], enc[3], enc[4], enc[0], enc[5], enc[4], enc[6], enc[3], enc[7], enc[0], enc[1]]]]  # 'hello world h'

model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(128, input_shape=(None, encoding_size), return_sequences=True))
model.add(tf.keras.layers.Dense(encoding_size, activation='softmax'))

optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.01)

model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=optimizer)


def on_epoch_end(epoch, data):
    if epoch % 10 == 9:
        print("epoch", epoch)
        print("loss", data['loss'])

        # Generate text from the initial text ' h'
        text = ' h'
        for i in range(50):
            x = np.zeros((1, i + 2, encoding_size))
            for t, char in enumerate(text):
                x[0, t, char_to_index[char]] = 1
            y = model.predict(x)[0][-1]
            text += index_to_char[y.argmax()]
        print(text)


model.fit(x_train, y_train, batch_size=batch_size, epochs=500, verbose=False, callbacks=[tf.keras.callbacks.LambdaCallback(on_epoch_end=on_epoch_end)])
