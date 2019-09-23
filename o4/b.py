import numpy as np
import tensorflow as tf

batch_size = 1

enc = [
    [1, 0, 0, 0, 0, 0, 0, 0, 0],  # ' '
    [0, 1, 0, 0, 0, 0, 0, 0, 0],  # 'r'
    [0, 0, 1, 0, 0, 0, 0, 0, 0],  # 'a'
    [0, 0, 0, 1, 0, 0, 0, 0, 0],  # 't'
    [0, 0, 0, 0, 1, 0, 0, 0, 0],  # 'b'
    [0, 0, 0, 0, 0, 1, 0, 0, 0],  # 'i'
    [0, 0, 0, 0, 0, 0, 1, 0, 0],  # '\U0001F400'
    [0, 0, 0, 0, 0, 0, 0, 1, 0],  # '\U0001F407'
    [0, 0, 0, 0, 0, 0, 0, 0, 1],  # '\U0001F987'
]



encoding_size = np.shape(enc)[1]

index_to_char = [' ', 'r', 'a', 't', 'b', 'i', '🐀', '🐇', '🦇']
char_to_index = dict((char, i) for i, char in enumerate(index_to_char))

x_train = [[[enc[1], enc[2], enc[3], enc[0]], [enc[1], enc[2], enc[4], enc[4]], [enc[4], enc[2], enc[3], enc[0]]]]  # 'rat, rabb and bat '
y_train = [[[enc[6]], [enc[7]], [enc[8]]]]  # 'rat emoji'

model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(128, input_shape=(None, encoding_size), return_sequences=True))
model.add(tf.keras.layers.Dense(encoding_size, activation='softmax'))

optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)

model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=optimizer)


def on_epoch_end(epoch, data):
    if epoch % 10 == 9:
        print("epoch", epoch)
        print("loss", data['loss'])

        # Generate text from the initial text
        text = 'rb '
        result = ''
        x = np.zeros((1, 4, encoding_size))
        for t, char in enumerate(text):
            x[0, t, char_to_index[char]] = 1
        y = model.predict(x)[0][-1]
        result = index_to_char[y.argmax()]
        print(result)
        


model.fit(x_train, y_train, batch_size=batch_size, epochs=200, verbose=False, callbacks=[tf.keras.callbacks.LambdaCallback(on_epoch_end=on_epoch_end)])
