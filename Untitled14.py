#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

plt.figure()
plt.imshow(x_test[1])
plt.colorbar()
plt.grid(False)
plt.show()

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])

prediction1 = model(x_train[:1]).numpy()
print(prediction1)
predictions1 = model(x_test[:1]).numpy()
print(predictions1)
unmodel = tf.nn.softmax(predictions1).numpy()
print(unmodel[0])


# In[1]:


loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
loss = loss_fn(y_train[:1], prediction1).numpy()
model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test, verbose=2)
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
print(probability_model(x_test[:5]))


# In[ ]:


import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

model=tf.keras.Sequential([keras.layers.Dense(units=1,input_shape=[1])])

for layer in model.layers:
    weights = layer.get_weights()
weights = np.array([[0.6519221]], dtype=np.float32)

model.compile(optimizer='sgd', loss='mse')
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=np.float32)

plt.scatter(xs, ys)

model.fit(xs, ys, epochs=500)
print(model.predict([10.0]))

