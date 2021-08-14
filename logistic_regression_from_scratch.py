import tensorflow as tf
import numpy as np
from sklearn.datasets import make_classification

def sigmoid(val):
    return 1/(1+tf.math.exp(-val))

feature_amount = 20
steps = 1000
X, y = make_classification(n_samples=100, n_classes=2, n_features=feature_amount)
y = y.reshape(100,1)
X = tf.convert_to_tensor(X)
y = tf.convert_to_tensor(y, dtype=tf.double)
# print(tf.random.normal(shape=[feature_amount,1], dtype=tf.double) * 0.1)
# exit()
weights = tf.Variable(initial_value=tf.random.truncated_normal(shape=[feature_amount,1], dtype=tf.double) * 0.5, trainable=True)
bias = tf.Variable([0], dtype=tf.double)
lr = 1e-1

for _ in range(steps):
    with tf.GradientTape(persistent=True) as tape:
        y_pred = sigmoid(tf.matmul(X, weights) + bias)
        loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y, y_pred, from_logits=True))
    grads = tape.gradient(loss, weights)
    grads_b = tape.gradient(loss, bias)
    weights.assign_sub(grads*lr)
    bias.assign_sub(grads_b*lr)
    print(loss)

def predict(y_pred):
    return [1 if x >= 0.5 else 0 for x in y_pred]

print(predict(y_pred))
print(y)
print(f"error: {sum(y.numpy().ravel()-predict(y_pred))}")
