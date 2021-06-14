# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
import numpy as np
from tensorflow.keras import initializers
from tensorflow.python.keras import activations

print(tf.__version__)

fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images / 255.0

test_images = test_images / 255.0              

import mlflow.tensorflow
mlflow.tensorflow.autolog()

a1 = tf.keras.activations.relu
# relu, sigmoid, tanh, softmax, selu

# init = tf.keras.initializers.lecun_uniform
# he_normal(relu), glorot_normal(sigmoid, tanh, softmax), lecun_normal(selu)


model = tf.keras.Sequential([
tf.keras.layers.Flatten(input_shape=(28, 28)),
# tf.keras.layers.Dense(128, activation=a1),
tf.keras.layers.Dense(128, activation=a1),
tf.keras.layers.Dense(10)
])

model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])
with mlflow.start_run():

    model.fit(train_images, train_labels, epochs=5)

    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

    print('test_loss', test_loss)
    print('test_accuracy', test_acc)


    mlflow.tensorflow.mlflow.log_param('activation',a1)
    mlflow.tensorflow.mlflow.log_metric('test_loss', test_loss)
    mlflow.tensorflow.mlflow.log_metric('test_acc', test_acc) 
    # mlflow.tensorflow.mlflow.log_param('initiazlizer',init)  

    # run = mlflow.active_run()
    # print(f"run id: {run.info.run_id}")
    # print(f"active run info: \n {mlflow.get_run(run_id=run.info.run_id)}")