import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

mnist = tf.keras.datasets.mnist

# x is the image and y is the classification
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# normalizing scales every value down between 0 and 1
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)


def train_model():
    model = tf.keras.models.Sequential()
    # flatten turns a matrix into a one dimensional array
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))

    # dense layer each node being connected to every node in the next layer (3 layers total)
    # activation function changes the data every time it goes to the next layer
    # soft max makes sure the value is between 0 and 1
    model.add(tf.keras.layers.Dense(128, activation="relu"))
    model.add(tf.keras.layers.Dense(128, activation="relu"))
    model.add(tf.keras.layers.Dense(10, activation="softmax"))

    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    model.fit(x_train, y_train, epochs=3)

    model.save("handwriten.keras")
    return model


def see_results():
    if (os.path.exists("handwriten.keras")):
        model = tf.keras.models.load_model("handwriten.keras")
        loss, accuracy = model.evaluate(x_test, y_test)

        print(loss)
        print(accuracy)
    else:
        train_model()


def test_data():

    model = tf.keras.models.load_model("handwriten.keras")

    image_number = 1
    while os.path.isfile(f"digits/digit{image_number}.png"):
        try:
            img = cv2.imread(f"digits/digit{image_number}.png")[:, :, 0]
            img = np.invert(np.array([img]))
            prediction = model.predict(img)
            # arg max returns the index of the highest value
            print(f"The result is probably: {np.argmax(prediction)}")
            # cmap is color map and binary is black and white
            plt.imshow(img[0], cmap=plt.cm.binary)
            plt.show()
        except:
            print("An exception occurred")
        finally:
            image_number += 1


if __name__ == "__main__":

    run_model = input(
        "What do you want to do? (train / see results / test data): ")

    if run_model == "train":
        train_model()

    elif run_model == "see results":
        see_results()

    elif run_model == "test data":
        test_data()
