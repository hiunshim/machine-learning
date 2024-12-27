from keras.models import Sequential, load_model as keras_load_model
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.datasets import mnist
import numpy as np

MODEL_PATH = "digit_model.h5"


def train_and_save_model():
    """
    Train a simple CNN on the MNIST dataset and save it to disk.
    """
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalize and reshape data
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    # One-hot encode labels
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # Build a simple CNN model
    model = Sequential(
        [
            Conv2D(
                32,
                kernel_size=(3, 3),
                activation="relu",
                input_shape=(28, 28, 1),
            ),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, kernel_size=(3, 3), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(128, activation="relu"),
            Dense(10, activation="softmax"),
        ]
    )

    # Compile the model
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    # Train the model
    model.fit(
        x_train,
        y_train,
        validation_data=(x_test, y_test),
        epochs=5,
        batch_size=128,
    )

    # Evaluate the model
    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)
    print("Accuracy:", accuracy_score(y_test_classes, y_pred_classes))

    # Save the model
    model.save(MODEL_PATH)
    print("Model saved to", MODEL_PATH)


def load_model():
    """
    Load the trained model from disk.
    """
    return keras_load_model(MODEL_PATH)


def predict_digit(model, image_array):
    """
    Predict the digit from the input image.
    """
    # Make predictions
    predictions = model.predict(image_array)
    predicted_digit = np.argmax(predictions)
    confidence = predictions[0][predicted_digit]

    return predicted_digit, confidence
