import random
from keras.datasets import mnist
from PIL import Image


def download_random_test_digit(save_path="random_test_digit.png"):
    """
    Downloads a random digit image from the MNIST test dataset and saves it as a PNG file.

    Args:
        save_path (str): Path to save the image (default is 'random_test_digit.png').
    """
    # MNIST train, test
    (_, _), (x_test, y_test) = mnist.load_data()

    random_index = random.randint(0, len(x_test) - 1)
    digit_image = x_test[random_index]
    digit_label = y_test[random_index]

    pil_image = Image.fromarray(digit_image)
    pil_image.save(save_path)

    print(f"Random digit {digit_label} from test set saved as '{save_path}'")


download_random_test_digit()
