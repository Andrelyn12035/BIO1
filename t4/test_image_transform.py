import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from scipy.signal import convolve2d



IMAGE_DIR = "./imagenes/equipo 7/Original_Medica5R.png"
IMAGE_OBJECT = plt.imread(IMAGE_DIR)

# Parámetros de prueba
ALPHA = 25035.47881845735
DELTA = 0.3490477025094941

IMAGE_DIR = "./imagenes/equipo 7/RSNA_Mammography_1058522855.png"
IMAGE_OBJECT = plt.imread(IMAGE_DIR)

START_DATETIME = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def ensure_uint8(image):
    image = np.asarray(image)
    if image.dtype == np.uint8:
        return image
    if np.issubdtype(image.dtype, np.floating):
        if image.max() <= 1.0:
            image = image * 255.0
        return np.clip(image, 0, 255).astype(np.uint8)
    return np.clip(image, 0, 255).astype(np.uint8)

def to_grayscale(image):
    image = ensure_uint8(image)
    if image.ndim == 2:
        return image
    if image.ndim == 3:
        gray = (
            0.299 * image[:, :, 0] +
            0.587 * image[:, :, 1] +
            0.114 * image[:, :, 2]
        )
        return np.clip(gray, 0, 255).astype(np.uint8)
    raise ValueError("Formato de imagen no soportado")

IMAGE_GRAY = to_grayscale(IMAGE_OBJECT)


def objective_function(subject):
    x = subject['variables'][0]
    y = subject['variables'][1]
    transformed_image = get_sigmoid_transform(IMAGE_GRAY, x, y)
    # Optional safeguard against collapse
    if transformed_image.std() < 2:
        return 1e9
    contraste = get_sobel(transformed_image)
    return -contraste

def get_sigmoid_transform(image, x, y):
    image = ensure_uint8(image)
    i = np.arange(256) / 255.0
    z = -x * (i - y)
    z = np.clip(z, -60, 60)

    lut = 255 * (1 / (1 + np.exp(z)))
    lut = np.clip(lut, 0, 255).astype(np.uint8)

    return lut[image]

def get_sigmoid_image(image, x, y):
    image = ensure_uint8(image)
    i = np.arange(256) / 255.0
    z = -x * (i - y)
    z = np.clip(z, -60, 60)

    lut = 255 * (1 / (1 + np.exp(z)))
    lut = np.clip(lut, 0, 255).astype(np.uint8)

    return lut[image]

def get_entropy(image):
    image = ensure_uint8(image)
    histogram, _ = np.histogram(image, bins=256, range=(0, 256))
    probabilities = histogram / histogram.sum()
    probabilities = probabilities[probabilities > 0]
    return float(-np.sum(probabilities * np.log2(probabilities)))

def get_sobel(image):
    image = ensure_uint8(image).astype(np.float32)
    if image.ndim != 2:
        raise ValueError(f"get_sobel esperaba imagen 2D, recibió {image.shape}")
    sobel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float32)
    sobel_y = np.array([
        [ 1,  2,  1],
        [ 0,  0,  0],
        [-1, -2, -1]
    ], dtype=np.float32)
    gx = convolve2d(image, sobel_x, mode='same', boundary='symm')
    gy = convolve2d(image, sobel_y, mode='same', boundary='symm')
    magnitude = np.sqrt(gx**2 + gy**2)
    return float(np.mean(magnitude))


if __name__ == "__main__":
    x = ALPHA
    y = DELTA
    subject = {'variables': [x, y], 'fitness': None}
    fitness = objective_function(subject)
    print(f"Fitness for x={x}, y={y}: {fitness}")

    #show the original and transformed images
    transformed_image = get_sigmoid_image(IMAGE_OBJECT, x, y)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(IMAGE_OBJECT, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.title("Transformed Image")
    plt.imshow(transformed_image, cmap='gray')
    plt.show()
    






















