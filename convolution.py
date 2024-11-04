import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def conv2d(image, kernel):
    img_height, img_width = image.shape
    kernel_size = kernel.shape[0]

    output_height = img_height - kernel_size + 1
    output_width = img_width - kernel_size + 1
    output = np.zeros((output_height, output_width))

    for i in range(output_height):
        for j in range(output_width):
            output[i, j] = np.sum(image[i:i+kernel_size, j:j+kernel_size] * kernel)

    return output

img_path = '/content/drive/MyDrive/Univer/cat.jpeg'
img = Image.open(img_path).convert('L')
img = np.array(img)

kernel = np.array([[-1, -1, -1],
                   [-1,  8, -1],
                   [-1, -1, -1]])

conv_result = conv2d(img, kernel)

conv_result = (conv_result - np.min(conv_result)) / (np.max(conv_result) - np.min(conv_result)) * 255
conv_result = np.clip(conv_result * 2, 0, 255)
conv_result = conv_result.astype(np.uint8)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Оригинальное изображение')

plt.subplot(1, 2, 2)
plt.imshow(conv_result, cmap='gray')
plt.title('Результат свертки с повышенным контрастом')

plt.show()
