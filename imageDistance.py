
import numpy as np
from PIL import Image

# Short script for comparing the distance of two images

def calculateDistance(i1, i2):
    return np.sum((i1-i2)**2)

# Fill in file paths here
fileName1 = ''
fileName2 = ''

image1 = Image.open(fileName1)


image2 = Image.open(fileName2)

image1_sequence = image1.getdata()
image2_sequence = image2.getdata()

image1_array = np.array(image1_sequence)
image2_array = np.array(image2_sequence)

distance = calculateDistance(image1_array, image2_array)
print(distance)
