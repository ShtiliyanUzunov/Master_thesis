import os
import numpy
import matplotlib.pyplot as plt
from scipy import ndimage

myList = []
pathToImages = "C:\Users\Shtiliyan\Desktop\pytest"
imagesList = os.listdir(pathToImages)
images = numpy.empty(len(imagesList), dtype=numpy.ndarray)
for i in range(0, len(imagesList)):
    imageName = imagesList[i]
    if not imageName.endswith(".png"):
        continue
    imagePath = os.path.join(pathToImages, imageName)
    image = ndimage.imread(imagePath, True, 'RGB')
    images[i] = image
    
plt.imshow(images[0])
plt.show()
input("Enter something")