import matplotlib.pyplot as plt
from scipy import ndimage

image = ndimage.imread("C:\Users\Shtiliyan\Desktop\pytest\S005_001_00000003 - Copy - Copy.png", False, 'L')
plt.imshow(image, interpolation='none', cmap='Greys_r')
circle1 = plt.Circle((100, 100), 5, color='r')
circle2 = plt.Circle((0.5, 0.5), 0.2, color='blue')
circle3 = plt.Circle((1, 1), 0.2, color='g', clip_on=False)

#fig, ax = plt.subplots() # note we must use plt.subplots, not plt.subplot
# (or if you have an existing figure)
fig = plt.gcf()
ax = fig.gca()

ax.add_artist(circle1)
ax.add_artist(circle2)
ax.add_artist(circle3)

plt.show()
print image