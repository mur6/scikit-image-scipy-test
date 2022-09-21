import math
import numpy as np
import matplotlib.pyplot as plt

from skimage.draw import (line, polygon, disk,
                          circle_perimeter,
                          ellipse, ellipse_perimeter,
                          bezier_curve)


#fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(10, 6))


img = np.zeros((500, 500), dtype=np.double)

def intersect(ary1, ary2):
    aset = set([tuple(x) for x in ary1])
    bset = set([tuple(x) for x in ary2])
    return np.array([x for x in aset & bset])
# # fill polygon
poly = np.array((
    (0, 0),
    (200, 0),
    (200, 200),
    (0, 200),
))
rr, cc = polygon(poly[:, 0], poly[:, 1], img.shape)
img[rr, cc] = 1
# # fill circle
# rr, cc = disk((100, 400), 100, shape=img.shape)
# img[rr, cc] = 1

poly = np.array((
    (200, 0),
    (400, 0),
    (400, 200),
    (200, 200),
))
rr, cc = polygon(poly[:, 0], poly[:, 1], img.shape)
img[rr, cc] = 2

# # fill circle
# rr, cc = disk((100, 200), 100, shape=img.shape)
# img[rr, cc] = 2#(1, 1, 0)




import matplotlib.pyplot as plt
from skimage import measure
#print(r.shape)
contours = measure.find_contours(img, 1.5, mask=(img > 0))
print(len(contours))

for contour in contours:
    print(contour[:, 1], contour[:, 0])
    plt.plot(contour[:, 1], contour[:, 0], linewidth=2)

# fill ellipse
#rr, cc = ellipse(300, 300, 100, 200, img.shape)
#img[rr, cc, 2] = 1


# ellipses


plt.imshow(img)
# ax1.set_title('No anti-aliasing')
# ax1.axis('off')

# ax2.imshow(img, cmap=plt.cm.gray)
# ax2.set_title('Anti-aliasing')
# ax2.axis('off')

plt.show()
