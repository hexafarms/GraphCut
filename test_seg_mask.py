import numpy as np
import matplotlib.pyplot as plt
import imageio
import cv2

__title__ = 'Initial masking tester'
__Version__ = '1.0'
__Author__ = 'Kim, Huijo'
__Contact__ = 'huijo.k@hexafarms.com'
__Licence__ = 'Hexafarms'



''' Input '''
# Location of an image
im_loc = "images\image-1550434545.jpg"
# Pixels of fore&back ground. [left top, left botom, right top, right bottom]
ref_fg = [350, 450, 1800, 1910]
ref_bg = [0, 240, 0, 2500]
''''''''''''''''''



def draw_mask_on_image(image, mask, color=(0, 255, 255)):
    """Return a visualization of a mask overlaid on an image."""
    result = image.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_DILATE, kernel)
    outline = dilated > mask
    result[mask == 1] = (result[mask == 1] * 0.4 + 
                         np.array(color) * 0.6).astype(np.uint8)
    result[outline] = color
    return result


im = imageio.imread(im_loc)
h,w = im.shape[:2]

"""
Now set some rectangular region of the initial foreground mask to 1.
This should be a part of the image that is fully foreground.
"""

init_fg_mask = np.zeros([h, w])
init_bg_mask = np.zeros([h, w])
init_fg_mask[ref_fg[0]: ref_fg[1], ref_fg[2]:ref_fg[3]] = 1
init_bg_mask[ref_bg[0]:ref_bg[1], ref_bg[2]:ref_bg[3]] = 1


fig, axes = plt.subplots(1, 2, figsize=(10,5))
axes[0].set_title('Initial foreground mask')
axes[0].imshow(draw_mask_on_image(im, init_fg_mask))
axes[1].set_title('Initial background mask')
axes[1].imshow(draw_mask_on_image(im, init_bg_mask))
fig.tight_layout()
plt.show()