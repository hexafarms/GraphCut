import numpy as np
import matplotlib.pyplot as plt
import imageio
import argparse

__Version__ = '1.0'
__Author__ = 'Kim, Huijo'
__Contact__ = 'huijo.k@hexafarms.com'
__Licence__ = 'Hexafarms'


def calculate_histograms(images, loc_masks, n_bins):

    combined_histogram = np.full((n_bins, n_bins, n_bins), fill_value=0.001)

    for image, loc_mask in zip(images, loc_masks):

        histogram = np.full((n_bins, n_bins, n_bins), fill_value=0.001)

        im = imageio.imread(image)
        h, w = im.shape[:2]

        init_mask = np.zeros([h, w])
        init_mask[loc_mask[1]:loc_mask[3], loc_mask[0]:loc_mask[2]] = 1

        # convert values to range of bins
        binned_im = (im.astype(np.float32)/256*n_bins).astype(int)

        for y in range(im.shape[0]):
            for x in range(im.shape[1]):
                if init_mask[y, x] != 0:
                    histogram[binned_im[y, x, 0],
                              binned_im[y, x, 1],
                              binned_im[y, x, 2]] += 1
        combined_histogram += histogram

    # normalize
    combined_histogram /= np.sum(combined_histogram)

    return combined_histogram

def parse_args():
    '''Parse input arguments'''
    parser = argparse.ArgumentParser(
                description="Generate color histogram")

    parser.add_argument("image_dir", 
                        nargs='+', 
                        type=str, 
                        default='images\image-1550434545.jpg',
                        help="Location of input images in list form to generate the color histogram.")
    
    parser.add_argument("--init_fg_masks", 
                        nargs='*',
                        action='append',
                        type=int,
                        help="init foreground masks in a list form.")
    
    parser.add_argument("--init_bg_masks", 
                        nargs='*',
                        action='append',
                        type=int,
                        help="init background masks in a list form.")
    
    parser.add_argument("--work_dir",
                        help="Directory to save the color histogram.")

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    n_bins = 15

    fg_histogram = calculate_histograms(args.image_dir, args.init_fg_masks, n_bins)
    bg_histogram = calculate_histograms(args.image_dir, args.init_bg_masks, n_bins)

    np.save(args.work_dir + "/" +"histograms",
               (fg_histogram,bg_histogram),
               )

    fig, axes = plt.subplots(
    3, 2, figsize=(5,5), sharex=True, 
    sharey=True, num='Relative frequency of color bins')

    x = np.arange(n_bins)
    axes[0,0].bar(x, np.sum(fg_histogram, (1, 2)))
    axes[0,0].set_title('red (foreground)')
    axes[1,0].bar(x, np.sum(fg_histogram, (0, 2)))
    axes[1,0].set_title('green (foreground)')
    axes[2,0].bar(x, np.sum(fg_histogram, (0, 1)))
    axes[2,0].set_title('blue (foreground)')

    axes[0,1].bar(x, np.sum(bg_histogram, (1, 2)))
    axes[0,1].set_title('red (background)')
    axes[1,1].bar(x, np.sum(bg_histogram, (0, 2)))
    axes[1,1].set_title('green (background)')
    axes[2,1].bar(x, np.sum(bg_histogram, (0, 1)))
    axes[2,1].set_title('blue (background)')
    fig.tight_layout()
    plt.show()




"""
example input 
python generate_histogram.py images\image-1550434545.jpg images\image-1550079998.jpg images\image-1550434545.jpg images\image-1550434545.jpg ^
--init_fg_masks 350 450 1800 1910 ^
--init_fg_masks 880 940 1820 1920 ^
--init_fg_masks 1450 1500 1850 1940 ^
--init_fg_masks 950 1000 1200 1250 ^
--init_bg_masks 0 240 0 2500 ^
--init_bg_masks 250 320 0 500 ^
--init_bg_masks 0 1900 1420 1750 ^
--init_bg_masks 1000 1500 1000 1100 ^
--work_dir ground_data
"""