import numpy as np
import matplotlib.pyplot as plt
import imageio
import cv2
import argparse
from pyGCO.gco import pygco
import pickle
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os

__title__ = 'Initial masking tester'
__Version__ = '1.0'
__Author__ = 'Kim, Huijo'
__Contact__ = 'huijo.k@hexafarms.com'
__Licence__ = 'Hexafarms'


def fg_pmap_hist(img, fg_histogram, bg_histogram):
    h, w, c = img.shape
    n_bins = len(fg_histogram)
    binned_im = (img.astype(np.float32)/256*n_bins).astype(int)

    # prior probabilities
    p_fg = 0.5
    p_bg = 1 - p_fg

    # extract fg & bg prob from histograms
    p_rgb_given_fg = fg_histogram[binned_im[:, :, 0],
                                  binned_im[:, :, 1],
                                  binned_im[:, :, 2]]

    p_rgb_given_bg = bg_histogram[binned_im[:, :, 0],
                                  binned_im[:, :, 1],
                                  binned_im[:, :, 2]]

    p_fg_given_rgb = (p_fg * p_rgb_given_fg /
                      (p_fg * p_rgb_given_fg + p_bg * p_rgb_given_bg))
    return p_fg_given_rgb


def fg_pmap_MoG(img, fg_gmm, bg_gmm, mode = 'MinMax'):

    # prior probabilities
    p_fg = 0.5
    p_bg = 1 - p_fg

    # Standardization of the image
    scalers = {}
    img = img.astype(np.float32)

    for i in range(img.shape[2]):
        if mode == 'Standard':
            scalers[i] = StandardScaler()
        else:
            scalers[i] = MinMaxScaler()
        img[:,:,i] = scalers[i].fit_transform(img[:,:,i])

    pixel_values = img.reshape((-1, 3))

    # extract fg & bg prob from histograms
    p_rgb_given_fg = np.exp(fg_gmm.score_samples(pixel_values))

    p_rgb_given_bg = np.exp(bg_gmm.score_samples(pixel_values))

    p_fg_given_rgb = (p_fg * p_rgb_given_fg /
                      (p_fg * p_rgb_given_fg + p_bg * p_rgb_given_bg))
    # reshape to original image size
    return p_fg_given_rgb.reshape(img.shape[:2])


def unary_potentials(probability_map, unary_weight):
    return -unary_weight * np.log(probability_map) 


def pairwise_potential_prefactor(img, x1, y1, x2, y2, pairwise_weight):
    # * np.exp(-1e-1*np.sum((img[y1,x1]-img[y2,x2])**2))
    return pairwise_weight


def coords_to_index(x, y, width):
    return y * width + x


def pairwise_potentials(im, pairwise_weight):
    edges = []
    costs = []

    im = im.astype(np.float32)/255
    h, w = im.shape[:2]

    for y in range(h):
        for x in range(w):
            # Neighbor coordinates
            xs_neigh = x + np.array([0, 1, 0, -1])
            ys_neigh = y + np.array([-1, 0, 1, 0])

            # Make sure neighbors are within image
            mask = np.logical_and(
                np.logical_and(xs_neigh >= 0, xs_neigh < w),
                np.logical_and(ys_neigh >= 0, ys_neigh < h))
            xs_neigh = xs_neigh[mask]
            ys_neigh = ys_neigh[mask]

            center_index = coords_to_index(x, y, w)
            for x_neigh, y_neigh in zip(xs_neigh, ys_neigh):
                cost = pairwise_potential_prefactor(
                    im, x, y, x_neigh, y_neigh, pairwise_weight)
                neighbor_index = coords_to_index(x_neigh, y_neigh, w)
                edges.append((center_index, neighbor_index))
                costs.append(cost)

    edges = np.array(edges)
    costs = np.array(costs)

    return edges, costs


def graph_cut(unary_fg, unary_bg, pairwise_edges, pairwise_costs):
    unaries = np.stack([unary_bg.flat, unary_fg.flat], axis=-1)
    labels = pygco.cut_general_graph(
        pairwise_edges, pairwise_costs, unaries,
        1-np.eye(2), n_iter=-1, algorithm='expansion')
    return labels.reshape(unary_fg.shape)


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

def plot(im, graph_cut_result, thres=30, visualize=True):

    overlap = draw_mask_on_image(im, graph_cut_result)

    fig, axes = plt.subplots(2, 2, figsize=(12,5))
    axes[0,0].set_title("Leaf Segmentation")
    axes[0,0].imshow(graph_cut_result)
    axes[0,1].set_title('Original Image')
    axes[0,1].imshow(im)
    axes[1,0].set_title("Overlap")
    axes[1,0].imshow(overlap)
    
    kernel = np.ones((21, 21), np.uint8)

    output = cv2.morphologyEx(graph_cut_result.astype(
        'uint8'), cv2.MORPH_OPEN, kernel)
    contours, hierarchy = cv2.findContours(output,
                                           cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    area = 0
    filtered_contours = []
    for contour in contours:
        c_area = cv2.contourArea(contour)
        if c_area > thres:
            area += c_area
            filtered_contours.append(contour)

    leaf_area = np.zeros(im.shape[:2])
    cv2.drawContours(leaf_area, filtered_contours, -1, 255, 3)

    axes[1,1].set_title("Total leaf-area:{}".format(area))
    axes[1,1].imshow(leaf_area)
    fig.tight_layout()

    if visualize:
        plt.show()
    return overlap

def segment_api(histograms, in_dir, out_dir):

    unary_weight = 0.1
    pairwise_weight = 1

    im = imageio.imread(in_dir)

    fg_histogram, bg_histogram = np.load(histograms)
    foreground_prob = fg_pmap_hist(im, fg_histogram, bg_histogram)

    unary_fg = unary_potentials(foreground_prob, unary_weight)
    unary_bg = unary_potentials(1 - foreground_prob, unary_weight)

    pairwise_edges, pairwise_costs = pairwise_potentials(
        im, pairwise_weight=pairwise_weight)

    graph_cut_result = graph_cut(
        unary_fg, unary_bg, pairwise_edges, pairwise_costs)

    overlap = plot(im, graph_cut_result, thres=30, visualize=False)

    breakpoint()

    imageio.imwrite(os.path.join(out_dir, os.path.basename(in_dir)), overlap)


def parse_args():
    '''Parse input arguments'''
    parser = argparse.ArgumentParser(
        description="Graph cut")
    parser.add_argument("--input",
                        help="Location of input images to segment.")

    parser.add_argument("--histograms",
                        default='ground_data/histograms.npy',
                        help="Histograms of foreground and background.")

    parser.add_argument("--fg",
                        default='ground_data/fg_gmm.pkl',
                        help="Mixture of Gaussian of foreground.")

    parser.add_argument("--bg",
                        default='ground_data/bg_gmm.pkl',
                        help="Mixture of Gaussian of background.")

    parser.add_argument("--mode",
                        default='Histogram',
                        choices=['MoG', 'Histogram'],
                        help="Select the method to compute the probability distiribution. MoG or Histogram"
    )

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    # im = imageio.imread(args.input)
    im = imageio.imread('data/test.jpg')

    if args.mode == 'MoG':
        with open(args.fg, 'rb') as f:
            fg_gmm = pickle.load(f)
        with open(args.bg, 'rb') as b:
            bg_gmm = pickle.load(b)
        
        foreground_prob = fg_pmap_MoG(im, fg_gmm, bg_gmm)

    else:
        fg_histogram, bg_histogram = np.load(args.histograms)
        foreground_prob = fg_pmap_hist(im, fg_histogram, bg_histogram)

    plt.imshow(foreground_prob)
    plt.show()

    ''' main parameters '''
    unary_weight = 0.1
    pairwise_weight = 1
    thres = 30

    unary_fg = unary_potentials(foreground_prob, unary_weight)
    unary_bg = unary_potentials(1 - foreground_prob, unary_weight)

    pairwise_edges, pairwise_costs = pairwise_potentials(
        im, pairwise_weight=pairwise_weight)

    graph_cut_result = graph_cut(
        unary_fg, unary_bg, pairwise_edges, pairwise_costs)

    plot(im, graph_cut_result, thres)

    

'''
example input

python cut.py data/0.jpg --histograms ground_data/histograms.npy

'''