import numpy as np
import matplotlib.pyplot as plt
import imageio
import cv2
import argparse
from pyGCO_master.gco import pygco

__title__ = 'Initial masking tester'
__Version__ = '1.0'
__Author__ = 'Kim, Huijo'
__Contact__ = 'huijo.k@hexafarms.com'
__Licence__ = 'Hexafarms'


def foreground_pmap(img, fg_histogram, bg_histogram):
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


def unary_potentials(probability_map, unary_weight):
    return -unary_weight * np.log(probability_map) / np.log(1000000000)



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


def parse_args():
    '''Parse input arguments'''
    parser = argparse.ArgumentParser(
        description="Graph cut")
    parser.add_argument("input",
                        help="Location of input images to segment.")

    parser.add_argument("--histograms",
                        default='ground_data\histograms.npy',
                        help="Histograms of foreground and background.")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    im = imageio.imread(args.input)
    
    fg_histogram, bg_histogram = np.load(args.histograms)
    foreground_prob = foreground_pmap(im, fg_histogram, bg_histogram)
    unary_weight = 1
    pairwise_weight = 1
    unary_fg = unary_potentials(foreground_prob, unary_weight)
    unary_bg = unary_potentials(1 - foreground_prob, unary_weight)

    pairwise_edges, pairwise_costs = pairwise_potentials(
        im, pairwise_weight=pairwise_weight)

    graph_cut_result = graph_cut(
        unary_fg, unary_bg, pairwise_edges, pairwise_costs)

    kernel = np.ones((21, 21), np.uint8)

    output = cv2.morphologyEx(graph_cut_result.astype(
        'uint8'), cv2.MORPH_OPEN, kernel)
    contours, hierarchy = cv2.findContours(output,
                                           cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    drawing = np.zeros(im.shape[:2])
    cv2.drawContours(drawing, contours, -1, 255, 3)
    plt.imshow(drawing)

    area = 0
    filtered_contours = []
    for contour in contours:
        c_area = cv2.contourArea(contour)
        if c_area > 300:
            area += c_area
            filtered_contours.append(contour)

    drawing2 = np.zeros(im.shape[:2])
    cv2.drawContours(drawing2, filtered_contours, -1, 255, 3)
    plt.title("Total leaf-area:{}".format(area))
    plt.imshow(drawing2)
    plt.show()
