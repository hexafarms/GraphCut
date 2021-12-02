import numpy as np
import imageio
import argparse
from sklearn.mixture import GaussianMixture
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler

__Version__ = '1.0'
__Author__ = 'Kim, Huijo'
__Contact__ = 'huijo.k@hexafarms.com'
__Licence__ = 'Hexafarms'

def compute_MoG(images, K, loc_masks):

    pixel_values = []

    for image, loc_mask in zip(images, loc_masks):
        im = imageio.imread(image)

        h,w = im.shape[:2]
        
        init_mask = np.zeros([h, w])
        init_mask[loc_mask[1]:loc_mask[3], loc_mask[0]:loc_mask[2]] = 1
        
        crop_im = im[np.ix_(init_mask.any(1), init_mask.any(0))].astype(np.float32)
        
        # Standardization of the image
        scalers = {}
        for i in range(crop_im.shape[2]):
            scalers[i] = MinMaxScaler()
            crop_im[:,:,i] = scalers[i].fit_transform(crop_im[:,:,i])

        # reshape the image to a 2D array of pixels and 3 color values (RGB)
        pixel_values.append(crop_im.reshape((-1, 3)).astype(np.float32))

    pixel_values = np.concatenate(pixel_values, axis=0)

    output = GaussianMixture(n_components = K).fit(pixel_values)

    return output

def parse_args():
    '''Parse input arguments'''
    parser = argparse.ArgumentParser(
                description="Generate color histogram")

    parser.add_argument("image_dir", 
                        nargs='+', 
                        type=str, 
                        default='data',
                        help="Location of input images in list form.")

    parser.add_argument("--init_fg_masks", 
                        nargs='*',
                        action='append',
                        type=int,
                        help="init fg masks in a list form.")
    
    parser.add_argument("--init_bg_masks", 
                        nargs='*',
                        action='append',
                        type=int,
                        help="init bg masks in a list form.")
    
    
    parser.add_argument("--work_dir",
                        help="Directory to save the MoG.")

    args = parser.parse_args()

    return args

def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

def plot(fg_gmm, bg_gmm):
    
    x = np.linspace(-5, 5, 100)
    x = np.dstack([x]*3).reshape(-1,3)
    logprob_fg = fg_gmm.score_samples(x)
    pdf_fg = np.exp(logprob_fg)
    logprob_bg = bg_gmm.score_samples(x)
    pdf_bg = np.exp(logprob_bg)



    fig, axes = plt.subplots(1, 2, figsize=(10,5))
    axes[0].set_title('Foreground')
    axes[0].plot(x, pdf_fg)
    axes[1].set_title('Background')
    axes[1].plot(x, pdf_bg)
    fig.tight_layout()
    plt.show()

    
if __name__ == '__main__':
    args = parse_args()

    K = 15 # the number of mixture gaussian
    images = args.image_dir
    fg_loc_masks = args.init_fg_masks
    bg_loc_masks = args.init_bg_masks

    # images = ['data/1.jpg']
    # fg_loc_masks = [[553, 319, 606, 360]]
    # bg_loc_masks = [[52, 401, 1242, 431]]

    fg_gmm = compute_MoG(images, K, fg_loc_masks)
    bg_gmm = compute_MoG(images, K, bg_loc_masks)

    plot(fg_gmm, bg_gmm)

    # sample usage
    save_object(fg_gmm, os.path.join(args.work_dir,'fg_gmm.pkl'))
    save_object(bg_gmm, os.path.join(args.work_dir,'bg_gmm.pkl'))

"""
example input 
python generate_MoG.py data/1.jpg data/1.jpg data/2.jpg data/2.jpg \
--init_fg_masks  553 319 606 360 \
--init_fg_masks  1215 260 1244 278 \
--init_fg_masks  718 95 740 165 \
--init_fg_masks  273 308 300 335 \
--init_bg_masks 52 401 1242 431 \
--init_bg_masks 0 0 1277 214 \
--init_bg_masks 870 0 1054 717 \
--init_bg_masks 134 5 187 674 \
--work_dir ground_data

"""