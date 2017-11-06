from skimage import io, color, measure
from scipy import ndimage, stats
import numpy as np
from os import path


def get_fov_mask(image_rgb, threshold=0.01):
    '''
    Automatically calculate the FOV mask (see Orlando et al., SIPAIM 2016 for further details)
    '''

    illuminant = "D50" # default illuminant value from matlab implementation

    # format: [H, W, #channels]
    image_lab = color.rgb2lab(image_rgb)
    # normalize the luminosity plane
    image_lab[:, :, 0] /= 100.0
    # threshold the plane at the given threshold
    mask = image_lab[:, :, 0] >= threshold

    # fill holes in the resulting mask
    mask = ndimage.binary_fill_holes(mask)
    mask = ndimage.filters.median_filter(mask, size=(5, 5))

    # get connected components
    connected_components = measure.label(mask).astype(float)

    # replace background found in [0][0] to nan so mode skips it
    connected_components[connected_components == mask[0][0]] = np.nan

    # get largest connected component (== mode of the image)
    largest_component_label = stats.mode(connected_components, axis=None, nan_policy='omit')[0]

    # use the modal value of the labels as the final mask
    mask = connected_components == largest_component_label

    return mask.astype(float)



def crop_fov_mask(image_rgb, fov_mask):
    '''
    Crop an image and its FOV mask around the FOV
    '''

    rows = np.any(fov_mask, axis=1)
    cols = np.any(fov_mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    cropped_fov = fov_mask[rmin:rmax, cmin:cmax]
    cropped_image = image_rgb[rmin:rmax, cmin:cmax, :]

    return cropped_image.astype(np.uint8), cropped_fov



def generate_fov_masks(image_path, image_filenames, threshold=0.01):
    '''
    Generate FOV masks for all the images in image_filenames
    '''

    for i in range(0, len(image_filenames)):
        # get current filename
        current_filename = path.basename(image_filenames[i])
        # read the image
        img = io.imread(path.join(image_path, current_filename))
        # get fov mask
        fov_mask = get_fov_mask(img, threshold)
        # save the fov mask
        io.imsave(path.join(image_path, current_filename[:-4] + '_fov_mask.png'), fov_mask)