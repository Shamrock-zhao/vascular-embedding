
from skimage import io, color, measure, filters
from scipy import ndimage, stats
import numpy as np
from os import path
import cv2


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



def clahe_enhancement(image):
    '''
    Perform CLAHE contrast enhancement on each color band
    '''

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    for i in range(0, image.shape[2]):
        image[:,:,i] = clahe.apply(image[:,:,i]) 

    return image



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



def replace_out_of_fov_pixels(image_rgb, fov_mask):

    # get image size
    image_size = image_rgb.shape
    # for each color band, apply:
    for color_band in range(0, image_size[2]):

        # get current color band
        current_color_band = image_rgb[:, :, color_band]
        # compute the mean value inside the fov mask
        mean_value = (current_color_band[fov_mask>0]).mean()
        # assign this value to all pixels outside the fov
        current_color_band[fov_mask == 0] = mean_value
        # and copy back the color band
        image_rgb[:, :, color_band] = current_color_band

    return image_rgb


def equalize_fundus_image_intensities(image_rgb, fov_mask):

    # replace out of fov pixels with the average intensity
    image_rgb = replace_out_of_fov_pixels(image_rgb, fov_mask).astype(float)

    # these constants were assigned according to van Grinsven et al. 2016, TMI
    alpha = 4.0
    beta = -4.0
    gamma = 128.0

    # get image size
    image_size = image_rgb.shape

    # initialize the output image with the same size that the input image
    equalized_image = np.zeros(image_size).astype(float)

    # estimate the sigma parameter using the scaling approach by
    # Orlando et al. 2017, arXiv
    sigma = image_size[1] / 30.0

    # for each color band, apply:
    for color_band in range(0, image_size[2]):

        # apply a gaussian filter on the current band to estimate the background
        smoothed_band = ndimage.filters.gaussian_filter(image_rgb[:, :, color_band], sigma, truncate=3)
        # apply the equalization procedure on the current band
        equalized_image[:, :, color_band] = alpha * image_rgb[:, :, color_band] + beta * smoothed_band + gamma
        # remove elements outside the fov
        intermediate = np.multiply(equalized_image[:, :, color_band], fov_mask > 0)

        intermediate[intermediate>255] = 255
        intermediate[intermediate<0] = 0

        equalized_image[:, :, color_band] = intermediate

    return equalized_image.astype(np.uint8)


def preprocess(image, fov_mask, preprocessing=None):
    
    if len(fov_mask.shape) > 2:
        fov_mask = fov_mask[:,:,0]

    if preprocessing == 'rgb':
        preprocessed_image = image # RGB image

    elif preprocessing == 'green':
        preprocessed_image = np.stack((image[:,:,1], image[:,:,1], image[:,:,1]), axis=2) # Green band

    elif preprocessing == 'eq':
        preprocessed_image = equalize_fundus_image_intensities(np.copy(image), fov_mask) # RGB equalized

    elif preprocessing == 'clahe':
        preprocessed_image = clahe_enhancement(np.copy(image))

    return preprocessed_image