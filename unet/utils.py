
import PIL
import numpy as np
import random



def get_square(img, pos):
    """Extract a left or a right square from PILimg shape : (H, W, C))"""
    img = np.array(img)
    h = img.shape[0]
    if pos == 0:
        return img[:, :h]
    else:
        return img[:, -h:]



def batch(iterable, batch_size):
    """Yields lists by batch"""
    b = []
    for i, t in enumerate(iterable):
        b.append(t)
        if (i+1) % batch_size == 0:
            yield b
            b = []

    if len(b) > 0:
        yield b



def normalize(x):
    return x / 255