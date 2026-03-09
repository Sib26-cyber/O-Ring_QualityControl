import numpy as np


def to_binary_mask(binary_img):
    # Convert 0/255 binary image into 0/1 mask
    return (binary_img > 0).astype(np.uint8)


def to_binary_image(mask):
    # Convert 0/1 mask back into 0/255 binary image
    return (mask * 255).astype(np.uint8)


def erode(mask, kernel_size=3):
    # Shrink foreground: pixel stays 1 only if all neighbours are 1  
    # all = erosion, any = dilation
    pad = kernel_size // 2
    padded = np.pad(mask, pad, mode="constant", constant_values=0)
    output = np.zeros_like(mask)
    
    for x in range(mask.shape[0]):
        for y in range(mask.shape[1]):
            window = padded[x:x + kernel_size, y:y + kernel_size]
            output[x, y] = 1 if np.all(window == 1) else 0

    return output


def dilate(mask, kernel_size=3):
    # Expand foreground: pixel becomes 1 if any neighbour is 1
    pad = kernel_size // 2
    padded = np.pad(mask, pad, mode="constant", constant_values=0)
    output = np.zeros_like(mask)

    for x in range(mask.shape[0]):
        for y in range(mask.shape[1]):
            window = padded[x:x + kernel_size, y:y + kernel_size]
            output[x, y] = 1 if np.any(window == 1) else 0

    return output


def opening(mask, kernel_size=3):
    # Remove isolated foreground noise
    eroded = erode(mask, kernel_size)
    opened = dilate(eroded, kernel_size)
    return opened


def closing(mask, kernel_size=3):
    # Fill small gaps in the foreground
    dilated = dilate(mask, kernel_size)
    closed = erode(dilated, kernel_size)
    return closed

#def clean_ring_binary(binary_img, kernel_size=3):
    # Full cleanup for thresholded O-ring image
  #  mask = to_binary_mask(binary_img)

   # cleaned = opening(mask, kernel_size=kernel_size)
   # cleaned = closing(cleaned, kernel_size=kernel_size)

   # return to_binary_image(cleaned)
   
def clean_ring_binary(binary_img, kernel_size=3):
    mask = to_binary_mask(binary_img)
    cleaned = opening(mask, kernel_size=kernel_size)
    cleaned = opening(cleaned, kernel_size=kernel_size)
    return to_binary_image(cleaned)