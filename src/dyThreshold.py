import time
import numpy as np
from histogram import calculate_histogram



def manual_binary_threshold(gray_img, threshold):
    
    #Apply binary threshold manually using nested loops.
    #Since O-rings are dark on a light background,
    #dark pixels become foreground (255), background becomes 0 #
    
    height, width = gray_img.shape
    thresholded_img = np.zeros((height, width), dtype=np.uint8)

    before = time.time()

    for x in range(height):
        for y in range(width):
            if gray_img[x, y] < threshold:
                thresholded_img[x, y] = 255
            else:
                thresholded_img[x, y] = 0

    elapsed = time.time() - before
    return thresholded_img, elapsed




def smooth_histogram(histogram, window_size=5):
    
    #Smooth histogram using a moving average#
    
    half_window = window_size // 2
    smoothed = [0] * len(histogram)

    for i in range(len(histogram)):
        start = max(0, i - half_window)
        end = min(len(histogram), i + half_window + 1)
        smoothed[i] = sum(histogram[start:end]) / (end - start)

    return smoothed

def find_two_main_peaks(histogram):
    
    #Find the two main peaks in the histogram#
    
    first_peak = max(range(256), key=lambda i: histogram[i])

    suppressed = histogram[:]
    for i in range(max(0, first_peak - 10), min(256, first_peak + 11)):
        suppressed[i] = 0

    second_peak = max(range(256), key=lambda i: suppressed[i])

    return first_peak, second_peak

def calculate_two_valley_threshold(gray_img):
    #Find threshold as the valley between the two dominant peaks.
    # 
    histogram = calculate_histogram(gray_img)
    smoothed_histogram = smooth_histogram(histogram, window_size=7)

    peak_a, peak_b = find_two_main_peaks(smoothed_histogram)
    left_peak = min(peak_a, peak_b)
    right_peak = max(peak_a, peak_b)

    if right_peak - left_peak <= 1:
        return 127

    valley_index = left_peak
    valley_value = smoothed_histogram[left_peak]
    for i in range(left_peak + 1, right_peak):
        if smoothed_histogram[i] < valley_value:
            valley_value = smoothed_histogram[i]
            valley_index = i

    return int(valley_index)

def manual_two_valley_threshold(gray_img):
    
    #Calculate dynamic threshold and apply manual thresholding.#
    before = time.time()

    threshold_value = calculate_two_valley_threshold(gray_img)
    thresholded_img, _ = manual_binary_threshold(gray_img, threshold_value)

    elapsed = time.time() - before
    return thresholded_img, threshold_value, elapsed
