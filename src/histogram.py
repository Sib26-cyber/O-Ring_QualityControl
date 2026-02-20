import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def read_image_and_create_histogram(image_path):
    #read the image
    img = cv.imread(image_path)

    if img is None:
        print("Error: Image not found.")
        return None
    #split the image into greyscale manually
    gray_img = convert_to_grayscale(img)
    #calculate the histogram of the grayscale image
    histogram = calculate_histogram(gray_img)

    return img, histogram



def convert_to_grayscale(img):
    #Converts the BGR image to grayscale by averaging the color channels (Red, Green, Blue) for each pixel.
    #Get the image dimensions
    height, width = img.shape[:2]
    #Create an empty grayscale image to store the converted pixel values
    gray = [[0 for _ in range(width)] for _ in range(height)]
    #Iterate through each pixel in the original image and calculate the average intensity to create the grayscale image
    for i in range(height):
        for j in range(width):
            b,g,r = img[i, j]
            #Calculate the average intensity of the pixel by averaging the BGR values
            gray[i][j] = int(0.299 *r + 0.587 * g + 0.114 * b)

    return gray



#Calculates the histogram of image by counting the number of pixels for each intensity value (0-255).
def calculate_histogram(gray_img):
    histogram = [0] * 256
    #Iterate through each pixel in the image and count the intensity values
    for row in gray_img:
        for pixel in row:
            histogram[pixel] += 1


    return histogram





