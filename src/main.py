import os
import cv2 as cv
from histogram import convert_to_grayscale
from dythreshold import manual_two_valley_threshold
from morphology import clean_ring_binary


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(base_dir)

    image_folder = os.path.join(project_dir, "images")
    results_folder = os.path.join(project_dir, "results")
    binary_folder = os.path.join(results_folder, "binary")  
    cleaned_folder = os.path.join(results_folder, "cleaned")

    os.makedirs(results_folder, exist_ok=True)
    os.makedirs(binary_folder, exist_ok=True)
    os.makedirs(cleaned_folder, exist_ok=True)

    

    for i in range(1, 16):
        image_name = f"Oring{i}.jpg"
        image_path = os.path.join(image_folder, image_name)

        img = cv.imread(image_path)

        if img is None:
            print(f"Could not load {image_path}")
            continue

        gray_img = convert_to_grayscale(img)
        binary_img, threshold_value, elapsed = manual_two_valley_threshold(gray_img)
        cleaned_img = clean_ring_binary(binary_img, kernel_size=3)

        print(f"{image_name}: threshold={threshold_value}, time={elapsed:.4f}s")

        cv.imwrite(os.path.join(binary_folder, f"binary_{image_name}"), binary_img)
        cv.imwrite(os.path.join(cleaned_folder, f"cleaned_{image_name}"), cleaned_img)


if __name__ == "__main__":
    main()