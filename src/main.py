import os
import cv2 as cv
from histogram import convert_to_grayscale
#from dythreshold import manual_two_valley_threshold
from dythreshold import manual_otsu_threshold
from morphology import clean_ring_binary
from validation import validate_oring


def annotate_result(img, label, threshold_value=None):
    annotated = img.copy()

    text = "PASS" if label == "Pass" else "FAIL"
    colour = (0, 255, 0) if label == "Pass" else (0, 0, 255)

    if threshold_value is not None:
        text = f"{text} | T={threshold_value}"

    cv.putText(
        annotated,
        text,
        (10, 30),
        cv.FONT_HERSHEY_SIMPLEX,
        0.8,
        colour,
        2,
        cv.LINE_AA,
    )

    return annotated


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(base_dir)

    image_folder = os.path.join(project_dir, "images")
    results_folder = os.path.join(project_dir, "results")
    binary_folder = os.path.join(results_folder, "binary")
    cleaned_folder = os.path.join(results_folder, "cleaned")
    mask_folder = os.path.join(results_folder, "ringmask")
    classified_folder = os.path.join(results_folder, "classified")

    os.makedirs(results_folder, exist_ok=True)
    os.makedirs(binary_folder, exist_ok=True)
    os.makedirs(cleaned_folder, exist_ok=True)
    os.makedirs(mask_folder, exist_ok=True)
    os.makedirs(classified_folder, exist_ok=True)

    for i in range(1, 16):
        image_name = f"Oring{i}.jpg"
        image_path = os.path.join(image_folder, image_name)

        img = cv.imread(image_path)

        if img is None:
            print(f"Could not load {image_path}")
            continue

        original = img.copy()

        gray_img = convert_to_grayscale(img)
        #binary_img, threshold_value, elapsed = manual_two_valley_threshold(gray_img)
        binary_img, threshold_value, elapsed = manual_otsu_threshold(gray_img)
        cleaned_img = clean_ring_binary(binary_img, kernel_size=3)

        label, details, ring_mask = validate_oring(cleaned_img)
        annotated_img = annotate_result(original, label, threshold_value)

        print(f"{image_name}: {label}, threshold={threshold_value}, time={elapsed:.4f}s")
        print(details)

        cv.imwrite(os.path.join(binary_folder, f"binary_{image_name}"), binary_img)
        cv.imwrite(os.path.join(cleaned_folder, f"cleaned_{image_name}"), cleaned_img)
        cv.imwrite(os.path.join(mask_folder, f"ringmask_{image_name}"), ring_mask)
        cv.imwrite(os.path.join(classified_folder, f"classified_{image_name}"), annotated_img)


if __name__ == "__main__":
    main()