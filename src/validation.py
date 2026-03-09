import math
import numpy as np


def to_binary_mask(binary_img):
    return (binary_img > 0).astype(np.uint8)


def get_neighbors_8(x, y, height, width):
    neighbors = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue

            nx = x + dx
            ny = y + dy

            if 0 <= nx < height and 0 <= ny < width:
                neighbors.append((nx, ny))

    return neighbors


def connected_component_labeling(binary_img):
    mask = to_binary_mask(binary_img)
    height, width = mask.shape

    label_map = np.zeros((height, width), dtype=np.int32)
    current_label = 0
    components = []

    for x in range(height):
        for y in range(width):
            if mask[x, y] == 1 and label_map[x, y] == 0:
                current_label += 1
                stack = [(x, y)]
                label_map[x, y] = current_label

                pixels = []
                min_x, max_x = x, x
                min_y, max_y = y, y

                while stack:
                    px, py = stack.pop()
                    pixels.append((px, py))

                    min_x = min(min_x, px)
                    max_x = max(max_x, px)
                    min_y = min(min_y, py)
                    max_y = max(max_y, py)

                    for nx, ny in get_neighbors_8(px, py, height, width):
                        if mask[nx, ny] == 1 and label_map[nx, ny] == 0:
                            label_map[nx, ny] = current_label
                            stack.append((nx, ny))

                area = len(pixels)
                centroid_x = float(np.mean([p[1] for p in pixels]))
                centroid_y = float(np.mean([p[0] for p in pixels]))

                components.append(
                    {
                        "label": current_label,
                        "pixels": pixels,
                        "area": area,
                        "bbox": (min_x, min_y, max_x, max_y),
                        "width": max_y - min_y + 1,
                        "height": max_x - min_x + 1,
                        "centroid_x": centroid_x,
                        "centroid_y": centroid_y,
                    }
                )

    return label_map, components


def component_to_mask(component, image_shape):
    mask = np.zeros(image_shape, dtype=np.uint8)
    for x, y in component["pixels"]:
        mask[x, y] = 255
    return mask


def select_best_ring_component(components, image_shape, min_area_ratio=0.005):
    if len(components) == 0:
        return None

    image_height, image_width = image_shape
    image_center_x = image_width / 2.0
    image_center_y = image_height / 2.0
    image_area = image_height * image_width
    min_area = image_area * min_area_ratio

    candidates = []

    for comp in components:
        if comp["area"] < min_area:
            continue

        dx = comp["centroid_x"] - image_center_x
        dy = comp["centroid_y"] - image_center_y
        center_distance = math.sqrt(dx * dx + dy * dy)

        score = comp["area"] - (2.0 * center_distance)
        candidates.append((score, comp))

    if len(candidates) == 0:
        return None

    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def get_foreground_points(binary_img):
    ys, xs = np.where(binary_img > 0)
    return xs, ys


def estimate_ring_center(binary_img):
    xs, ys = get_foreground_points(binary_img)
    if len(xs) == 0:
        return None

    center_x = float(np.mean(xs))
    center_y = float(np.mean(ys))
    return center_x, center_y


def radial_bounds_by_angle(binary_img, center_x, center_y, angle_steps=360):
    height, width = binary_img.shape
    max_radius = int(
        min(center_x, width - 1 - center_x, center_y, height - 1 - center_y)
    )

    inner_radii = np.full(angle_steps, -1.0, dtype=np.float32)
    outer_radii = np.full(angle_steps, -1.0, dtype=np.float32)

    for angle_idx in range(angle_steps):
        theta = 2.0 * math.pi * angle_idx / angle_steps
        hits = []

        for r in range(max_radius + 1):
            x = int(round(center_x + r * math.cos(theta)))
            y = int(round(center_y + r * math.sin(theta)))

            if 0 <= x < width and 0 <= y < height and binary_img[y, x] > 0:
                hits.append(r)

        if len(hits) > 0:
            inner_radii[angle_idx] = float(hits[0])
            outer_radii[angle_idx] = float(hits[-1])

    return inner_radii, outer_radii


def longest_zero_run(binary_array):
    doubled = np.concatenate([binary_array, binary_array])
    current_run = 0
    max_run = 0

    for value in doubled:
        if value == 0:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 0

    return min(max_run, len(binary_array))


def boundary_neighbor_consistency(radius_values, valid_mask, max_step_change=1.5):
    valid_indices = np.where(valid_mask == 1)[0]
    if len(valid_indices) < 10:
        return 0.0

    consistent_count = 0
    comparison_count = 0

    for idx in valid_indices:
        next_idx = (idx + 1) % len(radius_values)
        if valid_mask[next_idx] == 0:
            continue

        delta = abs(float(radius_values[next_idx]) - float(radius_values[idx]))
        if delta <= max_step_change:
            consistent_count += 1
        comparison_count += 1

    if comparison_count == 0:
        return 0.0

    return consistent_count / comparison_count


def max_radius_jump(radius_values, valid_mask):
    valid_indices = np.where(valid_mask == 1)[0]
    if len(valid_indices) < 2:
        return 0.0

    max_jump = 0.0

    for idx in valid_indices:
        next_idx = (idx + 1) % len(radius_values)
        if valid_mask[next_idx] == 0:
            continue

        delta = abs(float(radius_values[next_idx]) - float(radius_values[idx]))
        if delta > max_jump:
            max_jump = delta

    return max_jump


def classify_ring_component(
    binary_img,
    min_boundary_coverage=0.995,
    max_gap_degrees=4,
    max_radius_std=2.5,
    min_thickness=2.0,
    min_inner_radius=5.0,
    max_thickness_ratio=0.92,
    max_step_change=1.5,
    min_neighbor_consistency=0.995,
    max_radius_jump_allowed=3.0,
):
    center = estimate_ring_center(binary_img)
    if center is None:
        return "Fail", {"reason": "No ring foreground found"}

    center_x, center_y = center
    inner_radii, outer_radii = radial_bounds_by_angle(binary_img, center_x, center_y)

    valid_inner = (inner_radii >= 0).astype(np.uint8)
    valid_outer = (outer_radii >= 0).astype(np.uint8)
    valid_both = ((inner_radii >= 0) & (outer_radii >= 0)).astype(np.uint8)

    if np.sum(valid_both) == 0:
        return "Fail", {"reason": "No valid inner/outer boundary found"}

    inner_coverage = float(np.mean(valid_inner))
    outer_coverage = float(np.mean(valid_outer))
    both_coverage = float(np.mean(valid_both))
    missing_gap = int(longest_zero_run(valid_both))

    inner_values = inner_radii[valid_both == 1]
    outer_values = outer_radii[valid_both == 1]
    thickness_values = outer_values - inner_values

    mean_inner_radius = float(np.mean(inner_values))
    mean_outer_radius = float(np.mean(outer_values))
    inner_std = float(np.std(inner_values))
    outer_std = float(np.std(outer_values))
    mean_thickness = float(np.mean(thickness_values))
    thickness_ratio = (
        mean_thickness / mean_outer_radius if mean_outer_radius > 0 else 1.0
    )

    inner_neighbor_consistency = boundary_neighbor_consistency(
        inner_radii, valid_both, max_step_change=max_step_change
    )
    outer_neighbor_consistency = boundary_neighbor_consistency(
        outer_radii, valid_both, max_step_change=max_step_change
    )

    inner_max_jump = max_radius_jump(inner_radii, valid_both)
    outer_max_jump = max_radius_jump(outer_radii, valid_both)

    details = {
        "center_x": round(center_x, 2),
        "center_y": round(center_y, 2),
        "inner_coverage": round(inner_coverage, 4),
        "outer_coverage": round(outer_coverage, 4),
        "both_coverage": round(both_coverage, 4),
        "largest_missing_gap_deg": missing_gap,
        "mean_inner_radius": round(mean_inner_radius, 3),
        "mean_outer_radius": round(mean_outer_radius, 3),
        "inner_radius_std": round(inner_std, 3),
        "outer_radius_std": round(outer_std, 3),
        "mean_ring_thickness": round(mean_thickness, 3),
        "thickness_ratio": round(thickness_ratio, 3),
        "inner_neighbor_consistency": round(inner_neighbor_consistency, 3),
        "outer_neighbor_consistency": round(outer_neighbor_consistency, 3),
        "inner_max_jump": round(inner_max_jump, 3),
        "outer_max_jump": round(outer_max_jump, 3),
    }

    if both_coverage < min_boundary_coverage:
        details["reason"] = "Low boundary coverage"
        return "Fail", details

    if missing_gap > max_gap_degrees:
        details["reason"] = "Gap too large"
        return "Fail", details

    if mean_inner_radius < min_inner_radius:
        details["reason"] = "Inner radius too small"
        return "Fail", details

    if inner_std > max_radius_std or outer_std > max_radius_std:
        details["reason"] = "Boundary irregular"
        return "Fail", details

    if mean_thickness < min_thickness:
        details["reason"] = "Ring too thin"
        return "Fail", details

    if thickness_ratio > max_thickness_ratio:
        details["reason"] = "Thickness ratio too large"
        return "Fail", details

    if inner_neighbor_consistency < min_neighbor_consistency:
        details["reason"] = "Inner boundary inconsistent"
        return "Fail", details

    if outer_neighbor_consistency < min_neighbor_consistency:
        details["reason"] = "Outer boundary inconsistent"
        return "Fail", details

    if inner_max_jump > max_radius_jump_allowed or outer_max_jump > max_radius_jump_allowed:
        details["reason"] = "Local boundary jump too large"
        return "Fail", details

    details["reason"] = "All checks passed"
    return "Pass", details


def validate_oring(binary_img):
    label_map, components = connected_component_labeling(binary_img)

    best_component = select_best_ring_component(components, binary_img.shape)

    if best_component is None:
        return "Fail", {
            "reason": "No suitable ring component found",
            "component_count": len(components),
        }, np.zeros_like(binary_img, dtype=np.uint8)

    ring_mask = component_to_mask(best_component, binary_img.shape)
    label, details = classify_ring_component(ring_mask)

    details["component_count"] = len(components)
    details["selected_component_area"] = best_component["area"]
    details["selected_component_bbox"] = best_component["bbox"]
    details["selected_component_centroid"] = (
        round(best_component["centroid_x"], 2),
        round(best_component["centroid_y"], 2),
    )

    return label, details, ring_mask