import math
import numpy as np


def get_foreground_points(binary_img):
	#collect coordinates of foreground pixels
	ys, xs = np.where(binary_img > 0)
	return xs, ys


def estimate_ring_center(binary_img):
	#estimate ring center as centroid of foreground pixels
	xs, ys = get_foreground_points(binary_img)
	if len(xs) == 0:
		return None

	center_x = float(np.mean(xs))
	center_y = float(np.mean(ys))
	return center_x, center_y


def estimate_ring_radius(binary_img, center_x, center_y):
	#estimate ring radius from median distance of foreground pixels
	xs, ys = get_foreground_points(binary_img)
	if len(xs) == 0:
		return None

	distances = np.sqrt((xs - center_x) ** 2 + (ys - center_y) ** 2)
	return float(np.median(distances))


def radial_bounds_by_angle(binary_img, center_x, center_y, angle_steps=360):
	#for each angle, find the inner and outer radius where foreground exists
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
	#find the largest consecutive missing-angle gap in circular coverage
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


def boundary_neighbor_consistency(radius_values, valid_mask, max_step_change=3.0):
	#check if neighboring boundary points have similar radius values
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


def classify_ring(
	binary_img,
	min_boundary_coverage=0.85,
	max_gap_degrees=20,
	max_radius_std=8.0,
	min_thickness=2.0,
	min_inner_radius=5.0,
	max_thickness_ratio=0.92,
	max_step_change=3.0,
	min_neighbor_consistency=0.70,
):
	#classify o-ring as pass/fail using both inner and outer ring boundaries
	center = estimate_ring_center(binary_img)
	if center is None:
		return "Fail", {"reason": "No ring foreground found"}

	center_x, center_y = center
	inner_radii, outer_radii = radial_bounds_by_angle(binary_img, center_x, center_y)

	valid_inner = (inner_radii >= 0).astype(np.uint8)
	valid_outer = (outer_radii >= 0).astype(np.uint8)
	valid_both = ((inner_radii >= 0) & (outer_radii >= 0)).astype(np.uint8)

	inner_coverage = float(np.mean(valid_inner))
	outer_coverage = float(np.mean(valid_outer))
	both_coverage = float(np.mean(valid_both))
	missing_gap = int(longest_zero_run(valid_both))

	if np.sum(valid_both) == 0:
		return "Fail", {"reason": "No valid inner/outer ring boundary found"}

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

	is_pass = (
		inner_coverage >= min_boundary_coverage
		and outer_coverage >= min_boundary_coverage
		and both_coverage >= min_boundary_coverage
		and missing_gap <= max_gap_degrees
		and mean_inner_radius >= min_inner_radius
		and inner_std <= max_radius_std
		and outer_std <= max_radius_std
		and mean_thickness >= min_thickness
		and thickness_ratio <= max_thickness_ratio
		and inner_neighbor_consistency >= min_neighbor_consistency
		and outer_neighbor_consistency >= min_neighbor_consistency
	)
	label = "Pass" if is_pass else "Fail"

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
		"min_boundary_coverage": min_boundary_coverage,
		"max_gap_degrees": max_gap_degrees,
		"max_radius_std": max_radius_std,
		"min_thickness": min_thickness,
		"min_inner_radius": min_inner_radius,
		"max_thickness_ratio": max_thickness_ratio,
		"max_step_change": max_step_change,
		"min_neighbor_consistency": min_neighbor_consistency,
	}
	return label, details
