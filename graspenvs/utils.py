import numpy as np
from scipy.interpolate import interp1d

# 生成随机轮廓, 包括两个相互垂直的矩形。详见论文agent5
def generate_contour():
    # Create a block comsists of two rectangles, whose size is length x width
    # initialize the center of the main rectangle
    center = np.random.uniform(5, 45, 2) / 50
    theta = np.random.uniform(0, 2*np.pi)
    length = np.random.uniform(10, 30) / 50
    width = np.random.uniform(5, 10) / 50

    t_vec = np.array([np.cos(theta), np.sin(theta)])
    n_vec = np.array([-np.sin(theta), np.cos(theta)])
    pt1 = center + length / 2 * t_vec + width / 2 * n_vec
    pt2 = center + length / 2 * t_vec - width / 2 * n_vec
    pt3 = center - length / 2 * t_vec - width / 2 * n_vec
    pt4 = center - length / 2 * t_vec + width / 2 * n_vec
    rectangle1 = np.array([pt1, pt2, pt3, pt4]).copy()

    # initialize the size of the co-rectangles
    co_length = np.random.uniform(0, 30, 2) / 50
    co_width = np.random.uniform(5, 10) / 50
    offset = np.random.uniform(0, length / 2 - co_width / 2)

    pt5 = center + offset * t_vec + co_width / 2 * t_vec + co_length[0] / 2 * n_vec
    pt6 = center + offset * t_vec + co_width / 2 * t_vec + width / 2 * n_vec
    pt7 = center + offset * t_vec + co_width / 2 * t_vec - width / 2 * n_vec
    pt8 = center + offset * t_vec + co_width / 2 * t_vec - co_length[1] / 2 * n_vec
    pt9 = center + offset * t_vec - co_width / 2 * t_vec + co_length[0] / 2 * n_vec
    pt10 = center + offset * t_vec - co_width / 2 * t_vec + width / 2 * n_vec
    pt11 = center + offset * t_vec - co_width / 2 * t_vec - width / 2 * n_vec
    pt12 = center + offset * t_vec - co_width / 2 * t_vec - co_length[1] / 2 * n_vec
    rectangle2 = np.array([pt5, pt8, pt12, pt9]).copy()

    # merge the points
    if co_length[0] / 2 < width / 2:
        left_pts = [pt2, pt1, pt4]
        left_convex = [pt1, pt6, pt10, pt4]
    else:
        left_pts = [pt2, pt1, pt6, pt5, pt9, pt10, pt4]
        left_convex = [pt1, pt5, pt9, pt4]

    if co_length[1] / 2 < width / 2:
        right_pts = [pt4, pt3, pt2]
        right_convex = [pt3, pt11, pt7, pt2]
    else:
        right_pts = [pt4, pt3, pt11, pt12, pt8, pt7, pt2]
        right_convex = [pt3, pt12, pt8, pt2]

    loop_outline = np.array(right_pts + left_pts)

    # rearrange the points to 50 points
    # Compute the cumulative distance along the outline
    distance = np.cumsum(np.sqrt(np.sum(np.diff(loop_outline, axis=0)**2, axis=1)))
    distance = np.insert(distance, 0, 0)

    # Create an interpolator for the x and y coordinates
    fx = interp1d(distance, loop_outline[:, 0])
    fy = interp1d(distance, loop_outline[:, 1])

    # Create an array of evenly spaced distance values
    sample_distances = np.linspace(0, distance[-1], 100)

    # Use the interpolators to compute the x and y coordinates of the samples
    sample_x = fx(sample_distances)
    sample_y = fy(sample_distances)
    contour = np.column_stack((sample_x, sample_y))
    convex = np.array(right_convex + left_convex)
    # scale the contour to 0-1
    scale = np.max((contour.max(axis=0) - contour.min(axis=0)))
    scale_point = contour.min(axis=0)
    contour = (contour - scale_point) / scale
    convex = (convex - scale_point) / scale
    rectangle1 = (rectangle1 - scale_point) / scale
    rectangle2 = (rectangle2 - scale_point) / scale

    return contour, convex