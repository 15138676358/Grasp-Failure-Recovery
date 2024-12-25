import numpy as np
from scipy.interpolate import interp1d

# 生成随机轮廓, 包括两个相互垂直的矩形。详见论文agent5
def generate_contour():
    # Create a block comsists of two rectangles, whose size is length x width
    # initialize the center of the main rectangle
    center = np.random.uniform(5, 45, 2) / 50
    theta = np.random.uniform(0, 2*np.pi)
    length = np.random.uniform(10, 30) / 50
    width = np.random.uniform(3, 7) / 50 # decrease the width

    t_vec = np.array([np.cos(theta), np.sin(theta)])
    n_vec = np.array([-np.sin(theta), np.cos(theta)])
    pt1 = center + length / 2 * t_vec + width / 2 * n_vec
    pt2 = center + length / 2 * t_vec - width / 2 * n_vec
    pt3 = center - length / 2 * t_vec - width / 2 * n_vec
    pt4 = center - length / 2 * t_vec + width / 2 * n_vec

    # initialize the size of the co-rectangles
    co_length = np.random.uniform(0, 30, 2) / 50
    co_width = np.random.uniform(3, 7) / 50 # decrease the width
    offset = np.random.uniform(0, length / 2 - co_width / 2)

    pt5 = center + offset * t_vec + co_width / 2 * t_vec + co_length[0] / 2 * n_vec
    pt6 = center + offset * t_vec + co_width / 2 * t_vec + width / 2 * n_vec
    pt7 = center + offset * t_vec + co_width / 2 * t_vec - width / 2 * n_vec
    pt8 = center + offset * t_vec + co_width / 2 * t_vec - co_length[1] / 2 * n_vec
    pt9 = center + offset * t_vec - co_width / 2 * t_vec + co_length[0] / 2 * n_vec
    pt10 = center + offset * t_vec - co_width / 2 * t_vec + width / 2 * n_vec
    pt11 = center + offset * t_vec - co_width / 2 * t_vec - width / 2 * n_vec
    pt12 = center + offset * t_vec - co_width / 2 * t_vec - co_length[1] / 2 * n_vec

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

    return contour, convex

def interpolate_contour(contour, num_grid=100):
        # Interpolate the contour to a grid
        # contour: a list of points (x, y)
        # num_grid: the number of grids
        distances = np.sqrt(np.sum(np.diff(contour, axis=0)**2, axis=1))  # Calculate the distance between each point
        cumulative_lengths = np.concatenate(([0], np.cumsum(distances)))  # Calculate the cumulative length
        total_length = cumulative_lengths[-1]
        new_lengths = np.linspace(0, total_length, num_grid)
        interp_func_x = interp1d(cumulative_lengths, contour[:, 0], kind='linear')
        interp_func_y = interp1d(cumulative_lengths, contour[:, 1], kind='linear')
        x_new = interp_func_x(new_lengths)
        y_new = interp_func_y(new_lengths)
        interpolated_contour = np.column_stack((x_new, y_new))

        return interpolated_contour

def add_normal_to_contour(contour):
    # Calculate the normal of the contour
    # contour: a list of points (x, y)
    # return: a list of points (x, y, theta)
    x, y= contour[:, 0], contour[:, 1]
    dx, dy = np.gradient(x), np.gradient(y)
    theta = np.arctan2(dy, dx) - np.pi / 2
    # Normalize the theta to [0, 2 * pi]
    theta = np.where(theta < 0, theta + 2 * np.pi, theta)
    contour = np.column_stack((x, y, theta))

    return contour
