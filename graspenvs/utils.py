import cv2
import matplotlib.path as mplPath
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.interpolate import interp1d

# 生成随机轮廓
def generate_contour():
    # Create a block comsists of two rectangles, whose size is length x width
    # initialize the center of the main rectangle
    center = np.random.uniform(5, 45, 2)
    theta = np.random.uniform(0, 2*np.pi)
    length = np.random.uniform(10, 30)
    width = np.random.uniform(5, 10)
    # center = np.array([25, 25])
    # theta = np.pi / 4
    # length = 20
    # width = 5
    t_vec = np.array([np.cos(theta), np.sin(theta)])
    n_vec = np.array([-np.sin(theta), np.cos(theta)])
    pt_top = center + length / 2 * t_vec
    pt_bottom = center - length / 2 * t_vec
    pt1 = center + length / 2 * t_vec + width / 2 * n_vec
    pt2 = center + length / 2 * t_vec - width / 2 * n_vec
    pt3 = center - length / 2 * t_vec - width / 2 * n_vec
    pt4 = center - length / 2 * t_vec + width / 2 * n_vec

    # initialize the size of the co-rectangles
    co_length = np.random.uniform(0, 30, 2)
    co_width = np.random.uniform(5, 10)
    offset = np.random.uniform(0, length / 2 - co_width / 2)
    # co_length = np.array([20, 20])
    # co_width = 5
    # offset = 0
    # theta += np.random.uniform(-np.pi / 2, np.pi / 2)
    # t_vec = np.array([np.cos(theta), np.sin(theta)])
    # n_vec = np.array([-np.sin(theta), np.cos(theta)])
    pt_left = center + offset * t_vec + co_length[0] / 2 * n_vec
    pt_right = center + offset * t_vec - co_length[1] / 2 * n_vec
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

    # print(co_length, width / 2)
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
    principal_pts = np.row_stack((pt_top, pt_right, pt_bottom, pt_left))

    return contour, convex, principal_pts

# 生成随机状态
def generate_state():
    while True:
        contour, convex, principal_pts = generate_contour()
        if contour.min() > 0 and contour.max() < 50:
            break
    mass = np.array([np.random.uniform(5, 25)])
    # mass = np.array([15])
    while True:
        com = np.random.uniform(0, 50, 2)
        # com = np.array([30, 30])
        if mplPath.Path(convex).contains_point(com) and np.linalg.norm(contour - com, axis=1).min() > 1:
            break
    state = {'contour': contour / 50, 'convex': convex / 50, 'principal_pts': principal_pts / 50, 'mass': mass / 25, 'com': com / 50, 'attempt': 0}

    return state

def line_segment_intersection(p1, p2, q1, q2):
    """计算两条线段 (p1, p2) 和 (q1, q2) 的交点"""
    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    xdiff = (p1[0] - p2[0], q1[0] - q2[0])
    ydiff = (p1[1] - p2[1], q1[1] - q2[1])

    div = det(xdiff, ydiff)
    if div == 0:
        return None  # 平行或共线

    d = (det(p1, p2), det(q1, q2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div

    # 检查交点是否在两条线段上
    if (min(p1[0], p2[0]) <= x <= max(p1[0], p2[0]) and
        min(p1[1], p2[1]) <= y <= max(p1[1], p2[1]) and
        min(q1[0], q2[0]) <= x <= max(q1[0], q2[0]) and
        min(q1[1], q2[1]) <= y <= max(q1[1], q2[1])):
        return x, y
    return None

def find_intersections(convex, line):
    """计算凸多边形与线段的交点"""
    intersections = []
    num_points = len(convex)
    for i in range(num_points):
        p1 = convex[i]
        p2 = convex[(i + 1) % num_points]
        intersection = line_segment_intersection(p1, p2, line[0], line[1])
        if intersection:
            intersections.append(intersection)
    
    pt1, pt2 = intersections[0], intersections[1]
    distance = np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)

    return intersections, distance


if __name__ == '__main__':
    pass
