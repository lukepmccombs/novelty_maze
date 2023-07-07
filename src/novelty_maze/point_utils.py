import numpy as np

def _unpack(lines):
    return np.swapaxes(lines, 0, 1)

def point_distance(point_a, point_b):
    return np.linalg.norm(point_a - point_b)

def multiple_point_distance(point_as, point_b):
    return np.linalg.norm(point_as - point_b, axis=-1)

def angle(vector):
    return np.rad2deg(np.arctan2(*vector[::-1]))

def rotate_vector(vector, angle):
    radians = np.deg2rad(angle)
    cos_sin = np.array([np.cos(radians), np.sin(radians)])
    rot_matrix = np.array([cos_sin, np.flip(cos_sin)])
    rot_matrix[0, 1] *= -1
    return rot_matrix @ vector

def vectorized_dot(left, right):
    return (left * right).sum(-1)

def multiple_intersection(lines_left, lines_right):
    point_as, point_bs = _unpack(lines_left)
    point_cs, point_ds = _unpack(lines_right)

    diff_ab = np.repeat((point_as - point_bs)[:, None, :], len(lines_right), 1)
    diff_ac = point_as[:, None, :] - point_cs[None, :, :]
    diff_cd = np.repeat((point_cs - point_ds)[None, :, :], len(lines_left), 0)

    denom = np.linalg.det(np.stack([diff_ab, diff_cd], -2))
    t = np.linalg.det(np.stack([diff_ac, diff_cd], -2)) / denom
    u = np.linalg.det(np.stack([diff_ac, diff_ab], -2)) / denom

    valid = (0 <= t) & (t <= 1) & (0 <= u) & (u <= 1)
    points = point_as[:, None, :] + t[:, :, None] * (point_bs - point_as)[:, None, :]
    return valid, points

def multiple_line_distance(lines, point):
    point_as, point_bs = _unpack(lines)
    dist_a = point_as - point
    dist_ba = point_bs - point_as
    return np.abs(np.sum(dist_a * np.flip(dist_ba, 1) * [1, -1], axis=1))\
        / np.linalg.norm(dist_ba, axis=1)

def multiple_lies_between(lines, points):
    point_as, point_bs = _unpack(lines)
    dist_ba = point_bs - point_as
    dist_ca = points - point_as
    dp = vectorized_dot(dist_ba, dist_ca)
    return (dp > 0) & (dp < np.square(np.linalg.norm(dist_ba, axis=-1)))

def multiple_proj_line_point(lines, point):
    point_as, point_bs = _unpack(lines)
    vec = point_bs - point_as
    p = point - point_as
    return (vectorized_dot(p, vec) / vectorized_dot(vec, vec))[:, None] * vec + point_as

def multiple_check_line_circle_collide(lines, center, radius):
    lines = np.array(lines)

    check = multiple_line_distance(lines, center) < radius
    if not np.any(check):
        return check
    
    check_lines = lines[check]

    proj_points = multiple_proj_line_point(check_lines, center)
    certain = multiple_lies_between(check_lines, proj_points)
    if np.all(certain):
        return check

    point_as, point_bs = _unpack(check_lines)

    a_dist = multiple_point_distance(proj_points[~certain], point_as[~certain])
    certain[~certain] |= a_dist < radius
    
    b_dist = multiple_point_distance(proj_points[~certain], point_bs[~certain])
    certain[~certain] |= b_dist < radius

    check[check] &= certain
    return check
