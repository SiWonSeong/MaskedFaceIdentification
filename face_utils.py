# coding: utf-8

import cv2
import os
import numpy as np
import pickle

def set_res(cap, x,y):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(x))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(y))
    return str(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),str(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

def mkdir(d):
    os.makedirs(d, exist_ok=True)

def _get_suffix(filename):
    """a.jpg -> jpg"""
    pos = filename.rfind('.')
    if pos == -1:
        return ''
    return filename[pos + 1:]

def _load(fp):
    suffix = _get_suffix(fp)
    if suffix == 'npy':
        return np.load(fp)
    elif suffix == 'pkl':
        return pickle.load(open(fp, 'rb'))

def draw_cali_circle(window_img, center_pt, size_factor, line_size, timer, start_time):
    cv2.circle(window_img, center_pt, int(size_factor) +1, (255,0,0), 5, cv2.LINE_AA)
    cv2.circle(window_img, center_pt, int((size_factor/3)*(timer-start_time)), (0,0,255), -1, cv2.LINE_AA)
    cv2.line(window_img, (center_pt[0] - line_size, center_pt[1]), (center_pt[0] + line_size, center_pt[1]), (0,255,0), 2, cv2.LINE_AA)
    cv2.line(window_img, (center_pt[0], center_pt[1] - line_size), (center_pt[0], center_pt[1] + line_size), (0,255,0), 2, cv2.LINE_AA)

def recon_vers(param_lst, roi_box_lst, u_base, w_shp_base, w_exp_base):
    ver_lst = []
    for param, roi_box in zip(param_lst, roi_box_lst):
        R, offset, alpha_shp, alpha_exp = _parse_param(param)
        pts3d = R @ (u_base + w_shp_base @ alpha_shp + w_exp_base @ alpha_exp). \
            reshape(3, -1, order='F') + offset
        pts3d = similar_transform(pts3d, roi_box, 120)

        ver_lst.append(pts3d)

    return ver_lst

def similar_transform(pts3d, roi_box, size):
    pts3d[0, :] -= 1  # for Python compatibility
    pts3d[2, :] -= 1
    pts3d[1, :] = size - pts3d[1, :]

    sx, sy, ex, ey = roi_box
    scale_x = (ex - sx) / size
    scale_y = (ey - sy) / size
    pts3d[0, :] = pts3d[0, :] * scale_x + sx
    pts3d[1, :] = pts3d[1, :] * scale_y + sy
    s = (scale_x + scale_y) / 2
    pts3d[2, :] *= s
    pts3d[2, :] -= np.min(pts3d[2, :])
    return np.array(pts3d, dtype=np.float32)

def _parse_param(param):
    """matrix pose form
    param: shape=(trans_dim+shape_dim+exp_dim,), i.e., 62 = 12 + 40 + 10
    """

    # pre-defined templates for parameter
    trans_dim, shape_dim, exp_dim = 12, 40, 10

    R_ = param[:trans_dim].reshape(3, -1)
    R = R_[:, :3]
    offset = R_[:, -1].reshape(3, 1)
    alpha_shp = param[trans_dim:trans_dim + shape_dim].reshape(-1, 1)
    alpha_exp = param[trans_dim + shape_dim:].reshape(-1, 1)

    return R, offset, alpha_shp, alpha_exp

def _to_ctype(arr):
    if not arr.flags.c_contiguous:
        return arr.copy(order='C')
    return arr

# To calculate head gaze by 3 rigid facial point
def surface_normal_cross(poly):
    n = np.cross(poly[1,:]-poly[0,:],poly[2,:]-poly[0,:])
    norm = np.linalg.norm(n)
    if norm==0:
        raise ValueError('zero norm')
    else:
        normalised = n/norm
    return n , normalised

def dist_2D(p1, p2):
    squared_dist = np.sum((p1-p2)**2, axis=0)
    dist = np.sqrt(squared_dist)
    return dist

def calc_unit_vec(vec):
    norm = np.linalg.norm(vec)
    if norm == 0:
        raise ValueError('zero norm')
    else:
        normalised = vec/norm
    return normalised