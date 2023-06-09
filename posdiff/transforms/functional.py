import math
import random

import numpy as np


def normalize_points(points):

    points = points - points.mean(axis=0)
    points = points / np.max(np.linalg.norm(points, axis=1))
    return points


def sample_points(points, num_samples, normals=None):

    points = points[:num_samples]
    if normals is not None:
        normals = normals[:num_samples]
        return points, normals
    else:
        return points


def random_sample_points(points, num_samples, normals=None):

    num_points = points.shape[0]
    sel_indices = np.random.permutation(num_points)
    if num_points > num_samples:
        sel_indices = sel_indices[:num_samples]
    elif num_points < num_samples:
        num_iterations = num_samples // num_points
        num_paddings = num_samples % num_points
        all_sel_indices = [sel_indices for _ in range(num_iterations)]
        if num_paddings > 0:
            all_sel_indices.append(sel_indices[:num_paddings])
        sel_indices = np.concatenate(all_sel_indices, axis=0)
    points = points[sel_indices]
    if normals is not None:
        normals = normals[sel_indices]
        return points, normals
    else:
        return points


def random_scale_shift_points(points, low=2.0 / 3.0, high=3.0 / 2.0, shift=0.2, normals=None):

    scale = np.random.uniform(low=low, high=high, size=(1, 3))
    bias = np.random.uniform(low=-shift, high=shift, size=(1, 3))
    points = points * scale + bias
    if normals is not None:
        normals = normals * scale
        normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
        return points, normals
    else:
        return points


def random_rotate_points_along_up_axis(points, normals=None):

    theta = np.random.rand() * 2.0 * math.pi

    rotation_t = np.array([
        [math.cos(theta), math.sin(theta), 0],
        [-math.sin(theta), math.cos(theta), 0],
        [0, 0, 1],
    ])

    points = np.matmul(points, rotation_t)
    if normals is not None:
        normals = np.matmul(normals, rotation_t)
        return points, normals
    else:
        return points


def random_rescale_points(points, low=0.8, high=1.2):

    scale = random.uniform(low, high)
    points = points * scale
    return points


def random_jitter_points(points, scale, noise_magnitude=0.05):

    noises = np.clip(np.random.normal(scale=scale, size=points.shape), a_min=-noise_magnitude, a_max=noise_magnitude)
    points = points + noises
    return points


def random_shuffle_points(points, normals=None):

    indices = np.random.permutation(points.shape[0])
    points = points[indices]
    if normals is not None:
        normals = normals[indices]
        return points, normals
    else:
        return points


def random_dropout_points(points, max_p):

    num_points = points.shape[0]
    p = np.random.rand(num_points) * max_p
    masks = np.random.rand(num_points) < p
    points[masks] = points[0]
    return points


def random_jitter_features(features, mu=0, sigma=0.01):

    if random.random() < 0.95:
        features = features + np.random.normal(mu, sigma, features.shape).astype(np.float32)
    return features


def random_sample_plane():

    phi = np.random.uniform(0.0, 2 * np.pi)  
    theta = np.random.uniform(0.0, np.pi)  

    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    normal = np.asarray([x, y, z])

    return normal


def random_crop_point_cloud_with_plane(points, p_normal=None, keep_ratio=0.7, normals=None):

    num_samples = int(np.floor(points.shape[0] * keep_ratio + 0.5))
    if p_normal is None:
        p_normal = random_sample_plane()  # (3,)
    distances = np.dot(points, p_normal)
    sel_indices = np.argsort(-distances)[:num_samples] 
    points = points[sel_indices]
    if normals is not None:
        normals = normals[sel_indices]
        return points, normals
    else:
        return points


def random_sample_viewpoint(limit=500):

    return np.random.rand(3) + np.array([limit, limit, limit]) * np.random.choice([1.0, -1.0], size=3)


def random_crop_point_cloud_with_point(points, viewpoint=None, keep_ratio=0.7, normals=None):

    num_samples = int(np.floor(points.shape[0] * keep_ratio + 0.5))
    if viewpoint is None:
        viewpoint = random_sample_viewpoint()
    distances = np.linalg.norm(viewpoint - points, axis=1)
    sel_indices = np.argsort(distances)[:num_samples]
    points = points[sel_indices]
    if normals is not None:
        normals = normals[sel_indices]
        return points, normals
    else:
        return points
