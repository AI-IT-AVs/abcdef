import importlib


ext_module = importlib.import_module('posdiff.ext')


def grid_subsample(points, lengths, voxel_size):

    s_points, s_lengths = ext_module.grid_subsampling(points, lengths, voxel_size)
    return s_points, s_lengths
