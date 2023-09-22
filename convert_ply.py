from plyfile import PlyData, PlyElement
import numpy as np
import os
from os import makedirs, path
from errno import EEXIST
import sys


def mkdir_p(folder_path):
    # Creates a directory. equivalent to using mkdir -p on the command line
    try:
        makedirs(folder_path)
    except OSError as exc:  # Python >2.5
        if exc.errno == EEXIST and path.isdir(folder_path):
            pass
        else:
            raise


def convert_ply(input_path, output_path):
    plydata = PlyData.read(input_path)

    xyz = np.stack((np.asarray(plydata.elements[0]["x"]), np.asarray(
        plydata.elements[0]["y"]), np.asarray(plydata.elements[0]["z"])),
                   axis=1)
    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
    opacities = np.clip(opacities, 0, 1)
    normals = np.zeros_like(xyz)
    rgb = np.zeros((xyz.shape[0], 3), dtype=np.float32)
    rgb[:,0] = np.clip(np.asarray(plydata.elements[0]["f_dc_0"]) * 0.28209479177387814 + 0.5, 0, 1) * 255
    rgb[:,1] = np.clip(np.asarray(plydata.elements[0]["f_dc_1"]) * 0.28209479177387814 + 0.5, 0, 1) * 255
    rgb[:,2] = np.clip(np.asarray(plydata.elements[0]["f_dc_2"]) * 0.28209479177387814 + 0.5, 0, 1) * 255

    mkdir_p(os.path.dirname(output_path))

    l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    dtype_full = [(attribute, 'f4') for attribute in l]
    dtype_full += [('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    # attributes = np.concatenate((xyz, normals), axis=1)
    # print(list(map(tuple, attributes)))
    # elements[:] = list(map(tuple, attributes))
    for i in range(xyz.shape[0]):
        elements[i] = (
            xyz[i][0],
            xyz[i][1],
            xyz[i][2],
            normals[i][0],
            normals[i][1],
            normals[i][2],
            rgb[i][0],
            rgb[i][1],
            rgb[i][2],
        )
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(output_path)


input_path = sys.argv[1]
output_path = sys.argv[2]
convert_ply(input_path, output_path)
