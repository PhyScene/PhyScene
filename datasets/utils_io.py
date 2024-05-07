import numpy as np
from plyfile import PlyElement, PlyData

def export_pointcloud(vertices, out_file, as_text=True):
    assert(vertices.shape[1] == 3)
    vertices = vertices.astype(np.float32)
    vertices = np.ascontiguousarray(vertices)
    vector_dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
    vertices = vertices.view(dtype=vector_dtype).flatten()
    plyel = PlyElement.describe(vertices, 'vertex')
    plydata = PlyData([plyel], text=as_text)
    plydata.write(out_file)

def export_pointcloud_with_rgb(vertices_in, out_file, colors, as_text=True):
    assert(vertices_in.shape[1] == 3)

    vertices = np.empty(vertices_in.shape[0], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    vertices['x'] = vertices_in[:,0].astype('f4')
    vertices['y'] = vertices_in[:,1].astype('f4')
    vertices['z'] = vertices_in[:,2].astype('f4')
    vertices['red'] = colors[:,0].astype('u1')
    vertices['green'] = colors[:,1].astype('u1')
    vertices['blue'] = colors[:,2].astype('u1')
    
    # colors = colors.view(dtype=color_dtype).flatten()
    plyel = PlyElement.describe(vertices, 'vertex')
    plydata = PlyData([plyel], text=as_text)
    plydata.write(out_file)

# def export_pointcloud_with_rgb(vertices, out_file, colors, as_text=True):
#     assert(vertices.shape[1] == 3)
#     vertices = np.concatenate((vertices,colors[:,:3]),axis = 1)
#     # vertices = vertices.astype(np.float32)
#     vertices = np.ascontiguousarray(vertices)
#     # colors = np.ascontiguousarray(colors[:,:3])


#     vector_dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
#     # color_dtype = [('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
#     vertices = vertices.view(dtype=vector_dtype).flatten()

    
#     # colors = colors.view(dtype=color_dtype).flatten()
#     plyel = PlyElement.describe(vertices, 'vertex')
#     plydata = PlyData([plyel], text=as_text)
#     plydata.write(out_file)

def load_pointcloud(in_file):
    plydata = PlyData.read(in_file)
    vertices = np.stack([
        plydata['vertex']['x'],
        plydata['vertex']['y'],
        plydata['vertex']['z']
    ], axis=1)
    return vertices