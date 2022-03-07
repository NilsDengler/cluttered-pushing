import numpy as np
from numba import njit, prange

def transform_is_valid(t, tolerance=1e-3):
    '''
    In:
        t: Numpy array [4x4] that is an transform candidate.
        tolerance: maximum absolute difference for two numbers to be considered close enough to each other.
    Out:
        bool: True if array is a valid transform else False.
    Purpose:
        Check if array is a valid transform.
    '''
    if t.shape != (4,4):
        return False

    rtr = np.matmul(t[:3, :3].T, t[:3, :3])
    rrt = np.matmul(t[:3, :3], t[:3, :3].T)

    inverse_check = np.isclose(np.eye(3), rtr, atol=tolerance).all() and np.isclose(np.eye(3), rrt, atol=tolerance).all()
    det_check = np.isclose(np.linalg.det(t[:3, :3]), 1.0, atol=tolerance).all()
    last_row_check = np.isclose(t[3, :3], np.zeros((1, 3)), atol=tolerance).all() and np.isclose(t[3, 3], 1.0, atol=tolerance).all()

    return inverse_check and det_check and last_row_check

def transform_world_point_to_camera_space(point, view_matrix):
    '''
        In: point: Numpy array [1x3] representing 3Dpoint in world coordinates
            view_matrix: Numpy array [3x3] representing view_matrix.
        Out: Numpy array [Nx2] representing the projection of the point3 on the image plane.
        Purpose: Project a 3D world point into camera space.
        '''
    ps_homogeneous = np.append(point, 1.)
    ps_transformed = np.dot(np.array(view_matrix).reshape(4, 4).T, ps_homogeneous.T).T
    return ps_transformed[:3]

def transform_camera_space_to_world_point(point, view_matrix):
    '''
        In: point: Numpy array [1x3] representing 3Dpoint in world coordinates
            view_matrix: Numpy array [3x3] representing view_matrix.
        Out: Numpy array [Nx2] representing the projection of the point3 on the image plane.
        Purpose: Project a 3D world point into camera space.
        '''
    ps_homogeneous = np.append(point, 1.)
    ps_transformed = np.dot(np.linalg.inv(np.array(view_matrix).reshape(4, 4).T), ps_homogeneous.T).T
    return ps_transformed[:3]



def get_cam_world_pose_from_view_matrix(view_matrix):
    '''
    out: camera world pose
    details: Convert camera view matrix to pose matrix
    '''
    cam_pose = np.linalg.inv(np.array(view_matrix).reshape(4, 4).T)
    cam_pose[:, 1:3] = -cam_pose[:, 1:3]
    return cam_pose

def get_point_from_pixel(pc, pixel):
    return pc[pixel[1],pixel[0]]

def transform_3Dpoint(t, point):
    '''
    In:
        t: Numpy array [4x4] to represent a transform
        ps: point3s represented as a numpy array [Nx3], where each row is a point.
    Out:
        Transformed point3s as a numpy array [Nx3].
    Purpose:
        Transfrom point from one space to another.
    '''
    #if not transform_is_valid(t):
    #    raise ValueError('Invalid input transform t')
    # convert to homogeneous
    ps_homogeneous = np.append(point, 1.)
    ps_transformed = np.dot(t, ps_homogeneous.T).T
    return ps_transformed[:3]

def camera_to_image_points(intrinsics, camera_points):
    '''
    In: intrinsics: Numpy array [3x3] containing camera pinhole intrinsics.
        p_camera: Numpy array [Nx3] representing point3s in camera coordinates.
    Out: Numpy array [Nx2] representing the projection of the point3 on the image plane.
    Purpose: Project a point3 in camera space to the image plane.
    '''
    if intrinsics.shape != (3, 3):
        raise ValueError('Invalid input intrinsics')
    if len(camera_points.shape) != 2 or camera_points.shape[1] != 3:
        raise ValueError('Invalid camera point')

    u0 = intrinsics[0, 2]
    v0 = intrinsics[1, 2]
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]

    image_coordinates = np.empty((camera_points.shape[0], 2), dtype=np.int64)
    for i in prange(camera_points.shape[0]):
        image_coordinates[i, 0] = int(np.round((camera_points[i, 0] * fx / camera_points[i, 2]) + u0))
        image_coordinates[i, 1] = int(np.round((camera_points[i, 1] * fy / camera_points[i, 2]) + v0))

    return image_coordinates

def camera_to_image_point(intrinsics, camera_points):
    '''
    In:
        intrinsics: Numpy array [3x3] containing camera pinhole intrinsics.
        p_camera: Numpy array [1x3] representing point3s in camera coordinates.
    Out:
        Numpy array [1x2] representing the projection of the point3 on the image plane.
    Purpose:
        Project a point3 in camera space to the image plane.
    '''
    if intrinsics.shape != (3, 3):
        raise ValueError('Invalid input intrinsics')

    u0 = intrinsics[0, 2]
    v0 = intrinsics[1, 2]
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]

    image_coordinates = np.empty((1, 2), dtype=np.int64)
    image_coordinates[0,0] = int(np.round((camera_points[0] * fx / camera_points[2]) + u0))
    image_coordinates[0,1] = int(np.round((camera_points[1] * fy / camera_points[2]) + v0))

    return image_coordinates