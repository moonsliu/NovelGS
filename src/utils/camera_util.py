import numpy as np
import torch, math
import torch.nn.functional as F
from PIL import Image

def pad_camera_extrinsics_4x4(extrinsics):
    if extrinsics.shape[-2] == 4:
        return extrinsics
    padding = torch.tensor([[0, 0, 0, 1]]).to(extrinsics)
    if extrinsics.ndim == 3:
        padding = padding.unsqueeze(0).repeat(extrinsics.shape[0], 1, 1)
    extrinsics = torch.cat([extrinsics, padding], dim=-2)
    return extrinsics


def center_looking_at_camera_pose(camera_position: torch.Tensor, look_at: torch.Tensor = None, up_world: torch.Tensor = None):
    """
    Create OpenGL camera extrinsics from camera locations and look-at position.

    camera_position: (M, 3) or (3,)
    look_at: (3)
    up_world: (3)
    return: (M, 3, 4) or (3, 4)
    """
    # by default, looking at the origin and world up is z-axis
    if look_at is None:
        look_at = torch.tensor([0, 0, 0], dtype=torch.float32)
    if up_world is None:
        up_world = torch.tensor([0, 0, 1], dtype=torch.float32)
    if camera_position.ndim == 2:
        look_at = look_at.unsqueeze(0).repeat(camera_position.shape[0], 1)
        up_world = up_world.unsqueeze(0).repeat(camera_position.shape[0], 1)

    # OpenGL camera: z-backward, x-right, y-up
    z_axis = camera_position - look_at
    z_axis = F.normalize(z_axis, dim=-1).float()
    x_axis = torch.linalg.cross(up_world, z_axis, dim=-1)
    x_axis = F.normalize(x_axis, dim=-1).float()
    y_axis = torch.linalg.cross(z_axis, x_axis, dim=-1)
    y_axis = F.normalize(y_axis, dim=-1).float()

    extrinsics = torch.stack([x_axis, y_axis, z_axis, camera_position], dim=-1)
    extrinsics = pad_camera_extrinsics_4x4(extrinsics)
    return extrinsics


def spherical_camera_pose(azimuths: np.ndarray, elevations: np.ndarray, radius=2.5):
    azimuths = np.deg2rad(azimuths)
    elevations = np.deg2rad(elevations)

    xs = radius * np.cos(elevations) * np.cos(azimuths)
    ys = radius * np.cos(elevations) * np.sin(azimuths)
    zs = radius * np.sin(elevations)

    cam_locations = np.stack([xs, ys, zs], axis=-1)
    cam_locations = torch.from_numpy(cam_locations).float()

    c2ws = center_looking_at_camera_pose(cam_locations)
    return c2ws


def get_circular_camera_poses(M=120, radius=2.5, elevation=30.0):
    # M: number of circular views
    # radius: camera dist to center
    # elevation: elevation degrees of the camera
    # return: (M, 4, 4)
    assert M > 0 and radius > 0

    elevation = np.deg2rad(elevation)

    camera_positions = []
    for i in range(M):
        azimuth = 2 * np.pi * i / M
        x = radius * np.cos(elevation) * np.cos(azimuth)
        y = radius * np.cos(elevation) * np.sin(azimuth)
        z = radius * np.sin(elevation)
        camera_positions.append([x, y, z])
    camera_positions = np.array(camera_positions)
    camera_positions = torch.from_numpy(camera_positions).float()
    extrinsics = center_looking_at_camera_pose(camera_positions)
    return extrinsics


def FOV_to_intrinsics(fov, device='cpu'):
    """
    Creates a 3x3 camera intrinsics matrix from the camera field of view, specified in degrees.
    Note the intrinsics are returned as normalized by image size, rather than in pixel units.
    Assumes principal point is at image center.
    """
    focal_length = 0.5 / np.tan(np.deg2rad(fov) * 0.5)
    intrinsics = torch.tensor([[focal_length, 0, 0.5], [0, focal_length, 0.5], [0, 0, 1]], device=device)
    return intrinsics

def normalize_cameras(extrinsics, canonical_camera_position: torch.Tensor = None, cond_camera_indices: torch.tensor = None, camera_system: str = 'opencv'):
    """
    Normalize the first camera to the canonical camera position, and transform other cameras accordingly.

    extrinsics: (N, 4, 4)
    """
    if canonical_camera_position is None:
        canonical_camera_position = torch.tensor([[0, -2, 0]]).float()
    if cond_camera_indices is None:
        cond_camera_indices = torch.arange(4).long()
    assert camera_system in ['opencv', 'blender']

    canonical_distance = canonical_camera_position.norm()

    # compute conditional camera distances
    cond_extrinsics = extrinsics[cond_camera_indices]
    cond_camera_distances = cond_extrinsics[:, :3, 3].norm(dim=-1, keepdim=False)

    # randomly choose a camera farther than 2.0 as the canonical view
    valid_indices = torch.where(cond_camera_distances >= canonical_distance)[0]
    if len(valid_indices) == 0:
        canonical_index = torch.argmax(cond_camera_distances).item()
    else:
        canonical_index = valid_indices[torch.randint(len(valid_indices), (1,)).item()].item()

    # scale camera distances
    scale = canonical_distance / cond_camera_distances[canonical_index]
    extrinsics[:, :3, 3] = extrinsics[:, :3, 3] * scale


    # rotate and translate all cameras
    cond_extrinsic = extrinsics[cond_camera_indices][canonical_index].unsqueeze(0)
    if camera_system == 'opencv':
        canonical_extrinsic = create_opencv_camera(canonical_camera_position)
    else:
        canonical_extrinsic = create_blender_camera(canonical_camera_position)

    transform_matrix = torch.matmul(canonical_extrinsic, torch.linalg.inv(cond_extrinsic))
    normalized_extrinsics = torch.matmul(transform_matrix, extrinsics)
    return normalized_extrinsics, canonical_index, transform_matrix


def normalize_vecs(vectors: torch.Tensor) -> torch.Tensor:
    """
    Normalize vector lengths.
    """
    return vectors / (torch.norm(vectors, dim=-1, keepdim=True))

def create_opencv_camera(camera_position: torch.Tensor, look_at: torch.Tensor = None, up_world: torch.Tensor = None):
    """
    Create OpenCV camera extrinsics from camera locations and look-at position.

    camera_position: (M, 3) or (3,)
    look_at: (3)
    up_world: (3)
    return: (M, 3, 4) or (3, 4)
    """
    # by default, looking at the origin and world up is z-axis
    if look_at is None:
        look_at = torch.tensor([0, 0, 0], dtype=torch.float32)
    if up_world is None:
        up_world = torch.tensor([0, 0, 1], dtype=torch.float32)
    if camera_position.ndim == 2:
        look_at = look_at.unsqueeze(0).repeat(camera_position.shape[0], 1)
        up_world = up_world.unsqueeze(0).repeat(camera_position.shape[0], 1)

    # OpenCV camera: z-forward, x-right, y-down
    z_axis = look_at - camera_position
    z_axis = normalize_vecs(z_axis)
    x_axis = torch.cross(z_axis, up_world)
    x_axis = normalize_vecs(x_axis)
    y_axis = torch.cross(z_axis, x_axis)
    y_axis = normalize_vecs(y_axis)

    extrinsics = torch.stack([x_axis, y_axis, z_axis, camera_position], dim=-1)
    extrinsics = pad_camera_extrinsics_4x4(extrinsics)
    return extrinsics

def create_blender_camera(camera_position: torch.Tensor, look_at: torch.Tensor = None, up_world: torch.Tensor = None):
    """
    Create OpenGL camera extrinsics from camera locations and look-at position.

    camera_position: (M, 3) or (3,)
    look_at: (3)
    up_world: (3)
    return: (M, 3, 4) or (3, 4)
    """
    # by default, looking at the origin and world up is z-axis
    if look_at is None:
        look_at = torch.tensor([0, 0, 0], dtype=torch.float32)
    if up_world is None:
        up_world = torch.tensor([0, 0, 1], dtype=torch.float32)
    if camera_position.ndim == 2:
        look_at = look_at.unsqueeze(0).repeat(camera_position.shape[0], 1)
        up_world = up_world.unsqueeze(0).repeat(camera_position.shape[0], 1)

    # OpenGL camera: z-backward, x-right, y-up
    z_axis = camera_position - look_at
    z_axis = F.normalize(z_axis, dim=-1).float()
    x_axis = torch.linalg.cross(up_world, z_axis, dim=-1)
    x_axis = F.normalize(x_axis, dim=-1).float()
    y_axis = torch.linalg.cross(z_axis, x_axis, dim=-1)
    y_axis = F.normalize(y_axis, dim=-1).float()

    extrinsics = torch.stack([x_axis, y_axis, z_axis, camera_position], dim=-1)
    extrinsics = pad_camera_extrinsics_4x4(extrinsics)
    return extrinsics

def get_sv3d_input_cameras(batch_size=1, radius=4.0, fov=30.0):
    """
    Get the input camera parameters.
    """
    azimuths = np.array([0, 90, 180, 270, 315]).astype(float)
    elevations = np.array([20, 20, 20, 20, -20]).astype(float)
    c2ws = []
    for elevation, azimuth in zip(elevations, azimuths):
        sph = np.array([elevation, azimuth, radius])
        c2w = spherical_to_cartesian(sph, opengl=True)
        c2ws.append(c2w)
    c2ws = torch.from_numpy(np.stack(c2ws))
    c2ws[:, :3, 1:3] *= -1
    c2ws, _ = normalize_cameras(c2ws, cond_camera_indices=[0])
    c2ws = c2ws.float().flatten(-2)

    Ks = FOV_to_intrinsics(fov).unsqueeze(0).repeat(5, 1, 1).float().flatten(-2)

    extrinsics = c2ws
    intrinsics = torch.stack([Ks[:, 0], Ks[:, 4], Ks[:, 2], Ks[:, 5]], dim=-1)
    cameras = torch.cat([extrinsics, intrinsics], dim=-1)

    return cameras.unsqueeze(0).repeat(batch_size, 1, 1)


def get_zero123plus_input_cameras(batch_size=1, radius=4.0, fov=30.0):
    """
    Get the input camera parameters.
    """
    azimuths = np.array([30, 90, 150, 210, 270, 330]).astype(float)
    elevations = np.array([20, -10, 20, -10, 20, -10]).astype(float)
    c2ws = []
    for elevation, azimuth in zip(elevations, azimuths):
        sph = np.array([elevation, azimuth, radius])
        c2w = spherical_to_cartesian(sph, opengl=True)
        c2ws.append(c2w)
    c2ws = torch.from_numpy(np.stack(c2ws))
    c2ws, _, transform_matrix = normalize_cameras(c2ws, cond_camera_indices=[0], camera_system='blender')
    c2ws[:, :3, 1:3] *= -1
    c2ws = c2ws.float().flatten(-2)

    Ks = FOV_to_intrinsics(fov).unsqueeze(0).repeat(6, 1, 1).float().flatten(-2)

    extrinsics = c2ws
    intrinsics = torch.stack([Ks[:, 0], Ks[:, 4], Ks[:, 2], Ks[:, 5]], dim=-1)
    cameras = torch.cat([extrinsics, intrinsics], dim=-1)

    return cameras.unsqueeze(0).repeat(batch_size, 1, 1), transform_matrix


def get_zero123plus_all_cameras(batch_size=1, radius=4.0, fov=30.0, orbit_camera=None):
    """
    Get the input camera parameters.
    """
    azimuths = np.array([orbit_camera[1][0], 30 + orbit_camera[1][0], 90 + orbit_camera[1][0], 150 + orbit_camera[1][0], 210 + orbit_camera[1][0], 270 + orbit_camera[1][0], 330 + orbit_camera[1][0]]).astype(float)
    elevations = np.array([orbit_camera[0][0], 20, -10, 20, -10, 20, -10]).astype(float)
    radiuss = np.array([orbit_camera[2][0], 4, 4, 4, 4, 4, 4]).astype(float)
    c2ws = []
    for elevation, azimuth, radius in zip(elevations, azimuths, radiuss):
        sph = np.array([elevation, azimuth, radius])
        c2w = spherical_to_cartesian(sph, opengl=True)
        c2ws.append(c2w)
    c2ws = torch.from_numpy(np.stack(c2ws))
    c2ws[:, :3, 1:3] *= -1
    c2ws, _ = normalize_cameras(c2ws, cond_camera_indices=[0])
    c2ws = c2ws.float().flatten(-2)
    extrinsics = c2ws

    Ks = FOV_to_intrinsics(fov).unsqueeze(0).repeat(7, 1, 1).float().flatten(-2)

    intrinsics = torch.stack([Ks[:, 0], Ks[:, 4], Ks[:, 2], Ks[:, 5]], dim=-1)
    cameras = torch.cat([extrinsics, intrinsics], dim=-1)

    return cameras.unsqueeze(0).repeat(batch_size, 1, 1)

def pad_image_to_fit_fov(image, new_fov, old_fov):
    img = Image.fromarray(image)

    scale_factor = math.tan(np.deg2rad(new_fov/2)) / math.tan(np.deg2rad(old_fov/2))

    # Calculate the new size
    new_size = (int(img.size[0] * scale_factor), int(img.size[1] * scale_factor))

    # Calculate padding
    pad_width = (new_size[0]-img.size[0]) // 2
    pad_height = (new_size[1] - img.size[1]) // 2

    # Create padding
    padding = (pad_width, pad_height, pad_width+img.size[0], pad_height+img.size[1])

    # Pad the image
    img_padded = Image.new(img.mode, (new_size[0], new_size[1]), color='white')
    img_padded.paste(img, padding)
    img_padded = np.array(img_padded)
    return img_padded


def c2w_to_elu(c2w):

    w2c = np.linalg.inv(c2w)
    eye = c2w[:3, 3]
    lookat_dir = -w2c[2, :3]
    lookat = eye + lookat_dir
    up = w2c[1, :3]

    return eye, lookat, up

def elu_to_c2w(eye, lookat, up):

    if isinstance(eye, list):
        eye = np.array(eye)
    if isinstance(lookat, list):
        lookat = np.array(lookat)
    if isinstance(up, list):
        up = np.array(up)

    l = eye - lookat
    if np.linalg.norm(l) < 1e-8:
        l[-1] = 1
    l = l / np.linalg.norm(l)

    s = np.cross(l, up)
    if np.linalg.norm(s) < 1e-8:
        s[0] = 1
    s = s / np.linalg.norm(s)
    uu = np.cross(s, l)

    rot = np.eye(3)
    rot[0, :] = -s
    rot[1, :] = uu
    rot[2, :] = l

    c2w = np.eye(4)
    c2w[:3, :3] = rot.T
    c2w[:3, 3] = eye

    return c2w

def cartesian_to_spherical(xyz):

    xy = xyz[0]**2 + xyz[1]**2
    radius = np.sqrt(xy + xyz[2]**2)
    theta = np.arctan2(xyz[2], np.sqrt(xy))
    azimuth = np.arctan2(xyz[1], xyz[0])

    return np.array([np.rad2deg(theta), np.rad2deg(azimuth), radius])

def spherical_to_cartesian(sph, opengl=True):

    theta, azimuth, radius = sph
    theta, azimuth, radius = np.deg2rad(theta), np.deg2rad(azimuth), radius

    x = radius * np.cos(theta) * np.cos(azimuth)
    y = radius * np.cos(theta) * np.sin(azimuth)
    z = radius * np.sin(theta)
    # if target is None:

    target = np.zeros([3], dtype=np.float32)
    campos = np.array([x, y, z]) + target  # [3]
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = look_at(campos, target, opengl) #elu_to_c2w(campos, target, )
    T[:3, 3] = campos

    return T

def safe_normalize(x, eps=1e-20):
    """normalize an array (along the last dim).

    Args:
        x (Union[Tensor, ndarray]): x, [..., C]
        eps (float, optional): eps. Defaults to 1e-20.

    Returns:
        Union[Tensor, ndarray]: normalized x, [..., C]
    """

    return x / length(x, eps)


def length(x, eps=1e-20):
    """length of an array (along the last dim).

    Args:
        x (Union[Tensor, ndarray]): x, [..., C]
        eps (float, optional): eps. Defaults to 1e-20.

    Returns:
        Union[Tensor, ndarray]: length, [..., 1]
    """
    if isinstance(x, np.ndarray):
        return np.sqrt(np.maximum(np.sum(x * x, axis=-1, keepdims=True), eps))
    else:
        return torch.sqrt(torch.clamp(dot(x, x), min=eps))

def dot(x, y):
    """dot product (along the last dim).

    Args:
        x (Union[Tensor, ndarray]): x, [..., C]
        y (Union[Tensor, ndarray]): y, [..., C]

    Returns:
        Union[Tensor, ndarray]: x dot y, [..., 1]
    """
    if isinstance(x, np.ndarray):
        return np.sum(x * y, -1, keepdims=True)
    else:
        return torch.sum(x * y, -1, keepdim=True)

def look_at(campos, target, opengl=True):
    """construct pose rotation matrix by look-at.

    Args:
        campos (np.ndarray): camera position, float [3]
        target (np.ndarray): look at target, float [3]
        opengl (bool, optional): whether use opengl camera convention (forward direction is target --> camera). Defaults to True.

    Returns:
        np.ndarray: the camera pose rotation matrix, float [3, 3], normalized.
    """

    if not opengl:
        # forward is camera --> target
        forward_vector = safe_normalize(target - campos)
        up_vector = np.array([0, 0, 1], dtype=np.float32)
        right_vector = safe_normalize(np.cross(forward_vector, up_vector))
        up_vector = safe_normalize(np.cross(right_vector, forward_vector))
    else:
        # forward is target --> camera
        forward_vector = safe_normalize(campos - target)
        up_vector = np.array([0, 0, 1], dtype=np.float32)
        right_vector = safe_normalize(np.cross(up_vector, forward_vector))
        up_vector = safe_normalize(np.cross(forward_vector, right_vector))
    R = np.stack([right_vector, up_vector, forward_vector], axis=1)
    return R