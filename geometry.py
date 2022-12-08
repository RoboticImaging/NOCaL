"""
This code was originally use in the light field network.
See here: https://github.com/vsitzmann/light-field-networks

Modified (Ryan Griffiths) to inlcude other ray parameterization.
"""

import torch
from torch.nn import functional as F


def get_ray_origin(cam2world):
    """
    Get the origin of the ray
    """
    return cam2world[..., :3, 3]


def plucker_embedding(cam2world, uv, intrinsics, distortion):
    """
    Computes the plucker coordinates from batched cam2world & intrinsics matrices, as well as pixel coordinates
    :param cam2world: (b, 4, 4)
    :param intrinsics: k matrix(b, 4, 4)
    :param uv: pixel points in image plane(b, n, 2)
    :param distortion: distortion values to apply to x,y
    :return plucker: List of rays in plucker formate
    """
    ray_dirs = get_ray_directions(uv, cam2world=cam2world, intrinsics=intrinsics, distortion=distortion)
    cam_pos = get_ray_origin(cam2world)
    cam_pos = cam_pos[..., None, :].expand(list(uv.shape[:-1]) + [3])

    # https://www.euclideanspace.com/maths/geometry/elements/line/plucker/index.htm
    # https://web.cs.iastate.edu/~cs577/handouts/plucker-coordinates.pdf
    cross = torch.cross(cam_pos, ray_dirs, dim=-1)
    # TODO: Normalise Cross moments
    # cross = F.normalize(cross)
    plucker = torch.cat((ray_dirs, cross), dim=-1)
    return plucker


def plenoptic_embedding(cam2world, uv, intrinsics, distortion):
    """
    Computes the plenotic coordinates from batched cam2world & intrinsics matrices, as well as pixel coordinates
    :param cam2world: (b, 4, 4)
    :param intrinsics: k matrix(b, 4, 4)
    :param uv: pixel points in image plane(b, n, 2)
    :param distortion: distortion values to apply to x,y
    :return plucker: List of rays in plenotic formate
    """
    ray_dirs = get_ray_directions(uv, cam2world=cam2world, intrinsics=intrinsics, distortion=distortion)
    cam_pos = get_ray_origin(cam2world)
    cam_pos = cam_pos[..., None, :].expand(list(uv.shape[:-1]) + [3])

    plenoptic = torch.cat((cam_pos, ray_dirs), dim=-1)
    return plenoptic


def planes_embedding(cam2world, uv, intrinsics, distortion):
    """
    Computes the two plane parameterization from batched cam2world & intrinsics matrices, as well as pixel coordinates
    :param cam2world: (b, 4, 4)
    :param intrinsics: k matrix(b, 4, 4)
    :param uv: pixel points in image plane(b, n, 2)
    :param distortion: distortion values to apply to x,y
    :return plucker: List of rays in 2-plane formate
    """
    plenoptic = plenoptic_embedding(cam2world, uv, intrinsics, distortion)
    planes = two_plane(plenoptic)
    return planes


def closest_to_origin(plucker_coord):
    """Computes the point on a plucker line closest to the origin."""
    direction = plucker_coord[..., :3]
    moment = plucker_coord[..., 3:]
    return torch.cross(direction, moment, dim=-1)


def plucker_sd(plucker_coord, point_coord):
    """Computes the signed distance of a point on a line to the point closest to the origin
    (like a local coordinate system on a plucker line)"""
    # Get closest point to origin along plucker line.
    plucker_origin = closest_to_origin(plucker_coord)

    # Compute signed distance: offset times dot product.
    direction = plucker_coord[..., :3]
    diff = point_coord - plucker_origin
    signed_distance = torch.einsum('...j,...j', diff, direction)
    return signed_distance[..., None]


def get_relative_rotation_matrix(vector_1, vector_2):
    "https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d"
    a_plus_b = vector_1 + vector_2
    outer = a_plus_b.unsqueeze(-2) * a_plus_b.unsqueeze(-1)
    dot = torch.einsum('...j,...j', a_plus_b, a_plus_b)[..., None, None]
    R = 2 * outer/dot - torch.eye(3)[None, None, None].cuda()
    return R


def plucker_reciprocal_product(line_1, line_2):
    """Computes the reciprocal product between plucker coordinates. See:
    https://faculty.sites.iastate.edu/jia/files/inline-files/plucker-coordinates.pdf"""
    return torch.einsum('...j,...j', line_1[..., :3], line_2[..., 3:]) + \
           torch.einsum('...j,...j', line_2[..., :3], line_1[..., 3:])


def plucker_distance(line_1, line_2):
    """Computes the distance between the two closest points on lines parameterized as plucker coordinates."""
    line_1_dir, line_2_dir = torch.broadcast_tensors(line_1[..., :3], line_2[..., :3])
    direction_cross = torch.cross(line_1_dir, line_2_dir, dim=-1)
    # https://web.cs.iastate.edu/~cs577/handouts/plucker-coordinates.pdf
    return torch.abs(plucker_reciprocal_product(line_1, line_2))/direction_cross.norm(dim=-1)


def parse_intrinsics(intrinsics):
    fx = intrinsics[..., 0, :1]
    fy = intrinsics[..., 1, 1:2]
    cx = intrinsics[..., 0, 2:3]
    cy = intrinsics[..., 1, 2:3]
    return fx, fy, cx, cy


def expand_as(x, y):
    if len(x.shape) == len(y.shape):
        return x

    for i in range(len(y.shape) - len(x.shape)):
        x = x.unsqueeze(-1)

    return x


def lift(x, y, z, intrinsics, homogeneous=False):
    """
    :param self:
    :param x: Shape (batch_size, num_points)
    :param y:
    :param z:
    :param intrinsics:
    :return:
    """
    fx, fy, cx, cy = parse_intrinsics(intrinsics)

    x_lift = (x - expand_as(cx, x)) / expand_as(fx, x) * z
    y_lift = (y - expand_as(cy, y)) / expand_as(fy, y) * z

    if homogeneous:
        return torch.stack((x_lift, y_lift, z, torch.ones_like(z).to(x.device)), dim=-1)
    else:
        return torch.stack((x_lift, y_lift, z), dim=-1)


def project(x, y, z, intrinsics):
    """
    :param self:
    :param x: Shape (batch_size, num_points)
    :param y:
    :param z:
    :param intrinsics:
    :return:
    """
    fx, fy, cx, cy = parse_intrinsics(intrinsics)

    x_proj = expand_as(fx, x) * x / z + expand_as(cx, x)
    y_proj = expand_as(fy, y) * y / z + expand_as(cy, y)

    return torch.stack((x_proj, y_proj, z), dim=-1)


def world_from_xy_depth(xy, depth, cam2world, intrinsics, distortion):
    batch_size, *_ = cam2world.shape

    x_cam = xy[..., 0]
    y_cam = xy[..., 1]
    z_cam = depth

    pixel_points_cam = lift(x_cam, y_cam, z_cam, intrinsics=intrinsics, homogeneous=True)
    if distortion is not None:
        pixel_points_cam[..., :2] += distortion
    world_coords = torch.einsum('b...ij,b...kj->b...ki', cam2world, pixel_points_cam)[..., :3]
    return world_coords


def project_point_on_line(projection_point, line_direction, point_on_line):
    dot = torch.einsum('...j,...j', projection_point-point_on_line, line_direction)
    return point_on_line + dot[..., None] * line_direction


def get_ray_directions(xy, cam2world, intrinsics, distortion):
    z_cam = torch.ones(xy.shape[:-1]).to(xy.device)
    pixel_points = world_from_xy_depth(xy, z_cam, intrinsics=intrinsics, cam2world=cam2world, distortion=distortion)  # (batch, num_samples, 3)

    cam_pos = cam2world[..., :3, 3]
    ray_dirs = pixel_points - cam_pos[..., None, :]  # (batch, num_samples, 3)
    ray_dirs = F.normalize(ray_dirs, dim=-1)
    return ray_dirs


def depth_from_world(world_coords, cam2world):
    batch_size, num_samples, _ = world_coords.shape

    points_hom = torch.cat((world_coords, torch.ones((batch_size, num_samples, 1)).cuda()),
                           dim=2)  # (batch, num_samples, 4)

    # permute for bmm
    points_hom = points_hom.permute(0, 2, 1)

    points_cam = torch.inverse(cam2world).bmm(points_hom)  # (batch, 4, num_samples)
    depth = points_cam[:, 2, :][:, :, None]  # (batch, num_samples, 1)
    return depth


def ray_sphere_intersect(ray_origin, ray_dir, sphere_center=None, radius=1):
    if sphere_center is None:
        sphere_center = torch.zeros_like(ray_origin)

    ray_dir_dot_origin = torch.einsum('b...jd,b...id->b...ji', ray_dir, ray_origin - sphere_center)
    discrim = torch.sqrt( ray_dir_dot_origin**2 - (torch.einsum('b...id,b...id->b...i', ray_origin-sphere_center, ray_origin - sphere_center)[..., None] - radius**2) )

    t0 = - ray_dir_dot_origin + discrim
    t1 = - ray_dir_dot_origin - discrim
    return ray_origin + t0*ray_dir, ray_origin + t1*ray_dir


def intersect_axis_plane(rays, val, dim, exclude=False, normalize=False):
    rays_o, rays_d = rays[..., :3], rays[..., 3:6]
    t = (val - rays_o[..., dim]) / rays_d[..., dim]
    loc = rays_o + t[..., None] * rays_d

    t = torch.where(
        torch.isnan(t),
        torch.zeros_like(t),
        t
    )

    if exclude:
        loc = torch.cat(
            [
                loc[..., :dim],
                loc[..., dim+1:],
            ],
            dim=-1
        )

        t = torch.stack([t, t], dim=-1)
    else:
        t = torch.stack([t, t, t], dim=-1)

    if normalize:
        loc = loc / torch.maximum(torch.abs(val), torch.ones_like(val)).unsqueeze(-1)

    return loc, t


def two_plane(rays):
    rays = rays

    isect_pts_1, _ = intersect_axis_plane(
        rays, -0.5, 2, exclude=True
    )

    isect_pts_2, _ = intersect_axis_plane(
        rays, 0.5, 2, exclude=True
    )

    param_rays = torch.cat([isect_pts_1, isect_pts_2], dim=-1)
    param_rays = param_rays

    return param_rays