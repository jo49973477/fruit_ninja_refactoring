import numpy as np
import torch
import math


def generate_rotation_matrix(degree: float, axis: int, device="cpu") -> torch.Tensor:

    rad = torch.deg2rad(torch.tensor(degree, device=device))
    c, s = torch.cos(rad), torch.sin(rad)

    if axis == 0:  # X-axis
        return torch.tensor([[1, 0, 0], [0, c, -s], [0, s, c]])
    elif axis == 1:  # Y-axis
        return torch.tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    elif axis == 2:  # Z-axis
        return torch.tensor([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    else:
        raise ValueError(
            "💀The only axis which can be selected is 0, 1, 2, You dumbass!"
        )


def apply_cov_rotation(cov_tensor, rotation_matrix):
    rotated = torch.matmul(cov_tensor, rotation_matrix.T)
    rotated = torch.matmul(rotation_matrix, rotated)
    return rotated


def get_mat_from_upper(upper_mat: torch.Tensor) -> torch.Tensor:
    """[N, 6] 텐서를 [N, 3, 3] 대칭 행렬로 뻥튀기!"""

    # 0, 1, 2, 3, 4, 5 인덱스가 3x3 행렬의 어디에 들어가야 할지 '설계도'만 짜주면 됨!
    # [0,1,2]
    # [1,3,4]
    # [2,4,5]
    mapping = [0, 1, 2, 1, 3, 4, 2, 4, 5]

    # 빈 텐서(zeros) 만들 필요 없이, 맵핑대로 뽑아서 모양만(view) 바꿔버려!
    return upper_mat[:, mapping].view(-1, 3, 3)


def get_upper_from_mat(mat: torch.Tensor) -> torch.Tensor:
    """[N, 3, 3] 행렬에서 상삼각 원소 6개를 우아하게 추출!"""

    i, j = torch.triu_indices(3, 3, device=mat.device)
    return mat[:, i, j]


def get_upper_from_mat(mat: torch.Tensor) -> torch.Tensor:
    """[N, 3, 3] 행렬에서 위쪽 삼각형(Upper Triangular) 6개만 쏙 뽑아먹기!"""

    # 파이토치 내장 함수 찬스!! "3x3 행렬에서 위쪽 삼각형 인덱스 좀 가져와봐!"
    i, j = torch.triu_indices(3, 3)

    # 인덱스 던져주면 알아서 6개 쏙 뽑아옴! (결과: [N, 6])
    return mat[:, i, j]


def apply_cov_rotations(upper_cov_tensor, rotation_matrices):
    cov_tensor = get_mat_from_upper(upper_cov_tensor)
    for i in range(len(rotation_matrices)):
        cov_tensor = apply_cov_rotation(cov_tensor, rotation_matrices[i])
    return get_upper_from_mat(cov_tensor)


def shift2center111(position_tensor):
    tensor111 = torch.tensor([1.0, 1.0, 1.0], device="cuda")
    return position_tensor + tensor111


def build_cov3D_from_scales_quats(
    scales: torch.Tensor, quats: torch.Tensor
) -> torch.Tensor:
    """
    scales: [N, 3] (log-scale saved in ply)
    quats: [N, 4] (quartanions that expresses rotation - w, x, y, z)
    """
    # 1. reconstructing scale
    s = torch.exp(scales)

    # 2. quartanions optimization
    q = quats / quats.norm(dim=-1, keepdim=True)
    r, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    # 3. 쿼터니언 -> 3x3 회전 행렬(R) 변환
    R = torch.zeros((q.shape[0], 3, 3), device=q.device, dtype=torch.float32)
    R[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    R[:, 0, 1] = 2.0 * (x * y - r * z)
    R[:, 0, 2] = 2.0 * (x * z + r * y)
    R[:, 1, 0] = 2.0 * (x * y + r * z)
    R[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    R[:, 1, 2] = 2.0 * (y * z - r * x)
    R[:, 2, 0] = 2.0 * (x * z - r * y)
    R[:, 2, 1] = 2.0 * (y * z + r * x)
    R[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)

    # 4. 공분산 행렬 계산! (R * S * S^T * R^T)
    # S가 대각행렬(Diagonal)이라서 브로드캐스팅으로 엄청 빠르게 곱할 수 있어!
    L = R * s.unsqueeze(1)  # [N, 3, 3]
    cov3D = torch.bmm(L, L.transpose(1, 2))  # [N, 3, 3]

    # 5. 메모리 다이어트! (대칭 행렬이므로 Upper Triangular 6개 요소만 추출)
    cov3D_precomp = torch.zeros((q.shape[0], 6), device=q.device, dtype=torch.float32)
    cov3D_precomp[:, 0] = cov3D[:, 0, 0]
    cov3D_precomp[:, 1] = cov3D[:, 0, 1]
    cov3D_precomp[:, 2] = cov3D[:, 0, 2]
    cov3D_precomp[:, 3] = cov3D[:, 1, 1]
    cov3D_precomp[:, 4] = cov3D[:, 1, 2]
    cov3D_precomp[:, 5] = cov3D[:, 2, 2]

    return cov3D_precomp


def transform2origin(position_tensor):
    min_pos = torch.min(position_tensor, 0)[0]
    max_pos = torch.max(position_tensor, 0)[0]
    max_diff = torch.max(max_pos - min_pos)
    original_mean_pos = (min_pos + max_pos) / 2.0
    scale = 1.0 / max_diff
    original_mean_pos = original_mean_pos.to(device="cuda")
    scale = scale.to(device="cuda")
    new_position_tensor = (position_tensor - original_mean_pos) * scale

    return new_position_tensor, scale, original_mean_pos


def undotransform2origin(position_tensor, scale, original_mean_pos):
    return original_mean_pos + position_tensor / scale


def undoshift2center111(position_tensor, device="cuda"):
    tensor111 = torch.tensor([1.0, 1.0, 1.0], device=device)
    return position_tensor - tensor111


def undo_all_transforms(input, rotation_matrices, scale_origin, original_mean_pos):
    return torch.mm(
        undotransform2origin(
            undoshift2center111(input), scale_origin, original_mean_pos
        ),
        rotation_matrices,
    )


def generate_local_coord(vertical_vector):
    vertical_vector = vertical_vector / np.linalg.norm(vertical_vector)
    horizontal_1 = np.array([1, 1, 1])
    if np.abs(np.dot(horizontal_1, vertical_vector)) < 0.01:
        horizontal_1 = np.array([0.72, 0.37, -0.67])
    # gram schimit
    horizontal_1 = (
        horizontal_1 - np.dot(horizontal_1, vertical_vector) * vertical_vector
    )
    horizontal_1 = horizontal_1 / np.linalg.norm(horizontal_1)
    horizontal_2 = np.cross(horizontal_1, vertical_vector)

    return vertical_vector, horizontal_1, horizontal_2


def get_center_view_worldspace_and_observant_coordinate(
    mpm_space_viewpoint_center,
    mpm_space_vertical_upward_axis,
    rotation_matrices,
    scale_origin,
    original_mean_pos,
):
    viewpoint_center_worldspace = undo_all_transforms(
        mpm_space_viewpoint_center, rotation_matrices, scale_origin, original_mean_pos
    )
    mpm_space_up = mpm_space_vertical_upward_axis + mpm_space_viewpoint_center
    worldspace_up = undo_all_transforms(
        mpm_space_up, rotation_matrices, scale_origin, original_mean_pos
    )
    world_space_vertical_axis = worldspace_up - viewpoint_center_worldspace
    viewpoint_center_worldspace = np.squeeze(
        viewpoint_center_worldspace.clone().detach().cpu().numpy(), 0
    )
    vertical, h1, h2 = generate_local_coord(
        np.squeeze(world_space_vertical_axis.clone().detach().cpu().numpy(), 0)
    )
    observant_coordinates = np.column_stack((h1, h2, vertical))

    return viewpoint_center_worldspace, observant_coordinates


def generate_rotation_matrices(degrees, axises):
    assert len(degrees) == len(axises)

    matrices = []

    for i in range(len(degrees)):
        matrices.append(generate_rotation_matrix(degrees[i], axises[i]))

    return torch.stack(matrices)


def apply_rotation(position_tensor, rotation_matrix):
    rotated = torch.mm(position_tensor, rotation_matrix.T)
    return rotated


def apply_cov_rotation(cov_tensor, rotation_matrix):
    rotated = torch.matmul(cov_tensor, rotation_matrix.T)
    rotated = torch.matmul(rotation_matrix, rotated)
    return rotated


def apply_rotations(position_tensor, rotation_matrices):
    for i in range(len(rotation_matrices)):
        position_tensor = apply_rotation(position_tensor, rotation_matrices[i])
    return position_tensor


def apply_cov_rotations(upper_cov_tensor, rotation_matrices):
    cov_tensor = get_mat_from_upper(upper_cov_tensor)
    for i in range(len(rotation_matrices)):
        cov_tensor = apply_cov_rotation(cov_tensor, rotation_matrices[i])
    return get_upper_from_mat(cov_tensor)


def apply_inverse_rotation(position_tensor, rotation_matrix):
    rotated = torch.mm(position_tensor, rotation_matrix)
    return rotated


def apply_inverse_rotations(position_tensor, rotation_matrices):
    for i in range(len(rotation_matrices)):
        R = rotation_matrices[len(rotation_matrices) - 1 - i]
        position_tensor = apply_inverse_rotation(position_tensor, R)
    return position_tensor
