import numpy as np
from numpy.lib.function_base import diff
from scipy.sparse import csc_matrix, identity
from scipy.sparse.linalg import factorized


def mat_id_in_vec(i, j, k, ny, nz):
    return (i * ny + j) * nz + k


def assemble_div_op(nx, ny, nz, h):
    # for a grid of size (nx, ny), we have edge values u of size (nx, ny - 1) and v of size (nx - 1, ny)
    # B.dot(np.concatenate([u.flatten(), v.flatten()])) gives div(velocity) at each cell
    # B should be of shape [nx * ny, (nx + 1) * ny + nx * (ny + 1)] where
    # each row corresponds to each cell, and each col corresponds to an edge
    rows = []
    cols = []
    data = []

    num_u = (nx + 1) * ny * nz
    num_v = nx * (ny + 1) * nz
    num_w = nx * ny * (nz + 1)

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                row_idx = mat_id_in_vec(i, j, k, ny, nz)
                # for each cuboid(i, j, k), div = (u(i+1,j,k)-u(i,j,k)) / h + (v(i,j+1,k)-v(i,j,k)) / h + (w(i,j,k+1)-w(i,j,k)) / h
                
                # u(i, j, k)
                rows.append(row_idx)
                cols.append(mat_id_in_vec(i, j, k, ny, nz))
                data.append(-1.0 / h)

                # u(i+1, j, k)
                rows.append(row_idx)
                cols.append(mat_id_in_vec(i + 1, j, k, ny, nz))
                data.append(1.0 / h)

                # v(i, j, k)
                rows.append(row_idx)
                cols.append(num_u + mat_id_in_vec(i, j, k, ny + 1, nz))
                data.append(-1.0 / h)

                # v(i, j+1, k)
                rows.append(row_idx)
                cols.append(num_u + mat_id_in_vec(i, j + 1, k, ny + 1, nz))
                data.append(1.0 / h)

                # w(i, j, k)
                rows.append(row_idx)
                cols.append(num_u + num_v + mat_id_in_vec(i, j, k, ny, nz + 1))
                data.append(-1.0 / h)

                # w(i, j, k+1)
                rows.append(row_idx)
                cols.append(num_u + num_v + mat_id_in_vec(i, j, k + 1, ny, nz + 1))
                data.append(1.0 / h)

    B = csc_matrix((data, (rows, cols)), shape=(nx * ny * nz, num_u + num_v + num_w))
    return B  


class PressureSolver():
    """Staggered MAC grid for pressure solve"""
    def __init__(self, nx, ny, nz, h):
        # The input 3D voxel grid is [nx, ny, nz], so the MAC grid dimension is [nx - 1, ny - 1, nz - 1]
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.h = h  # grid size in world coordinate
        self.num_u = nx * (ny - 1) * (nz - 1)
        self.num_v = (nx - 1) * ny * (nz - 1)
        self.num_w = (nx - 1) * (ny - 1) * nz

        self.u = np.zeros([nx, ny - 1, nz - 1])  # u (first) component of velocity
        self.v = np.zeros([nx - 1, ny, nz - 1])  # v (second) component of velocity
        self.w = np.zeros([nx - 1, ny - 1, nz])  # w (third) component of velocity
        self.p = np.zeros([nx - 1, ny - 1, nz - 1])  # pressure at cuboid center

        self.div_op = assemble_div_op(nx - 1, ny - 1, nz - 1, h)  # for computing divergence for each cell
        self.grad_op = -self.div_op.T  # for computing gradient at cell centers
        laplacian_p_op = self.div_op.dot(self.grad_op)  # for computing laplacian at cell centers
        self.laplacian_p_op_factorized = factorized(laplacian_p_op)  # pre-factor to speed up solve time

        self.Pu, self.Pv, self.Pw = self.assemble_voxel_velo_to_grid_transform()
    
    def u_vec_idx(self, i, j, k):
        # Return the idx of u(i, j) in u.flatten()
        return mat_id_in_vec(i, j, k, self.ny - 1, self.nz - 1)
    
    def v_vec_idx(self, i, j, k):
        # Return the idx of v(i, j) in v.flatten()
        return mat_id_in_vec(i, j, k, self.ny, self.nz - 1)
    
    def w_vec_idx(self, i, j, k):
        # Return the idx of v(i, j) in v.flatten()
        return mat_id_in_vec(i, j, k, self.ny - 1, self.nz)
    
    def p_vec_idx(self, i, j, k):
        # Return the idx of p(i, j) in p.flatten()
        return mat_id_in_vec(i, j, k, self.ny - 1, self.nz - 1)

    def assemble_voxel_velo_to_grid_transform(self):
        # Build a [num_u, nx * ny * nz] maix, a [num_v, nx * ny * nz] matrix and a [num_w, nx * ny * nz] matrix to map
        # velocities in the (nx, ny, nz) voxel grid space to self.u, self.v and self.w on the staggered grid
        rows = []
        cols = []
        data = []
        for i in range(self.nx):
            for j in range(self.ny - 1):
                for k in range(self.nz - 1):
                    # u(i,j,k) is the average velocities from voxel[i, j:j+2, k:k+2]
                    row_idx = self.u_vec_idx(i, j, k)
                    for voxel_j in range(j, j + 2):
                        for voxel_k in range(k, k + 2):
                            rows.append(row_idx)
                            cols.append(mat_id_in_vec(i, voxel_j, voxel_k, self.ny, self.nz))
                            data.append(0.25)
        Pu = csc_matrix((data, (rows, cols)), shape=(self.num_u, self.nx * self.ny * self.nz))

        rows = []
        cols = []
        data = []
        for i in range(self.nx - 1):
            for j in range(self.ny):
                for k in range(self.nz - 1):
                    # v(i,j,k) is the average velocities from voxel[i:i+2, j, k:k+2]
                    row_idx = self.v_vec_idx(i, j, k)
                    for voxel_i in range(i, i + 2):
                        for voxel_k in range(k, k + 2):
                            rows.append(row_idx)
                            cols.append(mat_id_in_vec(voxel_i, j, voxel_k, self.ny, self.nz))
                            data.append(0.25)
        Pv = csc_matrix((data, (rows, cols)), shape=(self.num_v, self.nx * self.ny * self.nz))

        rows = []
        cols = []
        data = []
        for i in range(self.nx - 1):
            for j in range(self.ny - 1):
                for k in range(self.nz):
                    # w(i,j,k) is the average velocities from voxel[i:i+2, j:j+2, k]
                    row_idx = self.w_vec_idx(i, j, k)
                    for voxel_i in range(i, i + 2):
                        for voxel_j in range(j, j + 2):
                            rows.append(row_idx)
                            cols.append(mat_id_in_vec(voxel_i, voxel_j, k, self.ny, self.nz))
                            data.append(0.25)
        Pw = csc_matrix((data, (rows, cols)), shape=(self.num_w, self.nx * self.ny * self.nz))
        return Pu, Pv, Pw

    def init_velo_from_voxel_grid(self, voxel_velo):
        # Given velocities from an image, map to the velocity on the staggered grid
        self.u = self.Pu.dot(voxel_velo[..., 0].flatten()).reshape(self.u.shape)
        self.v = self.Pv.dot(voxel_velo[..., 1].flatten()).reshape(self.v.shape)
        self.w = self.Pw.dot(voxel_velo[..., 2].flatten()).reshape(self.w.shape)
    
    def convert_velo_on_voxel_grid(self):
        voxel_velo = np.zeros([self.nx, self.ny, self.nz, 3])
        voxel_velo[..., 0] = self.Pu.T.dot(self.u.flatten()).reshape(self.nx, self.ny, self.nz)
        voxel_velo[..., 1] = self.Pv.T.dot(self.v.flatten()).reshape(self.nx, self.ny, self.nz)
        voxel_velo[..., 2] = self.Pw.T.dot(self.w.flatten()).reshape(self.nx, self.ny, self.nz)
        return voxel_velo
    
    def pressure_solve(self, dt, density):
        div_velocity = self.div_op.dot(np.concatenate([self.u.flatten(), self.v.flatten(), self.w.flatten()]))
        self.p = self.laplacian_p_op_factorized(div_velocity)

        delta_velo_flattened = dt / density * self.grad_op.dot(self.p)
        self.u -= delta_velo_flattened[:self.num_u].reshape(self.u.shape)
        self.v -= delta_velo_flattened[self.num_u:self.num_u + self.num_v].reshape(self.v.shape)
        self.w -= delta_velo_flattened[self.num_u + self.num_v:].reshape(self.w.shape)


class DiffusionSolver():
    """Finite difference solver for diffusion"""
    def __init__(self, nx, ny, nz, h):
        # For each scalar field of size (nx, ny, nz), the values are at the cuboid center
        # We build and cache the gradient matrix for computing the gradient at cuboid center, to be used in diffusion
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.h = h

        div_op = assemble_div_op(nx, ny, nz, h)
        grad_op = -div_op.T
        self.laplacian_op = div_op.dot(grad_op)

        # cached values for diffusion, will update/cache once diffusion constant/dt are available
        self.diffusion_constant = None
        self.dt = None
        self.diffusion_op_factorized = None
    
    def compute_diffusion_op(self, dt, diffusion_constant):
        if self.dt != dt or self.diffusion_constant != diffusion_constant:
            self.diffusion_constant = diffusion_constant
            self.dt = dt
            self.diffusion_op_factorized = factorized(
                identity(self.nx * self.ny * self.nz, format="csc") - dt * diffusion_constant * self.laplacian_op
            )
    
    def diffuse(self, S, dt, diffusion_constant):
        self.compute_diffusion_op(dt, diffusion_constant)
        return self.diffusion_op_factorized(S.flatten()).reshape(S.shape)