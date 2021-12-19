import numpy as np
from numpy.lib.function_base import diff
from scipy.sparse import csc_matrix, identity
from scipy.sparse.linalg import factorized


def mat_id_in_vec(i, j, num_rows):
    return i * num_rows + j


def assemble_div_op(nx, ny, h):
    # for a grid of size (nx, ny), we have edge values u of size (nx, ny - 1) and v of size (nx - 1, ny)
    # B.dot(np.concatenate([u.flatten(), v.flatten()])) gives div(velocity) at each cell
    # B should be of shape [nx * ny, (nx + 1) * ny + nx * (ny + 1)] where
    # each row corresponds to each cell, and each col corresponds to an edge
    rows = []
    cols = []
    data = []

    num_u = (nx + 1) * ny

    for i in range(nx):
        for j in range(ny):
            row_idx = mat_id_in_vec(i, j, ny)
            # for each cell(i, j), div = (u(i+1,j)-u(i,j)) / h + (v(i,j+1),v(i,j)) / h
            
            # u(i, j)
            rows.append(row_idx)
            cols.append(mat_id_in_vec(i, j, ny))
            data.append(-1.0 / h)

            # u(i+1,j)
            rows.append(row_idx)
            cols.append(mat_id_in_vec(i + 1, j, ny))
            data.append(1.0 / h)

            # v(i, j)
            rows.append(row_idx)
            cols.append(num_u + mat_id_in_vec(i, j, ny + 1))
            data.append(-1.0 / h)

            # v(i, j+1)
            rows.append(row_idx)
            cols.append(num_u + mat_id_in_vec(i, j + 1, ny + 1))
            data.append(1.0 / h)

    B = csc_matrix((data, (rows, cols)), shape=(nx * ny, (nx + 1) * ny + nx * (ny + 1)))
    return B  


class VelocityGrid():
    """Staggered MAC grid for diffusion, pressure solve, etc."""
    def __init__(self, nx, ny, h):
        # The image size is [nx, ny], so each "particle" is each pixel in the image
        # as a result, the MAC cell corners are the center of each pixel, so
        # the dimension of the staggered MAC grid is (nx - 1) * (ny - 1)
        self.nx = nx
        self.ny = ny
        self.h = h
        self.num_u = nx * (ny - 1)
        self.num_v = (nx - 1) * ny

        self.u = np.zeros([nx, ny - 1])  # u (first) component of velocity
        self.v = np.zeros([nx - 1, ny])  # v (second) component of velocity
        self.p = np.zeros([nx - 1, ny - 1])  # pressure at cell center

        self.B = assemble_div_op(nx - 1, ny - 1, h)  # for computing divergence on edges
        self.D = self.B.T  # for computing gradient at cell centers
        laplacian_p_op = self.B.dot(self.D)  # for computing laplacian at cell centers
        self.laplacian_p_op_factorized = factorized(laplacian_p_op)  # pre-factor to speed up solve time
        
        # cached values for diffusion, will update/cache once viscosity/dt are available
        self.viscosity = None
        self.dt = None
        self.diffusion_op_u_factorized = None
        self.diffusion_op_v_factorized = None

        self.Pu, self.Pv = self.assemble_img_velo_to_grid_transform()
    
    def u_vec_idx(self, i, j):
        # Return the idx of u(i, j) in u.flatten()
        return mat_id_in_vec(i, j, self.ny - 1)
    
    def v_vec_idx(self, i, j):
        # Return the idx of v(i, j) in v.flatten()
        return mat_id_in_vec(i, j, self.ny)
    
    def p_vec_idx(self, i, j):
        # Return the idx of p(i, j) in p.flatten()
        return mat_id_in_vec(i, j, self.ny - 1)

    def assemble_img_velo_to_grid_transform(self):
        # Build a [num_u, nx * ny] matrix and a [num_v, nx * ny] matrix to map velocities in the (nx, ny) image
        # pixel space to self.u and self.v
        rows = []
        cols = []
        data = []
        for i in range(self.nx):
            for j in range(self.ny - 1):
                # u(i,j) is the average velocities from img[i,j] and img[i,j+1]
                rows.append(self.u_vec_idx(i, j))
                cols.append(mat_id_in_vec(i, j, self.ny))
                data.append(0.5)
                rows.append(self.u_vec_idx(i, j))
                cols.append(mat_id_in_vec(i, j + 1, self.ny))
                data.append(0.5)
        Pu = csc_matrix((data, (rows, cols)), shape=(self.num_u, self.nx * self.ny))

        rows = []
        cols = []
        data = []
        for i in range(self.nx - 1):
            for j in range(self.ny):
                # v(i,j) is the average velocities from img[i,j] and img[i+1,j]
                rows.append(self.v_vec_idx(i, j))
                cols.append(mat_id_in_vec(i, j, self.ny))
                data.append(0.5)
                rows.append(self.v_vec_idx(i, j))
                cols.append(mat_id_in_vec(i + 1, j, self.ny))
                data.append(0.5)
        Pv = csc_matrix((data, (rows, cols)), shape=(self.num_v, self.nx * self.ny))
        return Pu, Pv

    def init_velo_from_img(self, img_velo):
        # Given velocities from an image, map to the velocity on the staggered grid
        self.u = self.Pu.dot(img_velo[:, :, 0].flatten()).reshape(self.nx, self.ny - 1)
        self.v = self.Pv.dot(img_velo[:, :, 1].flatten()).reshape(self.nx - 1, self.ny)
    
    def convert_velo_on_img(self):
        img_velo = np.zeros([self.nx, self.ny, 2])
        img_velo[:, :, 0] = self.Pu.T.dot(self.u.flatten()).reshape(self.nx, self.ny)
        img_velo[:, :, 1] = self.Pv.T.dot(self.v.flatten()).reshape(self.nx, self.ny)
        return img_velo
    
    def compute_diffusion_op(self, dt, viscosity):
        if self.dt != dt or self.viscosity != viscosity:
            self.viscosity = viscosity
            self.dt = dt
            B_u = self.B[:, :self.num_u]
            B_v = self.B[:, self.num_u:]
            laplacian_u_op = B_u.T.dot(B_u)  # for computing laplacian at u
            laplacian_v_op = B_v.T.dot(B_v)  # for computing laplacian at v
            self.diffusion_op_u_factorized = factorized(
                identity(self.num_u, format="csc") - dt * viscosity * laplacian_u_op
            )
            self.diffusion_op_v_factorized = factorized(
                identity(self.num_v, format="csc") - dt * viscosity * laplacian_v_op
            )
    
    def diffuse(self, dt, viscosity):
        self.compute_diffusion_op(dt, viscosity)
        self.u = self.diffusion_op_u_factorized(self.u.flatten()).reshape(self.u.shape)
        self.v = self.diffusion_op_v_factorized(self.v.flatten()).reshape(self.v.shape)
    
    def pressure_solve(self, dt, density):
        div_velocity = self.B.dot(np.concatenate([self.u.flatten(), self.v.flatten()]))
        self.p = self.laplacian_p_op_factorized(div_velocity)

        delta_velo_flattened = dt / density * self.D.dot(self.p)
        self.u -= delta_velo_flattened[:self.num_u].reshape(self.nx, self.ny - 1)
        self.v -= delta_velo_flattened[self.num_u:].reshape(self.nx - 1, self.ny)


class ScalarGrid():
    """Staggered MAC grid for scalar diffusion"""
    def __init__(self, nx, ny, h):
        # For each scalar field of size (nx, ny), the values are at the cell center
        # We build and cache the gradient matrix for computing the gradient at cell center, to be used in diffusion
        self.nx = nx
        self.ny = ny
        self.h = h

        self.D = assemble_div_op(nx, ny, h).T

        # cached values for diffusion, will update/cache once diffusion constant/dt are available
        self.diffusion_constant = None
        self.dt = None
        self.diffusion_op_u_factorized = None
    
    def compute_diffusion_op(self, dt, diffusion_constant):
        if self.dt != dt or self.diffusion_constant != diffusion_constant:
            self.diffusion_constant = diffusion_constant
            self.dt = dt
            laplacian_op = self.D.dot(self.D.T)
            self.diffusion_op_factorized = factorized(
                identity(self.nx * self.ny, format="csc") - dt * diffusion_constant * laplacian_op
            )
    
    def diffuse(self, S, dt, diffusion_constant):
        self.compute_diffusion_op(dt, diffusion_constant)
        return self.diffusion_op_factorized(S.flatten()).reshape(S.shape)