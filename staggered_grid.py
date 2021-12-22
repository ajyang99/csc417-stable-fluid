import numpy as np
from numpy.lib.function_base import diff
from numpy.lib.twodim_base import diag
from scipy.sparse import csc_matrix, identity
from scipy.sparse.linalg import factorized
from utils import BoundaryCondition


def mat_id_in_vec(i, j, ny):
    return i * ny + j


def assemble_div_op(nx, ny, h, boundary_condition = BoundaryCondition.FIXED):
    # for a grid of size (nx, ny), we have edge values u of size (nx, ny - 1) and v of size (nx - 1, ny)
    # B.dot(np.concatenate([u.flatten(), v.flatten()])) gives div(velocity) at each cell
    # B should be of shape [nx * ny, (nx + 1) * ny + nx * (ny + 1)] where
    # each row corresponds to each cell, and each col corresponds to an edge
    rows = []
    cols = []
    data = []

    num_u = (nx + 1) * ny

    if boundary_condition == BoundaryCondition.FIXED:
        nx_new = nx
        ny_new = ny
    elif boundary_condition == BoundaryCondition.PERIODIC:
        # When the boundary condition is periodic, we add an addition row and column that represents cell
        # that wraps around the boundary
        nx_new = nx + 1
        ny_new = ny + 1
    else:
        raise ValueError(f"Unsupported boundary condition {boundary_condition}")

    for i in range(nx_new):
        for j in range(ny_new):
            row_idx = mat_id_in_vec(i, j, ny_new)
            # for each cell(i, j), div = (u(i+1,j)-u(i,j)) / h + (v(i,j+1),v(i,j)) / h
            
            # u(i, j)
            rows.append(row_idx)
            cols.append(mat_id_in_vec(i, j % ny, ny))
            data.append(-1.0 / h)

            # u(i+1,j)
            rows.append(row_idx)
            cols.append(mat_id_in_vec((i + 1) % (nx + 1), j % ny, ny))
            data.append(1.0 / h)

            # v(i, j)
            rows.append(row_idx)
            cols.append(num_u + mat_id_in_vec(i % nx, j, ny + 1))
            data.append(-1.0 / h)

            # v(i, j+1)
            rows.append(row_idx)
            cols.append(num_u + mat_id_in_vec(i % nx, (j + 1) % (ny + 1), ny + 1))
            data.append(1.0 / h)

    B = csc_matrix((data, (rows, cols)), shape=(nx_new * ny_new, (nx + 1) * ny + nx * (ny + 1)))
    return B  


def assemble_laplacian_op_for_wrapped_elements(nx, ny, h):
    rows = []
    cols = []
    data = []
    for i, dir in zip([0, -1], [-1, 1]):
        for j in range(ny):
            # for cell (i, j), we need to add contribution from cell[i + dir, j]
            rows.append(mat_id_in_vec(i % nx, j, ny))
            cols.append(mat_id_in_vec((i + dir) % nx, j, ny))
            data.append(1.0 / h ** 2)
    for i in range(nx):
        for j, dir in zip([0, -1], [-1, 1]):
            # for cell (i, j), we need to add contribution from cell[i, j + dir]
            rows.append(mat_id_in_vec(i, j % ny, ny))
            cols.append(mat_id_in_vec(i, (j + dir) % ny, ny))
            data.append(1.0 / h ** 2)
    delta = csc_matrix((data, (rows, cols)), shape=(nx * ny, nx * ny))
    return delta

class PressureSolver():
    """Staggered MAC grid for pressure solve"""
    def __init__(self, nx, ny, h, boundary_condition: BoundaryCondition):
        # The image size is [nx, ny]
        # The MAC cell corners are the center of each pixel, so
        # the dimension of the staggered MAC grid is (nx - 1) * (ny - 1)
        self.nx = nx
        self.ny = ny
        self.h = h  # grid size in world coordinate
        self.num_u = nx * (ny - 1)
        self.num_v = (nx - 1) * ny


        self.u = np.zeros([nx, ny - 1])  # u (first) component of velocity
        self.v = np.zeros([nx - 1, ny])  # v (second) component of velocity
        
        if boundary_condition == BoundaryCondition.FIXED:
            self.p = np.zeros([nx - 1, ny - 1])  # pressure at cell center
        elif boundary_condition == BoundaryCondition.PERIODIC:
            self.p = np.zeros([nx, ny])  # we will also have cells between u[0, :] and u[-1, :] and v[:, 0] and v[:, -1]
        else:
            raise ValueError(f"Unsupported boundary condition {boundary_condition}")
        self.boundary_condition = boundary_condition

        self.div_op = assemble_div_op(nx - 1, ny - 1, h, boundary_condition)  # for computing divergence for each cell
        self.grad_op = -self.div_op.T  # for computing gradient at cell centers
        laplacian_p_op = self.div_op.dot(self.grad_op)  # for computing laplacian at cell centers
        self.laplacian_p_op_factorized = factorized(laplacian_p_op)  # pre-factor to speed up solve time

        self.Pu, self.Pv = self.assemble_img_velo_to_grid_transform()
    
    def u_vec_idx(self, i, j):
        # Return the idx of u(i, j) in u.flatten()
        return mat_id_in_vec(i, j, self.ny - 1)
    
    def v_vec_idx(self, i, j):
        # Return the idx of v(i, j) in v.flatten()
        return mat_id_in_vec(i, j, self.ny)

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
    
    def pressure_solve(self, dt, density):
        if self.boundary_condition == BoundaryCondition.FIXED:
            # note that u[0, :] are at the vertical line that crosses the cell centers at img[0, :]
            # so if we make the boundary velocity zero, then by interpolation u[0, :] should be half
            # of what it should be, similarly for the other boundaries
            self.u[0, :] /= 2
            self.u[-1, :] /= 2
            self.v[:, 0] /= 2
            self.v[:, -1] /= 2

        div_velocity = self.div_op.dot(np.concatenate([self.u.flatten(), self.v.flatten()]))
        self.p = self.laplacian_p_op_factorized(div_velocity)

        delta_velo_flattened = dt / density * self.grad_op.dot(self.p)
        self.u -= delta_velo_flattened[:self.num_u].reshape(self.nx, self.ny - 1)
        self.v -= delta_velo_flattened[self.num_u:].reshape(self.nx - 1, self.ny)


class DiffusionSolver():
    """Finite difference solver for diffusion"""
    def __init__(self, nx, ny, h, boundary_condition):
        # For each scalar field of size (nx, ny), the values are at the cell center
        # We build and cache the gradient matrix for computing the gradient at cell center, to be used in diffusion
        self.nx = nx
        self.ny = ny
        self.h = h

        div_op = assemble_div_op(nx, ny, h, BoundaryCondition.FIXED)
        grad_op = -div_op.T
        self.laplacian_op = div_op.dot(grad_op)
        if boundary_condition == BoundaryCondition.PERIODIC:
            self.laplacian_op += assemble_laplacian_op_for_wrapped_elements(nx, ny, h)

        # cached values for diffusion, will update/cache once diffusion constant/dt are available
        self.diffusion_constant = None
        self.dt = None
        self.diffusion_op_factorized = None
    
    def compute_diffusion_op(self, dt, diffusion_constant):
        if self.dt != dt or self.diffusion_constant != diffusion_constant:
            self.diffusion_constant = diffusion_constant
            self.dt = dt
            self.diffusion_op_factorized = factorized(
                identity(self.nx * self.ny, format="csc") - dt * diffusion_constant * self.laplacian_op
            )
    
    def diffuse(self, S, dt, diffusion_constant):
        self.compute_diffusion_op(dt, diffusion_constant)
        return self.diffusion_op_factorized(S.flatten()).reshape(S.shape)