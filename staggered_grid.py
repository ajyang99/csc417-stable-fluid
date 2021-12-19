import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import factorized


def mat_id_in_vec(i, j, num_cols):
    return i * num_cols + j


class Grid():
    """Staggered MAC grid for pressure solve, etc."""
    def __init__(self, nx, ny, h):
        # The image size is [nx, ny], so each "particle" is each pixel in the image
        # as a result, the pressure cell corners are the center of each pixel, and
        # the dimension of the staggered MAC grid is (nx - 1) * (ny - 1)
        self.nx = nx
        self.ny = ny
        self.h = h
        self.num_u = nx * (ny - 1)
        self.num_v = (nx - 1) * ny

        self.u = np.zeros([nx, ny - 1])  # u (first) component of velocity
        self.v = np.zeros([nx - 1, ny])  # v (second) component of velocity
        self.p = np.zeros([nx - 1, ny - 1])  # pressure at cell center

        self.delta_u = np.zeros([nx, ny - 1])  # change in self.u after pressure solve
        self.delta_v = np.zeros([nx - 1, ny])  # change in self.v after pressure solve

        self.B = self.assemble_B_matrix()  # for computing divergence on edges
        self.D = self.B.T  # for computing gradient at cell centers
        A = self.B.dot(self.D)  # for computing laplacian at cell centers
        self.A_factorized = factorized(A)  # pre-factor to speed up solve time

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
    
    def assemble_B_matrix(self):
        # B.dot(np.concatenate([self.u.flatten(), self.v.flatten()])) gives div(velocity)
        # B should be of shape [(nx - 1) * (ny - 1), nx * (ny - 1) + (nx - 1) * ny] where
        # each row corresponds to each cell, and each col corresponds to a velocity term
        rows = []
        cols = []
        data = []

        for i in range(self.nx - 1):
            for j in range(self.ny - 1):
                row_idx = j * (self.nx - 1) + i
                # for each cell(i, j), div = (u(i+1,j)-u(i,j)) / h + (v(i,j+1),v(i,j)) / h
                
                # u(i, j)
                rows.append(row_idx)
                cols.append(self.u_vec_idx(i, j))
                data.append(-1.0 / self.h)

                # u(i+1,j)
                rows.append(row_idx)
                cols.append(self.u_vec_idx(i + 1, j))
                data.append(1.0 / self.h)

                # v(i, j)
                rows.append(row_idx)
                cols.append(self.num_u + self.v_vec_idx(i, j))
                data.append(-1.0 / self.h)

                # v(i, j+1)
                rows.append(row_idx)
                cols.append(self.num_u + self.v_vec_idx(i, j + 1))
                data.append(1.0 / self.h)

        B = csc_matrix((data, (rows, cols)), shape=((self.nx - 1) * (self.ny - 1), self.num_u + self.num_v))
        return B  

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
    
    def convert_velo_on_img(self, return_delta = False):
        img_velo = np.zeros([self.nx, self.ny, 2])
        if return_delta:
            u = self.delta_u
            v = self.delta_v
        else:
            u = self.u
            v = self.v
        img_velo[:, :, 0] = self.Pu.T.dot(u.flatten()).reshape(self.nx, self.ny)
        img_velo[:, :, 1] = self.Pv.T.dot(v.flatten()).reshape(self.nx, self.ny)
        return img_velo
    
    def pressure_solve(self, dt, density):
        div_velocity = self.B.dot(np.concatenate([self.u.flatten(), self.v.flatten()]))
        self.p = self.A_factorized(div_velocity)

        delta_velo_flattened = dt / density * self.D.dot(self.p)
        self.delta_u = -delta_velo_flattened[:self.num_u].reshape(self.nx, self.ny - 1)
        self.delta_v = -delta_velo_flattened[self.num_u:].reshape(self.nx - 1, self.ny)
        self.u += self.delta_u
        self.v += self.delta_v