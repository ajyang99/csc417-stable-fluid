import numpy as np
import cv2
import os
from fire import Fire
from scipy import ndimage

from staggered_grid import Grid


class StableFluidImg():
    """Apply the stable fluid algorithm to a 2D image.
    
    We treat each pixel of the image as a grid cell described in Section 3.1 of the paper.
    The 2D velocity field thus has the same width and height as the image, and the scalar field consists of the
    RGB values of the image.
    """
    def __init__(self, img, outdir):
        self.nx = img.shape[0]
        self.ny = img.shape[1]
        
        self.img = img  # scalar field at cell center of shape (nx, ny, 3)
        self.velocities = np.zeros([self.nx, self.ny, 2])  # 2D velocity field
        self.grid = Grid(self.nx, self.ny, 1.0)  # staggered MAC grid

        self.outdir = outdir
        os.makedirs(outdir, exist_ok=True)

        # positions of each pixel/cell on the grid to be used to trace particle in advection
        # note that unlike the paper we don't add the 0.5 offset since scipy.ndimage assumes (0, 0) is cell center
        self.pos_x, self.pos_y = np.indices([self.nx, self.ny])
    
    def add_external_forces(self, x, f, dt) -> None:
        x += dt * f

    def advect(self, dt):
        # trace the particle position with 2nd order Runge-Kutta
        # find the mid-point position at -0.5 * dt
        mid_point_pos = np.stack(
            [
                self.pos_x - 0.5 * dt * self.velocities[:, :, 0],
                self.pos_y - 0.5 * dt * self.velocities[:, :, 1]
            ], axis=0
        )
        # find the velocities at the midpoint with interpolation
        # we use scipy.ndimage to do the linear interpolation for us
        # we use the wrap mode since the image is a texture map that wraps around
        mid_point_velocities = np.zeros([self.nx, self.ny, 2])
        for i in range(2):
            mid_point_velocities[:, :, i] = ndimage.map_coordinates(self.velocities[:, :, i], mid_point_pos, order=2, mode="wrap")
        pos = np.stack(
            [
                self.pos_x - dt * mid_point_velocities[:, :, 0],
                self.pos_y - dt * mid_point_velocities[:, :, 1]
            ], axis=0
        )
        
        # find the new velocities and texture at each pixel/cell according to pos
        for i in range(3):
            self.img[:, :, i] = ndimage.map_coordinates(self.img[:, :, i], pos, order=2, mode="wrap")
        for i in range(2):
            self.velocities[:, :, i] = ndimage.map_coordinates(self.velocities[:, :, i], pos, order=2, mode="wrap")
    
    def simulate(self, num_iterations=300, dt=0.01, density=0.01, update_with_delta_velocity=False):
        for time_step in range(num_iterations):
            # add external forces to velocities based on how white each pixel is
            external_forces = np.linalg.norm(self.img, axis=2) * 0.2
            self.add_external_forces(self.velocities, external_forces[:, :, None], dt)

            # we can optionally add additional "forces" to the scalar field, but here we don't to not alter total color
            # self.add_external_force(self.img, optional_rgb_external_forces, dt)

            # advection on both velocities and RGB values
            self.advect(dt)

            # diffusion for both velocities and rgb

            # pressure solve
            # here we use a staggered MAC grid as the solver, in place of the FISHPAK subroutines mentioned in the paper
            self.grid.init_velo_from_img(self.velocities)  # project current velocities to the MAC grid
            self.grid.pressure_solve(dt, density)
            if update_with_delta_velocity:
                # use the delta (FLIP-like convention)
                self.velocities += self.grid.convert_velo_on_img(return_delta=True)
            else:
                self.velocities = self.grid.convert_velo_on_img()

            # save image
            cv2.imwrite(os.path.join(self.outdir, f"{time_step:06d}.png"), self.img)


def main(img_fpath="data/monroe.jpeg", outdir="out/monroe", num_iterations=300, dt=0.01, density=0.01):
    img = cv2.imread(img_fpath)
    fluid = StableFluidImg(img, outdir)
    fluid.simulate(num_iterations=num_iterations, dt=dt, density=density)


if __name__ == '__main__':
    Fire({
        "simulate": main
    })

