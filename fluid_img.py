import numpy as np
import cv2
import os
import imageio
from fire import Fire
from scipy import ndimage

from staggered_grid import VelocityGrid, ScalarGrid


class StableFluidImg():
    """Apply the stable fluid algorithm to a 2D image.
    
    We treat each pixel of the image as a grid cell described in Section 3.1 of the paper.
    The 2D velocity field thus has the same width and height as the image, and the scalar field consists of the
    RGB values of the image.
    """
    def __init__(self, img, outdir):
        self.nx = img.shape[0]
        self.ny = img.shape[1]
        
        # Note like unlike the paper which uses two variables to store the old and updated values for each of
        # the velocity and scalar fields, we just use one variable for each field to store the most up-to-date
        # values, since we can just always update the variables directly and there is no point in saving the
        # old values.
        self.img = img  # scalar field at cell center of shape (nx, ny, 3)
        self.velocities = np.zeros([self.nx, self.ny, 2])  # 2D velocity field
        
        self.velo_grid = VelocityGrid(self.nx, self.ny, 1.0)  # staggered MAC grid for velocity
        self.scalar_grid = ScalarGrid(self.nx, self.ny, 1.0)  # staggered MAC grid for scalar field

        # positions of each pixel/cell on the image grid to be used to trace particle in advection
        # note that unlike the paper we don't add the 0.5 offset since scipy.ndimage assumes (0, 0) is cell center
        self.pos_x, self.pos_y = np.indices([self.nx, self.ny])

        self.outdir = outdir
        os.makedirs(outdir, exist_ok=True)
    
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
            mid_point_velocities[:, :, i] = ndimage.map_coordinates(
                self.velocities[:, :, i], mid_point_pos, order=2, mode="wrap"
            )
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
    
    def simulate(
        self, num_iterations=300, dt=0.01, density=0.01, viscosity=0.0,
        diffusion_constant=0.0, dissipation_rate=0.0, create_video=True
    ):
        out_fpaths = []
        for time_step in range(num_iterations):
            # Step 1: add external forces
            # add external forces to velocities based on how white each pixel is
            external_forces = np.linalg.norm(self.img, axis=2) * 0.2
            self.add_external_forces(self.velocities, external_forces[:, :, None], dt)

            # we can optionally add additional "forces" to the scalar field, but here we don't add additional change
            # to the color
            # self.add_external_force(self.img, optional_rgb_external_forces, dt)

            # Step 2: advect both velocities and scalar/RGB values
            self.advect(dt)

            # Step 3 & 4 (velocity): diffusion & pressure solve
            # here we use a staggered MAC grid as the solver, in place of the FISHPAK subroutines mentioned in the paper
            self.velo_grid.init_velo_from_img(self.velocities)  # project current velocities to the MAC grid
  
            # diffusion for the velocity
            self.velo_grid.diffuse(dt, viscosity)

            # pressure solve
            self.velo_grid.pressure_solve(dt, density)
            self.velocities = self.velo_grid.convert_velo_on_img()

            # Step 3 & 4 (scalar values): diffusion & dissipation
            # diffusion for the scalar field
            if diffusion_constant != 0:
                for i in range(3):
                    self.img[:, :, i] = self.scalar_grid.diffuse(self.img[:, :, i], dt, diffusion_constant)

            # dissipation
            if dissipation_rate != 0:
                self.img = self.img / (1 + dt * dissipation_rate)

            # # save image
            out_fpath = os.path.join(self.outdir, f"{time_step:06d}.png")
            # cv2.imwrite(out_fpath, self.img)
            out_fpaths.append(out_fpath)
        
        if create_video:
            writer = imageio.get_writer(os.path.join(self.outdir, "out.mp4"), fps=20)
            for out_fpath in out_fpaths:
                im = imageio.imread(out_fpath)
                writer.append_data(im)
            writer.close()
            writer = imageio.get_writer(os.path.join(self.outdir, "reverse.mp4"), fps=20)
            for out_fpath in out_fpaths[::-1]:
                im = imageio.imread(out_fpath)
                writer.append_data(im)
            writer.close()


def main(
    img_fpath="data/monroe.jpeg",
    outdir="out/monroe",
    num_iterations=300,
    dt=0.01,
    density=0.01,
    viscosity=0.0,
    diffusion_constant=0.0,
    dissipation_rate=0.0,
    create_video=True,
):
    img = cv2.imread(img_fpath)
    fluid = StableFluidImg(img, outdir)
    fluid.simulate(
        num_iterations=num_iterations,
        dt=dt,
        density=density,
        viscosity=viscosity,
        diffusion_constant=diffusion_constant,
        dissipation_rate=dissipation_rate,
        create_video=create_video,
    )


if __name__ == "__main__":
    Fire({
        "simulate": main
    })

