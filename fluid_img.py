import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

from enum import Enum
from fire import Fire
from scipy import ndimage

from staggered_grid import PressureSolver, DiffusionSolver
from utils import create_video_from_img_fpths, BoundaryCondition


class ExternalForceType(str, Enum):
    PIXEL_INTENSITY = "intensity"
    BOT_PUSH_UP = "bot_push_up"
    GRAVITY_INTENSITY = "gravity"


def construct_external_forces_by_type(img, external_force_type):
    nx, ny = img.shape[:2]
    if external_force_type == ExternalForceType.PIXEL_INTENSITY:
        external_forces = np.linalg.norm(img, axis=2) * 0.2
        external_forces = external_forces[:, :, None]
    elif external_force_type == ExternalForceType.BOT_PUSH_UP:
        external_forces = np.zeros([nx, ny, 2])
        external_forces[int(0.35 * nx):int(0.65 * nx), int(0.5 * ny):, 0] = -0.1
    elif external_force_type == ExternalForceType.GRAVITY_INTENSITY:
        external_forces = np.zeros([nx, ny, 2])
        external_forces[:, :, 0] = np.linalg.norm(img, axis=2) * 0.2
    else:
        raise ValueError(f"Unsupported external force type {external_force_type}")
    return external_forces


class StableFluidImg():
    """Apply the stable fluid algorithm to a 2D image.
    
    We treat each pixel of the image as a grid cell described in Section 3.1 of the paper.
    The 2D velocity field thus has the same width and height as the image, and the scalar field consists of the
    RGB values of the image.
    """
    def __init__(
        self, img, outdir,
        boundary_condition,
        interp_order=2,
        external_force_type=ExternalForceType.PIXEL_INTENSITY
    ):
        self.nx = img.shape[0]
        self.ny = img.shape[1]
        
        # Note like unlike the paper which uses two variables to store the old and updated values for each of
        # the velocity and scalar fields, we just use one variable for each field to store the most up-to-date
        # values, since we can just always update the variables directly and there is no point in saving the
        # old values.
        self.img = img  # scalar color field at cell center of shape (nx, ny, 3)
        self.velocities = np.zeros([self.nx, self.ny, 2])  # 2D velocity field
        
        self.pressure_solver = PressureSolver(self.nx, self.ny, 1.0, boundary_condition)  # staggered MAC grid for pressure solve
        self.velo_diffusion_solver = DiffusionSolver(self.nx, self.ny, 1.0, boundary_condition)  # diffusion for velocity field
        self.scalar_diffusion_solver = DiffusionSolver(self.nx, self.ny, 1.0, boundary_condition)  # diffusion for scalar field

        # positions of each pixel/cell on the image grid to be used to trace particle in advection
        # note that unlike the paper we don't add the 0.5 offset since scipy.ndimage assumes (0, 0) is cell center
        self.pos_x, self.pos_y = np.indices([self.nx, self.ny])

        self.boundary_condition = boundary_condition
        self.interp_order = interp_order  # to be used in advection
        self.velo_interp_mode = "wrap" if self.boundary_condition == BoundaryCondition.PERIODIC else "constant"
        self.scalar_interp_mode = "wrap" if self.boundary_condition == BoundaryCondition.PERIODIC else "nearest"

        self.external_force_type = external_force_type

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
                self.velocities[:, :, i], mid_point_pos, order=self.interp_order, mode=self.velo_interp_mode
            )
        pos = np.stack(
            [
                self.pos_x - dt * mid_point_velocities[:, :, 0],
                self.pos_y - dt * mid_point_velocities[:, :, 1]
            ], axis=0
        )
        
        # find the new velocities and texture at each pixel/cell according to pos
        for i in range(3):
            self.img[:, :, i] = ndimage.map_coordinates(
                self.img[:, :, i], pos,
                order=self.interp_order,
                mode=self.scalar_interp_mode
            )
        for i in range(2):
            self.velocities[:, :, i] = ndimage.map_coordinates(
                self.velocities[:, :, i], pos,
                order=self.interp_order,
                mode=self.velo_interp_mode
            )
    
    def visualize_velocities(self, out_fpath = None):
        fig, ax = plt.subplots(figsize=(8,8))
        ax.quiver(
            self.pos_x,
            self.pos_y,
            self.velocities[:, :, 0],
            self.velocities[:, :, 1]
        )
        ax.set_aspect("equal")
        ax.invert_yaxis()
        if out_fpath is None:
            fig.show()
        else:
            os.makedirs(os.path.dirname(out_fpath), exist_ok=True)
            fig.savefig(out_fpath)
        plt.close()
    
    def simulate(
        self, num_iterations, dt, density, viscosity,
        diffusion_constant, dissipation_rate, 
        visualize_velocity, create_video
    ):
        out_color_fpaths = []
        out_velocity_fpaths = []
        for time_step in range(num_iterations):
            # Step 1: add external forces
            # add external forces to velocities based on how white each pixel is
            external_forces = construct_external_forces_by_type(self.img, self.external_force_type)
            self.add_external_forces(self.velocities, external_forces, dt)

            # we can optionally add additional "forces" to the scalar field, but here we don't add additional change
            # to the color
            # self.add_external_force(self.img, optional_rgb_external_forces, dt)

            # Step 2: advect both velocities and scalar/RGB values
            self.advect(dt)

            # Step 3: diffusion for both velocities and scalar/RGB values
            if viscosity != 0:
                for i in range(2):
                    self.velocities[:, :, i] = self.velo_diffusion_solver.diffuse(self.velocities[:, :, i], dt, viscosity)
            if diffusion_constant != 0:
                for i in range(3):
                    self.img[:, :, i] = self.scalar_diffusion_solver.diffuse(self.img[:, :, i], dt, diffusion_constant)

            # Step 4: pressure solve for velocity & dissipate scalar field
            # here we use a staggered MAC grid as the solver, in place of the FISHPAK subroutines mentioned in the paper
            # pressure solve
            self.pressure_solver.init_velo_from_img(self.velocities)  # project current velocities to the MAC grid
            self.pressure_solver.pressure_solve(dt, density)
            self.velocities = self.pressure_solver.convert_velo_on_img()
            # dissipation
            if dissipation_rate != 0:
                self.img = self.img / (1 + dt * dissipation_rate)

            # save image
            out_fpath = os.path.join(self.outdir, "colors", f"{time_step:06d}.png")
            os.makedirs(os.path.dirname(out_fpath), exist_ok=True)
            cv2.imwrite(out_fpath, self.img)
            out_color_fpaths.append(out_fpath)
        
            if visualize_velocity:
                out_fpath_velo = os.path.join(self.outdir, "velocities", f"{time_step:06d}.png")
                self.visualize_velocities(out_fpath_velo)
                out_velocity_fpaths.append(out_fpath_velo)

        if create_video:
            create_video_from_img_fpths(out_color_fpaths, os.path.join(self.outdir, "colors", "out.mp4"))
            create_video_from_img_fpths(out_color_fpaths[::-1], os.path.join(self.outdir, "colors", "reverse.mp4"))
            create_video_from_img_fpths(out_color_fpaths, os.path.join(self.outdir, "colors", "out.gif"))
            create_video_from_img_fpths(out_color_fpaths[::-1], os.path.join(self.outdir, "colors", "reverse.gif"))
            if visualize_velocity:
                create_video_from_img_fpths(out_velocity_fpaths, os.path.join(self.outdir, "velocities", "out.mp4"))
                create_video_from_img_fpths(out_velocity_fpaths, os.path.join(self.outdir, "velocities", "out.gif"))
            


def main(
    img_fpath="data/monroe.jpeg",
    outdir="out/monroe",
    boundary_condition="periodic",
    external_force_type="intensity",
    num_iterations=300,
    dt=0.1,
    density=1.0,
    viscosity=0.0,
    diffusion_constant=0.0,
    dissipation_rate=0.0,
    visualize_velocity=False,
    create_video=True,
):    
    img = cv2.imread(img_fpath)
    fluid = StableFluidImg(
        img, outdir, BoundaryCondition(boundary_condition),
        external_force_type=ExternalForceType(external_force_type)
    )
    fluid.simulate(
        num_iterations=num_iterations,
        dt=dt,
        density=density,
        viscosity=viscosity,
        diffusion_constant=diffusion_constant,
        dissipation_rate=dissipation_rate,
        visualize_velocity=visualize_velocity,
        create_video=create_video,
    )


if __name__ == "__main__":
    Fire({
        "simulate": main
    })

