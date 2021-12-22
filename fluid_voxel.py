import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from enum import Enum
from fire import Fire
from scipy import ndimage

from staggered_grid_3d import PressureSolver, DiffusionSolver
from utils import create_video_from_img_fpths

class ExternalForceType(str, Enum):
    INTENSITY = "intensity"
    CENTER_PUSH_UP = "bot_push_up"

def construct_external_forces_by_type(voxel_grid, external_force_type):
    if external_force_type == ExternalForceType.INTENSITY:
        external_forces = np.linalg.norm(voxel_grid, axis=2) * 0.2
        external_forces = external_forces[:, :, None]
    elif external_force_type == ExternalForceType.CENTER_PUSH_UP:
        nx, ny, nz = voxel_grid.shape[:3]
        external_forces = np.zeros([nx, ny, nz, 3])
        external_forces[int(0.25 * nx):int(0.75 * nx), int(0.25 * ny):int(0.75 * ny), int(0.25 * nz):int(0.75 * nz), :] = np.array([1.0, 1.0, 1.0])
    else:
        raise ValueError(f"Unsupported external force type {external_force_type}")
    return external_forces


def create_pretty_voxel_grid():
    # prepare some coordinates
    nx, ny, nz = 8, 8, 8
    x, y, z = np.indices((nx, ny, nz))

    colors = np.zeros([nx, ny, nz, 3])
    cmap = cm.get_cmap("magma")
    colors[:, :, :] = cmap(np.sqrt((x / nx) ** 2 + (y / ny) ** 2 + (z / nz) ** 2) * 0.8)[..., :3]

    return colors


def create_center_block_voxel_grid():
    # prepare some coordinates
    nx, ny, nz = 8, 8, 8

    colors = np.zeros([nx, ny, nz, 3])
    # colors[..., :] = cm.get_cmap("magma")(0.99)[:3]

    # colors[3, 3, 3, :] = np.array([1.0, 0.0, 0.0])
    colors[2:6, 2:6, 2:6, :] = np.array([1.0, 0.0, 0.0])

    return colors


def plot_voxel(voxel_colors, out_fpath = None):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    x, y, z = np.indices(voxel_colors.shape[:3])
    plt_color = np.copy(voxel_colors)
    plt_color[plt_color > 1.0] = 1.0
    plt_color[plt_color < 0.0] = 0.0
    ax.scatter(x.reshape(-1), y.reshape(-1), z.reshape(-1), color=plt_color.reshape(-1, 3))

    if out_fpath is not None:
        os.makedirs(os.path.dirname(out_fpath), exist_ok=True)
        plt.savefig(out_fpath)
    else:
        plt.show()
    plt.close()


class StableFluidVoxel():
    """Apply the stable fluid algorithm to a 3D voxel grid.
    
    We treat each pixel of the image as a grid cell described in Section 3.1 of the paper.
    The 2D velocity field thus has the same width and height as the image, and the scalar field consists of the
    RGB values of the image.
    """
    def __init__(self, voxel_grid, outdir, external_force_type):
        self.nx = voxel_grid.shape[0]
        self.ny = voxel_grid.shape[1]
        self.nz = voxel_grid.shape[2]
        
        # Note like unlike the paper which uses two variables to store the old and updated values for each of
        # the velocity and scalar fields, we just use one variable for each field to store the most up-to-date
        # values, since we can just always update the variables directly and there is no point in saving the
        # old values.
        self.voxel_grid = voxel_grid  # scalar color field at cell center of shape (nx, ny, nz, 3)
        self.velocities = np.zeros([self.nx, self.ny, self.nz, 3])  # 3D velocity field
        
        self.pressure_solver = PressureSolver(self.nx, self.ny, self.nz, 1.0)  # staggered MAC grid for pressure solve
        self.velo_diffusion_solver = DiffusionSolver(self.nx, self.ny, self.nz, 1.0)  # staggered MAC grid for velocity field
        self.scalar_diffusion_solver = DiffusionSolver(self.nx, self.ny, self.nz, 1.0)  # staggered MAC grid for scalar field

        # positions of each pixel/cell on the image grid to be used to trace particle in advection
        # note that unlike the paper we don't add the 0.5 offset since scipy.ndimage assumes (0, 0) is cell center
        self.pos_x, self.pos_y, self.pos_z = np.indices([self.nx, self.ny, self.nz])

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
                self.pos_x - 0.5 * dt * self.velocities[..., 0],
                self.pos_y - 0.5 * dt * self.velocities[..., 1],
                self.pos_z - 0.5 * dt * self.velocities[..., 2],
            ], axis=0
        )
        # find the velocities at the midpoint with interpolation
        # we use scipy.ndimage to do the linear interpolation for us
        # we use the wrap mode since the image is a texture map that wraps around
        mid_point_velocities = np.zeros(self.velocities.shape)
        for i in range(3):
            mid_point_velocities[..., i] = ndimage.map_coordinates(
                self.velocities[..., i], mid_point_pos, order=2, mode="wrap"
            )
        pos = np.stack(
            [
                self.pos_x - dt * mid_point_velocities[..., 0],
                self.pos_y - dt * mid_point_velocities[..., 1],
                self.pos_z - dt * mid_point_velocities[..., 2],
            ], axis=0
        )
        
        # find the new velocities and texture at each pixel/cell according to pos
        for i in range(3):
            self.voxel_grid[..., i] = ndimage.map_coordinates(self.voxel_grid[..., i], pos, order=2, mode="wrap")
        for i in range(3):
            self.velocities[..., i] = ndimage.map_coordinates(self.velocities[..., i], pos, order=2, mode="wrap")
    
    def simulate(
        self, num_iterations, dt, density, viscosity,
        diffusion_constant, dissipation_rate, create_video
    ):
        out_color_fpaths = []
        out_velocity_fpaths = []
        for time_step in range(num_iterations):
            # Step 1: add external forces
            # add external forces to velocities based on how white each pixel is
            external_forces = construct_external_forces_by_type(self.voxel_grid, self.external_force_type)
            self.add_external_forces(self.velocities, external_forces, dt)

            # we can optionally add additional "forces" to the scalar field, but here we don't add additional change
            # to the color
            # self.add_external_force(self.voxel_grid, optional_rgb_external_forces, dt)

            # Step 2: advect both velocities and scalar/RGB values
            self.advect(dt)

            # Step 3: diffusion for both velocities and scalar/RGB values
            if viscosity != 0:
                for i in range(3):
                    self.velocities[..., i] = self.velo_diffusion_solver.diffuse(self.velocities[..., i], dt, viscosity)
            if diffusion_constant != 0:
                for i in range(3):
                    self.voxel_grid[..., i] = self.scalar_diffusion_solver.diffuse(self.voxel_grid[..., i], dt, diffusion_constant)

            # Step 4: pressure solve for velocity & dissipate scalar field
            # here we use a staggered MAC grid as the solver, in place of the FISHPAK subroutines mentioned in the paper
            # pressure solve
            self.pressure_solver.init_velo_from_voxel_grid(self.velocities)  # project current velocities to the MAC grid
            self.pressure_solver.pressure_solve(dt, density)
            self.velocities = self.pressure_solver.convert_velo_on_voxel_grid()
            # dissipation
            if dissipation_rate != 0:
                self.voxel_grid = self.voxel_grid / (1 + dt * dissipation_rate)

            # save image
            out_fpath = os.path.join(self.outdir, "colors", f"{time_step:06d}.png")
            os.makedirs(os.path.dirname(out_fpath), exist_ok=True)
            plot_voxel(self.voxel_grid, out_fpath)
            out_color_fpaths.append(out_fpath)

        if create_video:
            create_video_from_img_fpths(out_color_fpaths, os.path.join(self.outdir, "colors", "out.mp4"))
            create_video_from_img_fpths(out_color_fpaths[::-1], os.path.join(self.outdir, "colors", "reverse.mp4"))
            create_video_from_img_fpths(out_color_fpaths, os.path.join(self.outdir, "colors", "out.gif"))
            create_video_from_img_fpths(out_color_fpaths[::-1], os.path.join(self.outdir, "colors", "reverse.gif"))            


def main(
    input="pretty",
    outdir="out/3d_voxels/pretty",
    num_iterations=300,
    dt=0.1,
    density=1.0,
    viscosity=0.0,
    diffusion_constant=0.0,
    dissipation_rate=0.0,
    create_video=True,
):
    if input == "pretty":
        voxel_grid = create_pretty_voxel_grid()
        external_force_type = ExternalForceType.INTENSITY
    elif input == "center":
        voxel_grid = create_center_block_voxel_grid()
        external_force_type = ExternalForceType.CENTER_PUSH_UP
    else:
        raise ValueError(f"Unknown input {input}")
    # plot_voxel(voxel_grid)
    fluid = StableFluidVoxel(voxel_grid, outdir, external_force_type)
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

