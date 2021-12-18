import numpy as np
import cv2
import os
from external_forces import apply_external_forces
from fire import Fire
from scipy import ndimage

from staggered_grid import Grid


class StableFluidImg():
    """Treat each pixel in an image as a particle and perform simulation accordingly"""
    def __init__(self, img, outdir):
        self.img = img  # (nx, ny, 3)
        self.nx = self.img.shape[0]
        self.ny = self.img.shape[1]
        self.velocities = np.zeros([self.nx, self.ny, 2])
        self.grid = Grid(self.nx, self.ny, 1.0)

        self.outdir = outdir
        os.makedirs(outdir, exist_ok=True)

        x = np.arange(self.nx) * 1.0
        y = np.arange(self.ny) * 1.0
        self.X_pos, self.Y_pos = np.meshgrid(x, y, indexing="ij")


    def advect(self, dt):
        x = np.linspace(0,1,self.nx, endpoint=False) 
        y = np.linspace(0,1,self.ny, endpoint=False) 
        X, Y = np.meshgrid(x,y, indexing='ij')
        coords = np.stack( [(X - self.velocities[:,:,0]*dt)* self.nx, (Y - self.velocities[:,:,1]*dt)*self.ny], axis=0)
        # coords = np.stack([self.X_pos - self.velocities[:, :, 0] * dt, self.Y_pos - self.velocities[:, :, 1] * dt], axis=0)
        for i in range(3):
            self.img[:, :, i] = ndimage.map_coordinates(self.img[:, :, i], coords, order=5, mode="wrap")
        for i in range(2):
            self.velocities[:, :, i] = ndimage.map_coordinates(self.velocities[:, :, i], coords, order=5, mode="wrap")
    
    def simulate(self, num_iterations=300, dt=0.01, density=1.0):
        for time_step in range(num_iterations):
            # advection
            self.advect(dt)

            # apply external forces
            external_forces = np.linalg.norm(self.img, axis=2) * 0.001
            apply_external_forces(self.velocities, external_forces[:, :, None], dt)

            # pressure solve
            self.grid.init_velo_from_img(self.velocities)
            self.grid.pressure_solve(dt, density)
            self.velocities += self.grid.convert_velo_on_img(return_delta=True)  # use the delta to appear more splashy
            # self.velocities = self.grid.convert_velo_on_img()

            # save image
            cv2.imwrite(os.path.join(self.outdir, f"{time_step:06d}.png"), self.img)


def main(img_fpath="data/monroe.jpeg", outdir="out/monroe", num_iterations=300, dt=0.01, density=1.0):
    img = cv2.imread(img_fpath)
    fluid = StableFluidImg(img, outdir)
    fluid.simulate(num_iterations=num_iterations, dt=dt, density=density)


if __name__ == '__main__':
    Fire({
        "simulate": main
    })

