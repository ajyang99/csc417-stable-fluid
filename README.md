# csc417-stable-fluid

This is a Python implementation of Jos Stam's 1999 paper [Stable Fluids](https://graphics.stanford.edu/courses/cs448-01-spring/papers/stam.pdf).

A 5-min video of the results is [here](https://youtu.be/DIZUKA8GSEc).

### Dependencies

The implementation is written in Python and tested with Python 3.7.4. The following packages are used and can be installed with

    python -m pip install --user numpy scipy matplotlib ipython jupyter pandas sympy nose

    pip install imageio imageio-ffmpeg opencv-python matplotlib fire


### Running the solver

We implement for both 2D and 3D applications. We support various types of external forces. We use the Python `fire` package to support various command line arguments.

TO recreate the 2D visualization with external force proportional to image pixel intensities, we use the command

    python fluid_img.py simulate --dt=0.01 --density=0.01

We can optionally set `--boundary_condition=fixed` to enforced the fixed boundary condition.

To recreate the visualization with external force based on a central force constantly applied to the bottom center of the image in the upward direction with viscosity 10.0, we use the command

    python fluid_img.py simulate --visualize_velocity=True --img_fpath=data/panda_small.jpeg --outdir=out/panda_small_vis10_bot/ --viscosity=10 --external_force_type=bot_push_up

Additionally we can set `--diffusion_constant=5.0` and `-dissipation_constant=5.0` for the scalar field.

To recreate the 3D visualization with the points initialized to be red in the center of the block, run

    python fluid_voxel.py simulate --input="center" --outdir=out/3d_voxels/center

To recreate the 3D visualization with the pretty colors, run

    python fluid_voxel.py simulate --input="pretty" --outdir=out/3d_voxels/pretty