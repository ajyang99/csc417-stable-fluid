import numpy as np

# Update the per-particle velocity and position due to external forces
def apply_external_forces(v: np.ndarray, f: np.ndarray, dt: float) -> None:
    v += dt * f