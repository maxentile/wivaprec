import mdtraj as md
import numpy as np

traj = md.load("samples_from_model.h5")
print(np.max(md.rmsd(traj, traj)))

from msmbuilder.example_datasets import AlanineDipeptide

traj = AlanineDipeptide().get().trajectories[0]
print(md.rmsd(traj, traj[:10]))
print(np.max(md.rmsd(traj, traj)))
