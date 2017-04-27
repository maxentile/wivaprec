import mdtraj as md
import simtk.openmm as mm
from autograd import numpy as np
from autograd.core import primitive
from autograd.util import quick_grad_check
from openmmtools.integrators import GHMCIntegrator, kB
from simtk import unit
from simtk.openmm import app


def make_openmm_system_autograd_compatible(topology, system, initial_positions, temperature=298 * unit.kelvin):
    """Given the specification of an openmm system and a bath temperature, and return a flattened function
    that computes the unnormalized log Gibbs density, and also defines its gradient for compatibility with autograd.
    """

    # Can also compute these on the GPU eventually.
    platform = mm.Platform.getPlatformByName("Reference")

    ghmc = GHMCIntegrator(temperature=temperature)

    beta = 1.0 / (temperature * kB)
    sim = app.Simulation(topology, system, ghmc, platform)
    sim.context.setPositions(initial_positions)

    n_atoms = len(initial_positions)

    def unflatten(x):
        """Given an (n_atoms * 3,)array, unpack into an (n_atoms, 3) array"""
        xyz = x.reshape((n_atoms, 3))
        return xyz

    @primitive
    def flat_log_q(x):
        """Use OpenMM to compute minus the reduced potential at x."""
        sim.context.setPositions(unflatten(x))
        return - sim.context.getState(getEnergy=True).getPotentialEnergy() * beta

    def grad_flat_log_q(x):
        """Use OpenMM to compute minus the gradient of the reduced potential at x."""
        sim.context.setPositions(unflatten(x))
        g = (sim.context.getState(getForces=True).getForces(asNumpy=True) * beta)
        return g.value_in_unit(g.unit).flatten()

    def make_grad_flat_log_q(ans, x):
        def gradient(g):
            return grad_flat_log_q(x)

        return gradient

    flat_log_q.defgrad(make_grad_flat_log_q)

    flat_pos = initial_positions.value_in_unit(initial_positions.unit).flatten()
    quick_grad_check(flat_log_q, flat_pos)

    return flat_log_q


def hold_some_dof_constant(flat_log_q, init_pos, atoms_to_restrain):
    """To isolate one difficulty of the problem, fix some degrees of freedom."""
    n_atoms = len(init_pos)
    n_restrained_atoms = len(atoms_to_restrain)

    # create an array of integers that will shuffle indices appropriately
    constrained_atom_indices = np.array(sorted(atoms_to_restrain))
    unconstrained_atom_indices = np.array(sorted(set(range(n_atoms)) - set(atoms_to_restrain)))

    def unflatten_constrained(x):
        xyz = x.reshape((n_atoms - n_restrained_atoms, 3))
        return xyz

    def constrained_log_q(x):
        unf_x = unflatten_constrained(x)
        xyz = np.zeros((n_atoms, 3))
        xyz[constrained_atom_indices] = init_pos[constrained_atom_indices]
        xyz[unconstrained_atom_indices] = unf_x
        return flat_log_q(xyz)

    return constrained_log_q()


def convert_samples_to_trajectory(x, openmm_topology):
    return md.Trajectory(x, md.Topology().from_openmm(openmm_topology))
