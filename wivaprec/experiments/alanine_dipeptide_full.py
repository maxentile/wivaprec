import mdtraj as md
from autograd import numpy as np
from openmmtools.integrators import kB
from openmmtools.testsystems import AlanineDipeptideVacuum
from simtk import unit

from wivaprec.evaluation import metropolis_hastings
from wivaprec.models import InvertibleNeuralNet, WildVariationalApproximation
from wivaprec.testsystems.openmm_utils import make_openmm_system_autograd_compatible, convert_samples_to_trajectory

if __name__ == "__main__":
    testsystem = AlanineDipeptideVacuum()
    top, sys, pos = testsystem.topology, testsystem.system, testsystem.positions
    temperature = 298 * unit.kelvin
    target_log_q = make_openmm_system_autograd_compatible(top, sys, pos, temperature)
    flat_pos = pos.value_in_unit(pos.unit).flatten()


    def transform(x, scale=0.005):
        return flat_pos + x * scale


    def easier_target_log_q(x):
        """To make it easier, center the variational distribution on a known reasonable configuration...
        Now, try to learn a distribution around this
        """
        return target_log_q(transform(x))


    def inspect_samples(model, n_samples=1000):
        theta = model.optimization_history[-1][0]
        raw_samples = model.sample_zK(n_samples, theta)
        rescaled_samples = transform(raw_samples)
        xyz = rescaled_samples.reshape((n_samples, n_atoms, 3))

        traj = convert_samples_to_trajectory(xyz, top)
        rmsd = md.rmsd(traj, traj)
        print(np.min(rmsd), np.max(rmsd))
        beta = 1.0 / (temperature * kB)
        energies = [- target_log_q(frame) / beta for frame in rescaled_samples]
        print(np.min(energies), np.max(energies), np.mean(energies))

        traj.save_hdf5("samples_from_model.h5")


    n_atoms = len(pos)
    n_dim = n_atoms * 3
    np.random.seed(0)
    # flow = NormalizingFlow(K=10, n_dim=n_dim)
    # flow.fit(easier_target_log_q, n_iter=10000, report_interval=100, n_samples=2, step_size=0.0001, annealed=False)
    # inspect_samples(flow)

    wva = WildVariationalApproximation(n_dim, InvertibleNeuralNet(n_dim, n_hidden_layers=2))
    wva.fit(easier_target_log_q, n_iter=1000, report_interval=50, n_samples=1, step_size=0.001, l2_penalty=0,
            annealed=False)
    inspect_samples(wva)


    def evaluate_on_mcmc(model, target_log_q, proposal_sigma=1.0, ):
        theta = model.optimization_history[-1][0]

        def preconditioned_target_log_q(x):
            """target_log_q(f(x)) - |log jacobian(f)(x)|"""
            y, _, log_det_jac = model.forward_pass(x, theta)
            log_q = target_log_q(y)
            return (log_q + log_det_jac)

        x_0 = np.random.randn(n_dim)

        xs_naive, acc_rate_naive = metropolis_hastings(x_0, target_log_q, proposal_sigma=proposal_sigma)
        xs, acc_rate = metropolis_hastings(x_0, preconditioned_target_log_q, proposal_sigma=proposal_sigma)

        print(acc_rate, acc_rate_naive)

        # now, we should evaluate the overall mixing time, by projecting onto known slow d.o.f. from
        # molecular dynamics


        # now, let's collect a ton of samples from this approximate model, and see how they look
        # RMSD to start, etc.
        # also, let's see if we can make that output "scale" a tunable parameter...
