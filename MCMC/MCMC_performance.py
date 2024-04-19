import numpy as np
from scipy.stats import norm
import time
import pickle
import os

def rosenbrock(x, y):
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2

class RosenbrockLikelihood:
    def loglike(self, x, y):
        return -rosenbrock(x, y)

def proposal_distribution_3(current_state):
    step_sizes = np.array([0.5, 0.5])
    proposal_step = np.random.normal(0, step_sizes)
    return current_state + proposal_step


class MCMC_MH:
    def __init__(self, likelihood, proposal_distribution, priors, num_samples, num_chains, stepsize=0.5, burnin_ratio=0.05, resume=False):
        self.likelihood = likelihood
        self.proposal_distribution = proposal_distribution
        self.priors = priors
        self.num_samples = num_samples
        self.num_chains = num_chains
        self.stepsize = stepsize
        self.burnin = int(burnin_ratio * num_samples)
        self.checkpoint_interval = 250
        self.resume = resume
        self.R_minus_one = None

        self.count = 0

        self.curr_state = self.generate_initial_states()
        self.curr_likeli = [likelihood.loglike(*state) for state in self.curr_state]
        self.samples = [[] for _ in range(num_chains)]
        self.likelihoods = [[] for _ in range(num_chains)]
        self.accepteds = [0 for _ in range(num_chains)]
        self.Rminusones = []


        if resume:
            self.load_checkpoint()
        elif os.path.exists('./checkpoints/PerformanceTest/checkpoint_burnin.pkl') or os.path.exists('./checkpoints/PerformanceTest/checkpoint_production.pkl'):
            raise ValueError("The checkpoint files 'checkpoint_burnin.pkl' and 'checkpoint_production.pkl' should not exist when starting a new run.")

    def generate_initial_states(self):
        initial_states = []
        for _ in range(self.num_chains):
            state = [prior.rvs() for prior in self.priors]
            initial_states.append(state)
        return initial_states

    def save_checkpoint(self, iteration, burn_in=False):
        if not os.path.exists('./checkpoints/PerformanceTest/'):
            os.makedirs('./checkpoints/PerformanceTest/')
        filename = f'./checkpoints/PerformanceTest/checkpoint_{"burnin" if burn_in else "production"}.pkl'
        with open(filename, 'wb') as f:
            pickle.dump((self.curr_state, self.samples, self.likelihoods, self.R_minus_one, self.count), f)

    def load_checkpoint(self):
        checkpoint_file_prod = './checkpoints/PerformanceTest/checkpoint_production.pkl'
        checkpoint_file_burnin = './checkpoints/PerformanceTest/checkpoint_burnin.pkl'
        
        if os.path.exists(checkpoint_file_prod):
            with open(checkpoint_file_prod, 'rb') as f:
                self.curr_state, self.samples, self.likelihoods, self.R_minus_one, self.count = pickle.load(f)
        elif os.path.exists(checkpoint_file_burnin):
            with open(checkpoint_file_burnin, 'rb') as f:
                self.curr_state, self.samples, self.likelihoods, self.R_minus_one, self.count = pickle.load(f)
        else:
            raise ValueError("No checkpoint files found to resume from.")

    def burn_in(self):
        start_time = time.time()
        if self.resume:
            self.load_checkpoint()

        for i in range(self.burnin):
            for j in range(self.num_chains):
                self.mcmc_updater(j)
            if (i + 1) % self.checkpoint_interval == 0:
                self.save_checkpoint(i + 1, burn_in=True)
                elapsed_time = time.time() - start_time
                print(f"Completed {i+1} burn-in steps in {elapsed_time} seconds")

    def mcmc_updater(self, chain_index):
        self.count += 1
        proposal_state = self.proposal_distribution(self.curr_state[chain_index])
        proposal_state = [max(0, value) for value in proposal_state]

        try:
            prop_loglikeli = self.likelihood.loglike(*proposal_state)
        except Exception as e:
            print(f"An error occurred: {e}")
            prop_loglikeli = -np.inf

        accept_crit = prop_loglikeli - self.curr_likeli[chain_index]
        accept_threshold = np.log(np.random.uniform(0, 1))

        if accept_crit > accept_threshold:
            self.curr_state[chain_index], self.curr_likeli[chain_index] = proposal_state, prop_loglikeli


        return self.curr_state[chain_index], self.curr_likeli[chain_index]

    def metropolis_hastings(self):
        start_time = time.time()
        self.burn_in()

        if self.resume:
            self.load_checkpoint()

        for i in range(self.count, self.num_samples):
            for j in range(self.num_chains):
                self.curr_state[j], self.curr_likeli[j] = self.mcmc_updater(j)
                self.samples[j].append(self.curr_state[j])
                self.likelihoods[j].append(self.curr_likeli[j])

            if (i + 1) % self.checkpoint_interval == 0:
                self.save_checkpoint(i + 1)
                print(f"Completed {i+1} steps in {time.time() - start_time} seconds")
            if (i + 1) % 50 == 0:
                self.calculate_convergence()

        return self.samples

    def calculate_convergence(self):
        chain_means = [np.mean(np.array(chain), axis=0) for chain in self.samples]
        chain_vars = [np.var(np.array(chain), axis=0, ddof=1) for chain in self.samples]

        grand_mean = np.mean(chain_means, axis=0)

        B = np.sum((chain_means - grand_mean)**2, axis=0) * len(self.samples[0]) / (len(self.samples) - 1)
        W = np.mean(chain_vars, axis=0)

        var_plus = (len(self.samples[0]) - 1) / len(self.samples[0]) * W + B / len(self.samples[0])
        self.R_minus_one = (var_plus / W - 1).mean()
        print(f'R-1 statistic: {self.R_minus_one}')      
        self.Rminusones.append(self.R_minus_one)


if __name__ == "__main__":
    likelihood_function = RosenbrockLikelihood()
    priors = [
        norm(loc=0, scale=1),
        norm(loc=0, scale=1)
    ]
    mcmc = MCMC_MH(likelihood_function, proposal_distribution_3, priors, 10000, num_chains=4, stepsize=0.1, burnin_ratio=0.00, resume=False)
    mcmc.metropolis_hastings()