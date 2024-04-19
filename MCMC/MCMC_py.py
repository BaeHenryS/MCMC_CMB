import camb
import numpy as np
import os
import pickle
import time
from camb.model import CAMBParamRangeError
from scipy.stats import norm
from numba import jit

def calculate_power_spectra(params, lensed=True, accuracy_boost=0.5, l_sample_boost=0.5, l_accuracy_boost=0.5):
    # Update the params dictionary with lower accuracy settings
    params.update({
        'AccuracyBoost': accuracy_boost,
        'lSampleBoost': l_sample_boost,
        'lAccuracyBoost': l_accuracy_boost,
        # You can adjust these values or add other parameters to decrease accuracy further
        'DoLateRadTruncation': True
    })
    
    # Set parameters including the updated accuracy settings
    pars = camb.set_params(**params)
    
    # Get results from CAMB
    results = camb.get_results(pars)
    
    # Get CMB power spectra with specified units
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK')
    
    if lensed:
        cl = powers['total']
    else:
        cl = powers['unlensed_scalar']
    
    # Removing the monopole and dipole terms at l=0 and l=1
    cl_cut = cl[2:]
    cl_tt = cl_cut[:, 0]  # TT spectra
    cl_te = cl_cut[:, 3]  # TE spectra
    cl_ee = cl_cut[:, 1]  # EE spectra
    
    return cl_tt, cl_te, cl_ee

def process_params_and_calculate_spectra(array):
    array = array.copy()
    array[2] /= 100
    array[5] = np.exp(array[5]) / 1e10
    params = {
        'ombh2': array[0],
        'omch2': array[1],
        'cosmomc_theta': array[2],
        'tau': array[3],
        'ns': array[4],
        'As': array[5],
        'mnu': 0.06,
        'omk': 0,
        'halofit_version': 'mead',
        'lmax': 2800
    }
    return calculate_power_spectra(params)

import likelihood
from likelihood import PlanckLitePy

a = 15
def proposal_distribution_3(current_state):
    step_sizes = np.array([
        0.0001/a,  # ombh2
        0.001/a,  # omch2
        0.0002/a,  # theta_MC_100
        0.003/a,   # tau
        0.002/a,   # ns
        0.01/a    # log(10^10 As)
    ])
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
        self.curr_likeli = [likelihood.loglike(*process_params_and_calculate_spectra(state)) for state in self.curr_state]
        self.samples = [[] for _ in range(num_chains)]
        self.likelihoods = [[] for _ in range(num_chains)]
        self.accepteds = [0 for _ in range(num_chains)]
        self.Rminusones = []


        if resume:
            self.load_checkpoint()
        elif os.path.exists('./checkpoints/checkpoint_burnin.pkl') or os.path.exists('./checkpoints/checkpoint_production.pkl'):
            raise ValueError("The checkpoint files 'checkpoint_burnin.pkl' and 'checkpoint_production.pkl' should not exist when starting a new run.")

    def generate_initial_states(self):
        initial_states = []
        for _ in range(self.num_chains):
            state = [prior.rvs() for prior in self.priors]
            initial_states.append(state)
        return initial_states

    def save_checkpoint(self, iteration, burn_in=False):
        if not os.path.exists('./checkpoints'):
            os.makedirs('./checkpoints')
        filename = f'./checkpoints/checkpoint_{"burnin" if burn_in else "production"}.pkl'
        with open(filename, 'wb') as f:
            pickle.dump((self.curr_state, self.samples, self.likelihoods, self.Rminusones, self.count), f)

    def load_checkpoint(self):
        checkpoint_file_prod = './checkpoints/checkpoint_production.pkl'
        checkpoint_file_burnin = './checkpoints/checkpoint_burnin.pkl'
        
        if os.path.exists(checkpoint_file_prod):
            with open(checkpoint_file_prod, 'rb') as f:
                self.curr_state, self.samples, self.likelihoods, self.Rminusones, self.count = pickle.load(f)
        elif os.path.exists(checkpoint_file_burnin):
            with open(checkpoint_file_burnin, 'rb') as f:
                self.curr_state, self.samples, self.likelihoods, self.Rminusones, self.count = pickle.load(f)
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
            self.count += 1

    def mcmc_updater(self, chain_index):
        proposal_state = self.proposal_distribution(self.curr_state[chain_index])
        proposal_state = [max(0, value) for value in proposal_state]

        try:
            prop_loglikeli = self.likelihood.loglike(*process_params_and_calculate_spectra(proposal_state))
        except CAMBParamRangeError:
            prop_loglikeli = -np.inf
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

        print(self.count, self.num_samples)
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
            
            self.count += 1

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
    likelihood_function = PlanckLitePy(spectra='TTTEEE')
    priors = [
        norm(loc=0.0224, scale=np.sqrt(1)*0.00015),
        norm(loc=0.12, scale=np.sqrt(1)*0.0015),
        norm(loc=1.04109, scale=np.sqrt(1)*0.0004),
        norm(loc=0.055, scale=np.sqrt(1)*0.009),
        norm(loc=0.965, scale=np.sqrt(1)*0.007),
        norm(loc=3.05, scale=np.sqrt(1)*0.01)
    ]
    start = time.time()
    mcmc = MCMC_MH(likelihood_function, proposal_distribution_3, priors, 10, num_chains=4, stepsize=0.1, burnin_ratio=0.00, resume=False)
    mcmc.metropolis_hastings()
    print("Execution time: ", time.time() - start, "seconds")
