import camb
from camb import model, initialpower
import numpy as np
import matplotlib.pyplot as plt

def run_camb():
    # Set up the parameters for a basic LambdaCDM cosmology
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=69.5, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06)
    pars.InitPower.set_params(As=2e-9, ns=0.965, r=0)
    pars.set_for_lmax(2000, lens_potential_accuracy=0)
    
    # Set WantTransfer to True
    pars.WantTransfer = True
    
    # Calculate results for these parameters
    results = camb.get_results(pars)
    
    # Get the CMB power spectra
    powers = results.get_cmb_power_spectra(pars)
    for name, cls in powers.items():
        print(name, cls[:3])  # Print the first few multipoles


    # # Calculate matter power spectrum at redshift z=0
    # k = np.logspace(-4, 1, num=300)  # Define a range of wave numbers
    # pk = results.get_matter_power_spectrum(minkh=1e-4, maxkh=1, npoints=300)
    # print("Matter power spectrum at z=0:", pk)

    #     # Plot the matter power spectrum
    # plt.loglog(k, pk[1], label='z=0')
    # plt.xlabel('Wave number k (h/Mpc)')
    # plt.ylabel('Power Spectrum P(k) [(Mpc/h)^3]')
    # plt.title('Matter Power Spectrum')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

import time

if __name__ == "__main__":
    start_time = time.time()
    run_camb()
    end_time = time.time()
    print("Execution time of run_camb: ", end_time - start_time, "seconds")