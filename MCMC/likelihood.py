'''
Python version of Planck's plik-lite likelihood with the option to include
the low-ell temperature as two Gaussian bins

The official Planck likelihoods are availabe at https://pla.esac.esa.int/
The papers describing the Planck likelihoods are
Planck 2018: https://arxiv.org/abs/1907.12875
Planck 2015: https://arxiv.org/abs/1507.02704

'''
import numpy as np
from scipy.io import FortranFile
import scipy.linalg

def main():
    TTTEEE2018=PlanckLitePy(year=2018, spectra='TTTEEE', use_low_ell_bins=False)
    TTTEEE2018.test()

    TTTEEE2018_lowTTbins=PlanckLitePy(year=2018, spectra='TTTEEE', use_low_ell_bins=True)
    TTTEEE2018_lowTTbins.test()

    TT2018=PlanckLitePy(year=2018, spectra='TT', use_low_ell_bins=False)
    TT2018.test()

    TT2018_lowTTbins=PlanckLitePy(year=2018, spectra='TT', use_low_ell_bins=True)
    TT2018_lowTTbins.test()


class PlanckLitePy:
    def __init__(self, data_directory='data', year=2018, spectra='TT', use_low_ell_bins=False):
        '''
        data_directory = path from where you are running this to the folder
          containing the planck2015/8_low_ell and planck2015/8_plik_lite data
        year = 2015 or 2018
        spectra = TT for just temperature or TTTEEE for temperature (TT),
          E mode (EE) and cross (TE) spectra
        use_low_ell_bins = True to use 2 low ell bins for the TT 2<=ell<30 data
          or False to only use ell>=30
        '''
        self.year=year
        self.spectra=spectra
        self.use_low_ell_bins=use_low_ell_bins #False matches Plik_lite - just l>=30

        if self.use_low_ell_bins:
            self.nbintt_low_ell=2
            self.plmin_TT=2
        else:
            self.nbintt_low_ell=0
            self.plmin_TT=30
        self.plmin=30
        self.plmax=2508
        self.calPlanck=1

        if year==2015:
            self.data_dir=data_directory+'/planck2015_plik_lite/'
            version=18
        elif year==2018:
            self.data_dir=data_directory+'/planck2018_plik_lite/'
            version=22
        else:
            print('Year must be 2015 or 2018')
            return 1

        if spectra=='TT':
            self.use_tt=True
            self.use_ee=False
            self.use_te=False
        elif spectra=='TTTEEE':
            self.use_tt=True
            self.use_ee=True
            self.use_te=True
        else:
            print('Spectra must be TT or TTTEEE')
            return 1

        self.nbintt_hi = 215 #30-2508   #used when getting covariance matrix
        self.nbinte = 199 #30-1996
        self.nbinee = 199 #30-1996
        self.nbin_hi=self.nbintt_hi+self.nbinte+self.nbinee

        self.nbintt=self.nbintt_hi+self.nbintt_low_ell #mostly want this if using low ell
        self.nbin_tot=self.nbintt+self.nbinte+self.nbinee

        self.like_file = self.data_dir+'cl_cmb_plik_v'+str(version)+'.dat'
        self.cov_file  = self.data_dir+'c_matrix_plik_v'+str(version)+'.dat'
        self.blmin_file = self.data_dir+'blmin.dat'
        self.blmax_file = self.data_dir+'blmax.dat'
        self.binw_file = self.data_dir+'bweight.dat'

        # read in binned ell value, C(l) TT, TE and EE and errors
        # use_tt etc to select relevant parts
        self.bval, self.X_data, self.X_sig=np.genfromtxt(self.like_file, unpack=True)
        self.blmin=np.loadtxt(self.blmin_file).astype(int)
        self.blmax=np.loadtxt(self.blmax_file).astype(int)
        self.bin_w=np.loadtxt(self.binw_file)

        if self.use_low_ell_bins:
            self.data_dir_low_ell=data_directory+'/planck'+str(year)+'_low_ell/'
            self.bval_low_ell, self.X_data_low_ell, self.X_sig_low_ell=np.genfromtxt(self.data_dir_low_ell+'CTT_bin_low_ell_'+str(year)+'.dat', unpack=True)
            self.blmin_low_ell=np.loadtxt(self.data_dir_low_ell+'blmin_low_ell.dat').astype(int)
            self.blmax_low_ell=np.loadtxt(self.data_dir_low_ell+'blmax_low_ell.dat').astype(int)
            self.bin_w_low_ell=np.loadtxt(self.data_dir_low_ell+'bweight_low_ell.dat')

            self.bval=np.concatenate((self.bval_low_ell, self.bval))
            self.X_data=np.concatenate((self.X_data_low_ell, self.X_data))
            self.X_sig=np.concatenate((self.X_sig_low_ell, self.X_sig))

            self.blmin_TT=np.concatenate((self.blmin_low_ell, self.blmin+len(self.bin_w_low_ell)))
            self.blmax_TT=np.concatenate((self.blmax_low_ell, self.blmax+len(self.bin_w_low_ell)))
            self.bin_w_TT=np.concatenate((self.bin_w_low_ell, self.bin_w))

        else:
            self.blmin_TT=self.blmin
            self.blmax_TT=self.blmax
            self.bin_w_TT=self.bin_w


        self.fisher=self.get_inverse_covmat()

    def get_inverse_covmat(self):
        # Read full covmat
        f = FortranFile(self.cov_file, 'r')
        covmat = f.read_reals(dtype=float).reshape((self.nbin_hi, self.nbin_hi))
        covmat = np.maximum(covmat, covmat.T)  # Make the matrix symmetric

        # Define a dictionary to map conditions to calculations
        conditions = {
            (True, False, False): (self.nbintt_hi, 0),
            (False, False, True): (self.nbinte, self.nbintt_hi),
            (False, True, False): (self.nbinee, self.nbintt_hi + self.nbinte),
            (True, True, True): (self.nbin_hi, 0)
        }

        # Select relevant covmat
        bin_no, start = conditions.get((self.use_tt, self.use_ee, self.use_te), (None, None))
        if bin_no is None:
            print("not implemented")
            return

        end = start + bin_no
        cov = covmat[start:end, start:end]

        # Invert high ell covariance matrix (cholesky decomposition should be faster)
        fisher = scipy.linalg.cho_solve(scipy.linalg.cho_factor(cov), np.identity(bin_no)).T

        if self.use_low_ell_bins:
            bin_no += self.nbintt_low_ell
            inv_covmat_with_lo = np.zeros(shape=(bin_no, bin_no))
            inv_covmat_with_lo[0:2, 0:2] = np.diag(1. / self.X_sig_low_ell**2)
            inv_covmat_with_lo[2:, 2:] = fisher
            fisher = inv_covmat_with_lo

        return fisher



    # def loglike(self, Cltt, Clte, Clee, ellmin=2):
        

    #     # If the input is Dl's, convert to Cl's
    #     #convert model Dl's to Cls then bin them
    #     # ls=np.arange(len(Dltt))+ellmin
    #     # fac=ls*(ls+1)/(2*np.pi)
    #     # Cltt=Dltt/fac
    #     # Clte=Dlte/fac
    #     # Clee=Dlee/fac



    def binning(self, Cl, blmin, blmax, bin_w, nbins, ellmin):
        Cl_bin = np.zeros(nbins)
        for i in range(nbins):
            Cl_bin[i] = np.sum(Cl[blmin[i]+self.plmin-ellmin:blmax[i]+self.plmin+1-ellmin]*bin_w[blmin[i]:blmax[i]+1])
        return Cl_bin

    def select_diff_vec(self, Y):
        conditions = {
            (True, False, False): (self.nbintt, 0),
            (False, False, True): (self.nbinte, self.nbintt),
            (False, True, False): (self.nbinee, self.nbintt + self.nbinte),
            (True, True, True): (self.nbin_tot, 0)
        }
        bin_no, start = conditions.get((self.use_tt, self.use_ee, self.use_te), (None, None))
        if bin_no is None:
            print("not implemented")
            return None
        end = start + bin_no
        return Y[start:end]

    def loglike(self, Dltt, Dlte, Dlee, ellmin=2):
        # Convert model Dl's to Cls then bin them
        ls = np.arange(len(Dltt)) + ellmin
        fac = ls * (ls + 1) / (2 * np.pi)
        Cltt, Clte, Clee = Dltt / fac, Dlte / fac, Dlee / fac

        # Bin the Cls
        Cltt_bin = self.binning(Cltt, self.blmin_TT, self.blmax_TT, self.bin_w_TT, self.nbintt, ellmin)
        Clte_bin = self.binning(Clte, self.blmin, self.blmax, self.bin_w, self.nbinte, ellmin)
        Clee_bin = self.binning(Clee, self.blmin, self.blmax, self.bin_w, self.nbinee, ellmin)

        # Construct the model
        X_model = np.zeros(self.nbin_tot)
        X_model[:self.nbintt] = Cltt_bin / self.calPlanck**2
        X_model[self.nbintt:self.nbintt+self.nbinte] = Clte_bin / self.calPlanck**2
        X_model[self.nbintt+self.nbinte:] = Clee_bin / self.calPlanck**2

        # Calculate the difference vector
        Y = self.X_data - X_model
        diff_vec = self.select_diff_vec(Y)
        if diff_vec is None:
            return None

        return -0.5 * diff_vec.dot(self.fisher.dot(diff_vec))


    def test(self):
        ls, Dltt, Dlte, Dlee = np.genfromtxt('data/Dl_planck2015fit.dat', unpack=True)
        ellmin=int(ls[0])
        loglikelihood=self.loglike(Dltt, Dlte, Dlee, ellmin)

        if self.year==2018 and self.spectra=='TTTEEE' and not self.use_low_ell_bins:
            print('Log likelihood for 2018 high-l TT, TE and EE:')
            expected = -291.33481235418026
            # Plik-lite within cobaya gives  -291.33481235418003
        elif self.year==2018 and self.spectra=='TTTEEE' and self.use_low_ell_bins:
            print('Log likelihood for 2018 high-l TT, TE and EE + low-l TT bins:')
            expected = -293.95586501795134
        elif self.year==2018 and self.spectra=='TT' and not self.use_low_ell_bins:
            print('Log likelihood for 2018 high-l TT:')
            expected = -101.58123068722583
            #Plik-lite within cobaya gives -101.58123068722568
        elif self.year==2018 and self.spectra=='TT' and self.use_low_ell_bins:
            print('Log likelihood for 2018 high-l TT + low-l TT bins:')
            expected = -104.20228335099686

        elif self.year==2015 and self.spectra=='TTTEEE' and not self.use_low_ell_bins:
            print('NB: Don\'t use 2015 polarization!')
            print('Log likelihood for 2015 high-l TT, TE and EE:')
            expected = -280.9388125627618
            # Plik-lite within cobaya gives  -291.33481235418003
        elif self.year==2015 and self.spectra=='TTTEEE' and self.use_low_ell_bins:
            print('NB: Don\'t use 2015 polarization!')
            print('Log likelihood for 2015 high-l TT, TE and EE + low-l TT bins:')
            expected = -283.1905700256343
        elif self.year==2015 and self.spectra=='TT' and not self.use_low_ell_bins:
            print('Log likelihood for 2015 high-l TT:')
            expected = -102.34403873289027
            #Plik-lite within cobaya gives -101.58123068722568
        elif self.year==2015 and self.spectra=='TT' and self.use_low_ell_bins:
            print('Log likelihood for 2015 high-l TT + low-l TT bins:')
            expected = -104.59579619576277
        else:
            expected=None

        print('Planck-lite-py:',loglikelihood)
        if(expected):
            print('expected:', expected)
            print('difference:', loglikelihood-expected, '\n')



if __name__=='__main__':
    main()