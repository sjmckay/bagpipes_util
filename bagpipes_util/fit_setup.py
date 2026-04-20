import numpy as np
import matplotlib.pyplot as plt

from astropy.coordinates import SkyCoord as SC
from astropy.table import Table
import astropy.units as u

import bagpipes as pipes



def default_instructions():
    delayed = {}                                  
    delayed["age"] = (0.001, 14.)
    delayed["tau"] = (0.1, 9.)    
    delayed["massformed"] = (6., 13.)            
    delayed["metallicity"] = (0.2,2.5) 

    dust = {}                                 
    dust["type"] = "CF00"                
    dust["n"] = 0.7             
    dust["Av"] = (0.0, 6)
    dust["qpah"] = (0.47,4.5)          
    dust["umin"] = (1,25.)         
    dust["gamma"] = (0.01,0.3)      
    dust["eta"] = 2.


    burst = {}                                 
    burst["age"] = (0.05,1)                      
    burst["metallicity"] = 'delayed:metallicity'#(0.2,1.2)        
    burst["massformed"] = (0, 13.)            

    nebular = {}
    nebular["logU"] = (-3.99, -1.0)


    fit_instructions={}
    fit_instructions['delayed'] = delayed
    fit_instructions["burst"] = burst           
    fit_instructions["redshift"] = (0., 10.)     
    fit_instructions["dust"] =dust
    fit_instructions["nebular"] = nebular
    fit_instructions["t_bc"] = 0.01

    return fit_instructions


def generate_flat_filter(filename, flo, fhi):
    llo = (fhi*u.GHz).to(u.AA,equivalencies=u.spectral())
    lhi = (flo*u.GHz).to(u.AA,equivalencies=u.spectral())
    print(filename)
    wavs = np.linspace(llo.value, lhi.value, 200)
    with open(filename, 'w+') as f:
        for wav in wavs:
            f.writelines(f'{wav}    1.0\n')

            
def load_data(ID, phot=None):
    try:
        n = int(ID)
        row = phot[n-1]
    except:
        raise ValueError("Invalid ID passed to load_phot, or phot doesn't exist")

    fluxes = np.array([row[n] for n in row.colnames[2:] if 'err' not in n]) 
    errs = np.array([row[n] for n in row.colnames[2:] if 'err' in n])
    
    for i, (flux,err) in enumerate(zip(fluxes, errs)):
        # if any infs or nans (or -99s), blow up err. I.e. not observed
        if ((err < -98 and flux < -98) or (err==0.0 and flux==0.0) or (not np.isfinite(flux)) or (not np.isfinite(err))):
            fluxes[i] = 0.0
            errs[i] = 1e99
        # if positive err and flux, add in error floor and calibration error
        if err > 0 and flux > 0:
            if flux/err>3.0: #this is to make an "upper limit" if flux is less than 3 sigma... may change
                if err < 0.05*flux: errs[i] = 0.05*flux
                if np.isfinite(err) and flux>0:
                    errs[i] = np.sqrt(err**2 + (0.05*flux)**2)
            else:
                fluxes[i] = 0.0
                errs[i] = 3*np.abs(err)
        # if negative flux and positive err or vice versa, make "upper limit" with 3 sigma error
        if (err < 0 and flux > 0) or (flux<0 and err > 0):
            fluxes[i] = 0.0
            errs[i] = 3*np.abs(err)
    return np.stack((fluxes, errs),axis=0).T


def run_pipes(run_name, phot, path_to_filters, fit_instructions=None):
    if fit_instructions is None: fit_instructions = default_instructions()
    filt_list = [path_to_filters+ '_'.join(l.split('/')) + '.dat' 
            for l in phot.colnames[2:] if 'err' not in l]

    redshifts = np.array(phot['redshift'])
    gids = range(1,len(phot)+1)
    cat_fit = pipes.fit_catalogue(IDs=gids,
                            fit_instructions=fit_instructions,
                            load_data=load_data,
                            spectrum_exists=False,
                            make_plots=True,
                            cat_filt_list=filt_list,
                            redshifts=redshifts, 
                            run=run_name,
                            full_catalogue=True)

    cat_fit.fit(n_live=1000, sampler='multinest', mpi_serial=True, verbose=False, pool=14)