import numpy as np
import bagpipes as pipes
import matplotlib.pyplot as plt
import astropy.units as u
from copy import deepcopy
from scipy.interpolate import interp1d
from astropy.cosmology import Planck18 as cosmo

def load_galaxy_fit(ID, current_pipes, load_phot, filt_list, fit_instructions, advanced=True):
    gal = pipes.galaxy(ID=ID,load_data=load_phot,spectrum_exists=False,filt_list=filt_list)
    fit = pipes.fit(gal,run=current_pipes,fit_instructions=fit_instructions)
    fit.fit(verbose=False)
    if advanced: fit.posterior.get_advanced_quantities()
    return fit, gal


def get_maxlike_model_range(fit, filt_list):
    #get max likelihood model
    ml_index = np.argmax(fit.results["lnlike"])
    fitcopy = deepcopy(fit)
    fitcopy.fitted_model._update_model_components(fit.results["samples2d"][ml_index, :])
    ml_model_components = fitcopy.fitted_model.model_components
    ml_model_galaxy = pipes.model_galaxy(ml_model_components,filt_list=filt_list)
    ml_wavs = ml_model_galaxy.wavelengths*u.AA
    ml_model = ml_model_galaxy.spectrum_full*u.erg/u.s/u.AA/u.cm**2                                                                    
    
    #get range in models from fit
    wavs=(fitcopy.posterior.model_galaxy.wavelengths*u.AA)
    spec_post = np.percentile(fitcopy.posterior.samples["spectrum_full"],(16, 84), axis=0).T
    spec_med = (np.median(fitcopy.posterior.samples["spectrum_full"],axis=0).T.astype('float')\
                *u.erg/u.s/u.AA/u.cm**2)
    spec_post = (spec_post.astype(float)*u.erg/u.s/u.AA/u.cm**2)
    return {'wavs':ml_wavs,'max_like':ml_model}, {'wavs':wavs,'median':spec_med,'range':spec_post}


def get_z(fit):
    if not np.isscalar(fit.fitted_model.model_components["redshift"]):
        redshift = np.median(fit.posterior.samples["redshift"]) 
    else: redshift = fit.fitted_model.model_components["redshift"]
    return redshift


def measure_lir(wavs,spec,z,lo=8,hi=1000):
    """Measures 8--1000 um IR luminosity from posterior model"""
    #assumes rest-frame wavelengths
    DL = cosmo.luminosity_distance(z).to(u.Mpc)
    mask = (wavs>=lo*u.um) & (wavs<=hi*u.um)
    dlam = wavs[mask][1:]-wavs[mask][0:-1] 
    integ = np.sum(spec[mask][:-1]*dlam)
    lir = (4*np.pi*(1.+z) * DL**2 *integ).to(u.Lsun)
    return lir
    

def measure_luv(wavs,spec,z):
    """measures monochromatic luminosity at 2800 AA"""
    DL = cosmo.luminosity_distance(z).to(u.Mpc)
    mask = (wavs>=2450*u.AA) & (wavs<=3150*u.AA)
    flux = np.mean(spec[mask])
    luv = (4 * np.pi * DL**2 * (2800*u.AA) * (1.+z) * flux).to(u.Lsun)
    return luv


def get_obs_flux(wavs,spec,z,wl=1000):
    DL = cosmo.luminosity_distance(z).to(u.Mpc)
    mask = (wavs*(1.+z) >= 0.95*wl*u.um) & (wavs*(1.+z) <= 1.05*wl*u.um)
    dlam = wavs[mask][1:]-wavs[mask][0:-1] 
    flux = np.mean(spec[mask][:-1]).to(u.uJy,equivalencies=u.spectral_density(wav=wl*u.um))
    return flux


def read_filter(fname,path):
    filt = np.loadtxt(path+f'{fname}.dat')
    d_filt = {'wl':filt[:,0],'T':filt[:,1]}
    return d_filt


def measure_flux_in_filter(wavs,spec,z, filter):
    mask = ((wavs*(1.+z))>=filter['wl'].min()*u.AA) & ((wavs*(1.+z))<=filter['wl'].max()*u.AA)

    filter_interp = interp1d(filter['wl'],filter['T']) # interpolate filter values to spectrum wls
    dl = wavs[mask][1:]-wavs[mask][:-1]
    result = np.sum(wavs[mask][1:]*dl*spec[mask][1:]*filter_interp(wavs[mask][1:]*(1.+z)))
    norm = np.sum(wavs[mask][1:]*dl*filter_interp(wavs[mask][1:]*(1.+z)))
    flux = result/norm
    flux = (flux).to(u.Jy,equivalencies=u.spectral_density(wav=np.mean(wavs[mask]*(1.+z))))
    
    return flux


def plot_pipes_sed(ID, current_pipes, load_phot,filt_list,fit_instructions,ax=None,label='GS',zlab='z',num=0,
                  secondary=False, zerr=None):
    
    if ax is None: f, ax = plt.subplots(figsize=(5,2),dpi=180)
    else: f = ax.get_figure()
        
    galtest = pipes.galaxy(ID=ID,load_data=load_phot,spectrum_exists=False,filt_list=filt_list)
    fit = pipes.fit(galtest,run=current_pipes,fit_instructions=fit_instructions)
    fit.fit(verbose=False)
    fit.posterior.get_advanced_quantities()
    
    #observed phot
    wavs=(galtest.photometry[:,0]*u.AA)
    fluxes = (galtest.photometry[:,1]*u.erg/u.s/u.AA/u.cm**2).to(u.uJy,
                                                                    equivalencies=u.spectral_density(wav=wavs))
    errs = (galtest.photometry[:,2]*u.erg/u.s/u.AA/u.cm**2).to(u.uJy,
                                                                    equivalencies=u.spectral_density(wav=wavs))
    observed = (errs<90*u.mJy)
    ulims = (fluxes/errs<3)
    if secondary: ms = 4
    else: ms=3
    p=ax.errorbar(wavs.to(u.um)[observed&~ulims],fluxes[observed&~ulims],yerr=errs[observed&~ulims],
                marker='o',mec='xkcd:dark slate blue',mfc='w',ms=ms,mew=0.9,
                elinewidth=0.5,ecolor='k',capsize=1.7,
                zorder=20,ls='',fillstyle='full')
    
    # upper limits
    plot, caps, bars=ax.errorbar(wavs.to(u.um)[observed&ulims],(errs*3)[observed&ulims],
                                 yerr=(errs*0.9)[observed&ulims],
                mfc='xkcd:red orange',mec='xkcd:red orange',ms=ms-0.5,mew=0.5,
                elinewidth=0.5,ecolor='xkcd:red orange',uplims=True, fmt='_', capsize=1.3,zorder=20,ls='')
    
    #modeled phot
    if not np.isscalar(fit.fitted_model.model_components["redshift"]):
        redshift = np.median(fit.posterior.samples["redshift"]) #np.median(fit.posterior.samples["redshift"]) 
    else: redshift = fit.fitted_model.model_components["redshift"]
    
    wavs=(fit.galaxy.filter_set.eff_wavs*u.AA)
    fluxes = (np.median(fit.posterior.samples["photometry"],axis=0)*u.erg/u.s/u.AA/u.cm**2).to(u.uJy,
                                                                    equivalencies=u.spectral_density(wav=wavs))
                                                            
    
    p=ax.set(xscale='log',yscale='log',xlim=(0.3,5e3),ylim=(1e-3,1e5))
    ax.set_yticks([1e-2,1,1e2,1e4])
    ax.set_xlabel(r'$\lambda_{\rm obs}$ [$\mu$m]',fontsize=12,labelpad=-1)
    ax.set_ylabel(r'$f_\nu$ [$\mu$Jy]',fontsize=12,labelpad=-2)
    ax.tick_params(axis='both',labelsize=10)       
    
    #spectrum 
    wavs=(fit.posterior.model_galaxy.wavelengths*u.AA)*(1.+redshift)
    spec_post = np.percentile(fit.posterior.samples["spectrum_full"],
                                  (16, 84), axis=0).T
    spec_med = (np.median(fit.posterior.samples["spectrum_full"],axis=0).T.astype('float')\
                *u.erg/u.s/u.AA/u.cm**2).to(u.uJy,
                equivalencies=u.spectral_density(wav=wavs))
    spec_post[:,0] = (spec_post[:,0].astype(float)*u.erg/u.s/u.AA/u.cm**2).to(u.uJy,
                                                            equivalencies=u.spectral_density(wav=wavs))
    spec_post[:,1] = (spec_post[:,1].astype(float)*u.erg/u.s/u.AA/u.cm**2).to(u.uJy,
                                                            equivalencies=u.spectral_density(wav=wavs))
    c1='lightgrey'
    c2='xkcd:gray'
    ls='-'
    if secondary:
        c1='xkcd:light peach'
        c2='xkcd:pumpkin orange' 
        ls='--'
    p=ax.plot(wavs.to(u.um), spec_post[:, 0], color=c1,lw=0.3,
                zorder=1)
    p=ax.plot(wavs.to(u.um), spec_post[:, 1], color=c1,lw=0.3,
                zorder=1)
    p=ax.plot(wavs.to(u.um), spec_med, color=c2,ls=ls,
                zorder=2,lw=0.8,label='BAGPIPES')
    p=ax.fill_between(wavs.to(u.um).value, spec_post[:, 0], spec_post[:, 1],
                        zorder=1, color=c1, linewidth=0)

    if zerr is not None: # the order is plus then minus
        zerrstr = r'^{+'+f'{np.round(zerr[0],2):.2f}'+r'}_{-'+f'{np.round(zerr[1],2):.2f}'+r'}'
        zerrstr = f'${zerrstr}$'
        redshift = np.round(redshift,2)
    else: zerrstr = ''
    ax.annotate(f'{label}-{num}',(0.025,0.87),xycoords='axes fraction',fontsize=12)
    if not secondary: ax.annotate(f'${zlab}$'+r'$ = $'+f'{np.round(redshift,3)}'+zerrstr,(0.025,0.73),xycoords='axes fraction',fontsize=12)
    else: ax.annotate(f'${zlab}$'+r'$ = $'+f'{np.round(redshift,2)}'+zerrstr,(0.025,0.64), color=c2,
                      xycoords='axes fraction',fontsize=12)
    ax.tick_params(axis='both',which='both',direction='in')
    
    return f,ax