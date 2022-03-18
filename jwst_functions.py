import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from joblib import Parallel, delayed
import astropy.units as u
from astropy.modeling import blackbody as bb
import sys, os
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    import pysynphot as psyn

import pandexo.engine.justdoit as jdi

from pandeia.engine.instrument_factory import InstrumentFactory
from pandeia.engine.perform_calculation import perform_calculation




class homebrew_pandexo:
    def __init__(self, mag, mag_band, temp, metal, logg,
                transit_duration=None, out_transit_duration=None, Rp2Rs=None):
        self.mag, self.mag_band, self.temp, self.metal, self.logg = \
            mag, mag_band, temp, metal, logg
        self.transit_duration, self.out_duration, self.Rp2Rs = \
            transit_duration, out_transit_duration, Rp2Rs
        
        self.pandeia_base = {
        'telescope' : 'jwst',
        'scene': [{'id': 1,
                'shape' : {'geometry' : 'point'} ,
                'position' : {'ang_unit': 'arcsec',
                             'x_offset' : 0.0,
                             'y_offset' : 0.0,
                             'orientation' : 0},
               'spectrum' : {'normalization': {'type': 'none'},
                             'sed': {'sed_type' : 'input',
                                     'spectrum' : []},
                             'lines' : [],
                             'redshift' : 0,
                             'extinction' : {'value' : 0.0,
                                             'law' : 'mw_rv_31',
                                             'unit' : 'mag',
                                             'bandpass' : 'j'}
                            }}],
        'background_level' : 'medium',
        'background' : 'ecliptic',
        'calculation' : {'effects' : {'background': True,
                      'ipc': True,
                      'saturation': True},
        'noise': {'crs': True,
               'darkcurrent': True,
               'ffnoise': True,
               'readnoise': True,
               'rn_correlation': True}},
        'configuration': {'instrument': {'instrument': 'nirspec',
                                      'mode': 'fixed_slit',
                                      'filter': 'clear',
                                      'aperture': 's1600a1',
                                      'disperser': 'prism'},
                        'detector': {'readout_pattern': 'nrsrapid',
                                     'subarray': 'sub512',
                                     'readmode': 'nrsrapid',
                                     'ngroup': None,
                                     'nint': 1,
                                     'nexp': 1}},
        'strategy': {'method': 'specapphot',
                  'background_subtraction': True,
                  'aperture_size': None,
                  'sky_annulus': [0.75, 1.5],
                  'target_xy': [0.0, 0.0],
                  'reference_wavelength': 2.5,
                  'units': 'arcsec'}}
        
    def get_stellar_spec(self):
        assert self.mag_band in ['J', 'H', 'K']
        if self.metal > 0.5: self.metal = 0.5
        sp = psyn.Icat("phoenix", self.temp, self.metal, self.logg)

        refdata = os.environ.get("pandeia_refdata")
        all_bps = {"H": 'bessell_h_004_syn.fits',
                     "J":'bessell_j_003_syn.fits' ,
                     "K": 'bessell_k_003_syn.fits'}
        bp_path = os.path.join(refdata, "normalization",
                               "bandpass", all_bps[self.mag_band])
        bp = psyn.FileBandpass(bp_path)

        sp.convert('angstroms')
        bp.convert('angstroms')
        rn_sp = sp.renorm(self.mag, 'vegamag', bp)
        rn_sp.convert("microns")
        rn_sp.convert("mjy")
        return {'wave':rn_sp.wave, 'flux':rn_sp.flux}
    
    
    def optimize_groups_per_int(self, star_spectrum):
        # Run it once with the minimum number of groups 
        pandeia_base = self.pandeia_base.copy()
        
        pandeia_base['scene'][0]['spectrum']['sed']['spectrum'] = \
                np.array([star_spectrum['wave'], star_spectrum['flux']])
        pandeia_base['strategy']['aperture_size'] = 0.15
        
        pandeia_base['configuration']['detector']['ngroup'] = 2
        pandeia_base['configuration']['detector']['nint'] = 1
        pandeia_base['configuration']['detector']['nexp'] = 1
        report_out = perform_calculation(pandeia_base, dict_report=False)


        # The subarrays have a full depth of 77,000 e's
        # The ETC limits you to 65,000, so I'm just going to take that
        # Following PandExo heavily here
        tframe = report_out.exposure_specification.tframe
        max_flux = np.max(report_out.signal.rate_plus_bg_list[0]['fp_pix'])
        max_time_per_int = 65000 / max_flux
        max_groups = np.floor(max_time_per_int / tframe)

        return max_groups
    
    def optimize_extraction(self, groups_per_int, star_spectrum):
        pandeia_base = self.pandeia_base.copy()
        
        pandeia_base['configuration']['detector']['ngroup'] = groups_per_int
        pandeia_base['scene'][0]['spectrum']['sed']['spectrum'] = \
                np.array([star_spectrum['wave'], star_spectrum['flux']])
        
        def opt(height):
            pandeia_base['strategy']['aperture_size'] = height
            r = perform_calculation(pandeia_base, dict_report=True)
            noise = np.sqrt(np.sum(r['1d']['extracted_noise'][1]**2))
            signal = np.sum(r['1d']['extracted_flux'][1])
            return r, signal/noise
        Heights = np.linspace(0.05, 0.5, 10)
        q = Parallel(n_jobs=5)(delayed(opt)(h) for h in Heights)
        
        best_ind = np.argmax(np.array(q)[:,1])
        report = q[best_ind][0]
        
        noise = np.sqrt(np.sum(report['1d']['extracted_noise'][1]**2))
        signal = np.sum(report['1d']['extracted_flux'][1])
        
        return report
    
    def manual_extraction(self, aperature_height, groups_per_int,
                          star_spectrum):
        pandeia_base = self.pandeia_base.copy()
        
        pandeia_base['strategy']['aperture_size'] = aperature_height
        pandeia_base['configuration']['detector']['ngroup'] = groups_per_int
        pandeia_base['scene'][0]['spectrum']['sed']['spectrum'] = \
                np.array([star_spectrum['wave'], star_spectrum['flux']])
        
        r = perform_calculation(pandeia_base, dict_report=True)
        return r
    
    def summarize_pandeia_report(self, report):
        noise = np.sqrt(np.sum(r['1d']['extracted_noise'][1]**2))
        flux = np.sum(r['1d']['extracted_flux'][1])
        return {'Time per integration' : 
                  (report['information']['exposure_specification']
                         ['total_exposure_time'])*u.second,
               'Integration error (ppm)' : noise/flux*1e6}
    
    
    def transit_summary(self):
        # Get the out of transit integration, optimizing over number of groups
        # and extraction aperature
        out_spec = self.get_stellar_spec()
        num_groups = self.optimize_groups_per_int(star_spectrum=out_spec)
        out_report = self.optimize_extraction(groups_per_int=num_groups,
                                              star_spectrum=out_spec)
        
        # Extract some stuff from that report to create the in transit report
        ap = out_report['input']['strategy']['aperture_size']
        time_per_integration = out_report['information']['exposure_specification']['total_exposure_time']*u.second
        
        # Almost definitely overkill, but I just want to run Pandeia
        # one more time using the true in transit flux and same parameters
        # derived for out of transit
        in_spec = out_spec.copy()
        in_spec['flux'] = out_spec['flux']*(1-self.Rp2Rs**2)
        in_report = self.manual_extraction(aperature_height=ap,
                                           groups_per_int=num_groups,
                                           star_spectrum=in_spec)
        
        points_out_transit = np.floor((self.out_duration / 
                            time_per_integration).to(u.dimensionless_unscaled)).value
        points_in_transit = np.floor((self.transit_duration / 
                            time_per_integration).to(u.dimensionless_unscaled)).value
        
        
        out_noise = np.sqrt(np.sum(out_report['1d']['extracted_noise'][1]**2))
        out_flux = np.sum(out_report['1d']['extracted_flux'][1])
        in_noise = np.sqrt(np.sum(in_report['1d']['extracted_noise'][1]**2))
        in_flux = np.sum(in_report['1d']['extracted_flux'][1])
        
        total_out_noise = out_noise / np.sqrt(points_out_transit)
        total_in_noise = in_noise / np.sqrt(points_in_transit)
        
        
        total_depth_err = np.sqrt( (in_flux/out_flux)**2 \
                                      * (total_in_noise**2/out_flux**2 \
                                         + total_out_noise**2/in_flux**2))
        
        return {
               'total depth error (ppm)' : total_depth_err*1e6,
               'groups per integration' : num_groups,
               'time per integration' : time_per_integration,
               'out transit total err (ppm)' : total_out_noise/out_flux * 1e6,
               'in transit total err (ppm)' : total_in_noise/in_flux * 1e6,
               'out transit indiv err (ppm)' : out_noise/out_flux * 1e6,
               'in transit indiv err (ppm)' : in_noise/in_flux * 1e6,
               'num integrations out of transit' : points_out_transit,
               'num integrations in transit' : points_in_transit}
    

    
def actual_pandexo(hmag=14.168,
                   temp=6157, metal=0, logg=4.37, 
                   transit_duration=19.13*u.hour,
                   Rp=0.8886*u.R_jup, Rs=1.117*u.R_sun):
    
    exo_dict = jdi.load_exo_dict()

    # Things which vary
    exo_dict['planet']['transit_duration'] = (transit_duration.to(u.s)).value
    exo_dict['star']['mag'] = hmag
    exo_dict['star']['ref_wave'] = 1.6 # H mag
    exo_dict['star']['temp'] = 	temp
    exo_dict['star']['metal'] = metal
    exo_dict['star']['logg'] = logg

    # Things which should stay the same each time
    exo_dict['observation']['sat_level'] = 100
    exo_dict['observation']['sat_unit'] = '%'
    exo_dict['observation']['noccultations'] = 1
    exo_dict['observation']['R'] = None
    exo_dict['observation']['baseline_unit'] = 'frac'
    exo_dict['observation']['baseline'] = 1 # Total observing time = 2 transit durations
    exo_dict['planet']['type'] = 'constant'
    exo_dict['planet']['td_unit'] = 's'
    exo_dict['planet']['radius'] = Rp.value
    exo_dict['planet']['r_unit'] = 'R_jup'
    exo_dict['planet']['f_unit'] = 'rp^2/r*^2'
    exo_dict['star']['type'] = 'phoenix'
    exo_dict['star']['radius'] = Rs.value
    exo_dict['star']['r_unit'] = 'R_sun'
    
    p = jdi.run_pandexo(exo_dict, ['NIRSpec Prism'], save_file = False)
    
    
    out_flux = np.sum(p['RawData']['electrons_out']) / p['timing']['Num Integrations Out of Transit']
    out_var = np.sum(p['RawData']['var_out'])  / p['timing']['Num Integrations Out of Transit']
    in_flux = np.sum(p['RawData']['electrons_in'])  / p['timing']['Num Integrations In Transit']
    in_var = np.sum(p['RawData']['var_in'])  / p['timing']['Num Integrations In Transit']

    total_out_noise = np.sqrt(out_var) / np.sqrt(p['timing']['Num Integrations Out of Transit'])
    total_in_noise = np.sqrt(in_var) / np.sqrt(p['timing']['Num Integrations In Transit'])

    total_depth_err = np.sqrt( (in_flux/out_flux)**2 \
                  * (total_in_noise**2/out_flux**2 \
                     + total_out_noise**2/in_flux**2))
    
    
    return p, {
       'total depth error (ppm)' : total_depth_err*1e6,
       'groups per integration' : p['timing']['APT: Num Groups per Integration'],
       'time per integration' : p['timing']['Time/Integration incl reset (sec)'],
       'out transit total err (ppm)' : total_out_noise/out_flux * 1e6,
       'in transit total err (ppm)' : total_in_noise/in_flux * 1e6,
       'out transit indiv err (ppm)' : np.sqrt(out_var)/out_flux * 1e6,
       'in transit indiv err (ppm)' : np.sqrt(in_var)/in_flux * 1e6,
       'num integrations out of transit' : p['timing']['Num Integrations Out of Transit'],
       'num integrations in transit' : p['timing']['Num Integrations In Transit']}


