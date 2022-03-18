import numpy as np
import astropy.units as u
import sys, os
import pandexo.engine.hst as hst



def optimize_hst(jmag, hmag, transit_duration):
    # Setup
    transit_duration = transit_duration.to(u.day).value
    schedule = '30' # Short program
    allnsamp = np.arange(1, 16)
    allsampseq = ['spars5', 'spars10', 'spars25']
    dispersers = ['G141', 'G102']
    scan_directions = ['Forward', 'Round Trip']
    subarrays = ['grism256', 'grism512']
    pandexo_input = {}
    pandexo_input['star'] = {'jmag' : jmag, 'hmag' : hmag}
    pandexo_input['planet'] = {'transit_duration' : transit_duration}
    pandexo_input['observation'] = {'noccultations' : 1}
    pandeia_input = {}
    best_err = 99999

    # Optimize over all instrument configurations
    sys._jupyter_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w') # Suppress prints
    for disp in dispersers:
        for direction in scan_directions:
            for sub_a in subarrays:
                for nsamp in allnsamp:
                    for samp_seq in allsampseq:
                        pandeia_input['strategy'] = {'schedulability' : schedule,
                                                    'scanDirection' : direction,
                                                    'nchan' : 1,
                                                    'norbits' :  None,
                                                    'useFirstOrbit' : False,
                                                    'targetFluence' : 30000}
                        pandeia_input['configuration'] = \
                            {'instrument' : {'disperser' : disp},
                             'detector' : {'subarray' : sub_a,
                                           'nsamp' : nsamp,
                                           'samp_seq' : samp_seq}}

                        sim = hst.wfc3_TExoNS({'pandeia_input':pandeia_input,
                                               'pandexo_input':pandexo_input})

                        err = sim['info']['Transit depth uncertainty(ppm)']
                        # print('%%%')
                        # print(err)
                        # print('%%%')
                        if err < best_err:
                            # print('WINNER ^')
                            # print()
                            best_err = err
                            best_pandeia = pandeia_input.copy()
                            
                            

    # Use the best instrument configuration to get a few other things
    sys.stdout = sys._jupyter_stdout # Reenable prints
    sim = hst.wfc3_TExoNS({'pandeia_input':best_pandeia,
                           'pandexo_input':pandexo_input})
    return sim, {'pandeia_input':best_pandeia,
                           'pandexo_input':pandexo_input}


class tweaked_pandexo:
    def __init__(self, refmag, refnoise, refexptime):
        self.refmag = refmag
        self.refnoise = refnoise
        self.refexptime = refexptime

    def wfc3_TExoNS(self, dictinput):
        '''Compute Transit depth uncertainty

        Compute the transit depth uncertainty for a defined system and number of spectrophotometric channels.
        Written by Kevin Stevenson      October 2016

        Parameters
        ----------
        dictinput : dict
            dictionary containing instrument parameters and exoplanet specific parameters. {"pandeia_input":dict1, "pandexo_input":dict1}

        Returns
        -------
        float
            deptherr--transit depth uncertainty per spectrophotometric channel
        float
            chanrms--light curve root mean squarerms
        float
            ptsOrbit--number of HST frames per orbit
        '''
        pandeia_input = dictinput['pandeia_input']
        pandexo_input = dictinput['pandexo_input']

        jmag = pandexo_input['star']['jmag']
        if np.ma.is_masked(jmag):
            print("Jmag not found.")
        try:
            hmag = pandexo_input['star']['hmag']
        except:
            hmag = jmag
            print("Hmag not found. Assuming no color dependence in the stellar type.")
        trdur = pandexo_input['planet']['transit_duration']
        numTr = pandexo_input['observation']['noccultations']

        schedulability = pandeia_input['strategy']['schedulability']
        scanDirection = pandeia_input['strategy']['scanDirection']
        nchan = pandeia_input['strategy']['nchan']
        norbits = pandeia_input['strategy']['norbits']
        useFirstOrbit = pandeia_input['strategy']['useFirstOrbit']
        try:
            targetFluence = pandeia_input['strategy']['targetFluence']
        except:
            targetFluence = 30000.
            print("Assuming a target fluence of 30,000 electrons.")
        disperser = pandeia_input['configuration']['instrument']['disperser'].lower(
        )
        subarray = pandeia_input['configuration']['detector']['subarray'].lower()
        nsamp = pandeia_input['configuration']['detector']['nsamp']
        samp_seq = pandeia_input['configuration']['detector']['samp_seq']

        try:
            samp_seq = samp_seq.lower()
        except:
            pass

        ##### HERE IS THE HACK #####
        if disperser == 'g141':
            # Define reference Jmag, flux, variance, and exposure time for GJ1214
            refmag = self.refmag
            refflux = 1
            refvar = (self.refnoise*1e-6)**2
            refexptime = self.refexptime
        elif disperser == 'g102':
            # Define reference Jmag, flux, variance, and exposure time for WASP12
            refmag = 10.477
            refflux = 8.26e7
            refvar = 9.75e7
            refexptime = 103.129
        else:
            print(("****HALTED: Unknown disperser: %s" % disperser))
            return

        # Determine max recommended scan height
        if subarray == 'grism512':
            maxScanHeight = 430
        elif subarray == 'grism256':
            maxScanHeight = 180
        else:
            print(("****HALTED: Unknown subarray aperture: %s" % subarray))
            return

        # Define maximum frame time
        maxExptime = 150.

        # Define available observing time per HST orbit in seconds
        if str(schedulability) == '30':
            obsTime = 51.3*60
        elif str(schedulability) == '100':
            obsTime = 46.3*60
        else:
            print(("****HALTED: Unknown schedulability: %s" % schedulability))
            return

        # Compute recommended number of HST orbits and compare to user specified value
        guessorbits = self.wfc3_GuessNOrbits(trdur)
        if norbits == None:
            norbits = guessorbits
        elif norbits != guessorbits:
            print(("****WARNING: Number of specified HST orbits does not match number of recommended orbits: %0.0f" % guessorbits))

        if nsamp == 0 or nsamp == None or samp_seq == None or samp_seq == "none":
            # Estimate reasonable values
            nsamp, samp_seq = wfc3_GuessParams(jmag, disperser, scanDirection, subarray, obsTime, maxScanHeight, maxExptime, targetFluence, hmag)

        # Calculate observation parameters
        exptime, tottime, scanRate, scanHeight, fluence = self.wfc3_obs(jmag, disperser, scanDirection, subarray,
                                                                   nsamp, samp_seq, targetFluence, hmag=hmag)
        if scanHeight > maxScanHeight:
            print(("****WARNING: Computed scan height exceeds maximum recommended height of %0.0f pixels." % maxScanHeight))
        if exptime > maxExptime:
            print(("****WARNING: Computed frame time (%0.0f seconds) exceeds maximum recommended duration of %0.0f seconds." % (exptime, maxExptime)))

        # Compute number of data points (frames) per orbit
        ptsOrbit = np.floor(obsTime/tottime)
        # First point (frame) is always low, ignore when computing duty cycle
        dutyCycle = (exptime*(ptsOrbit-1))/50./60*100

        # Compute number of non-destructive reads per orbit
        readsOrbit = ptsOrbit*(nsamp+1)

        # Look for mid-orbit buffer dumps
        if (subarray == 'grism256') and (readsOrbit >= 300) and (exptime <= 43):
            print(
                "****WARNING: Observing plan may incur mid-orbit buffer dumps.  Check with APT.")
        if (subarray == 'grism512') and (readsOrbit >= 120) and (exptime <= 100):
            print(
                "****WARNING: Observing plan may incur mid-orbit buffer dumps.  Check with APT.")

        # Compute number of HST orbits per transit
        # ~96 minutes per HST orbit
        orbitsTr = trdur*24.*60/96

        # Estimate number of good points during planet transit
        # First point in each HST orbit is flagged as bad; therefore, subtract from total

        if orbitsTr < 0.5:
            # Entire transit fits within one HST orbit
            ptsInTr = ptsOrbit * orbitsTr/0.5 - 1
        elif orbitsTr <= 1.5:
            # Assume one orbit centered on mid-transit time
            ptsInTr = ptsOrbit - 1
        elif orbitsTr < 2.:
            # Assume one orbit during full transit and one orbit during ingress/egress
            ptsInTr = ptsOrbit * \
                (np.floor(orbitsTr) +
                 np.min((1, np.remainder(orbitsTr-np.floor(orbitsTr)-0.5, 1)/0.5))) - 2
        else:
            # Assume transit contains 2+ orbits timed to maximize # of data points.
            ptsInTr = ptsOrbit * (np.floor(orbitsTr) + np.min(
                (1, np.remainder(orbitsTr-np.floor(orbitsTr), 1)/0.5))) - np.ceil(orbitsTr)

        # Estimate number of good points outside of transit
        # Discard first HST orbit
        ptsOutTr = (ptsOrbit-1) * (norbits-1) - ptsInTr

        # Compute transit depth uncertainty per spectrophotometric channel
        ratio = 10**((refmag - jmag)/2.5)
        flux = ratio*refflux*exptime/refexptime
        fluxvar = ratio*refvar*exptime/refexptime
        chanflux = flux/nchan
        chanvar = fluxvar/nchan
        chanrms = np.sqrt(chanvar)/chanflux*1e6  # ppm
        inTrrms = chanrms/np.sqrt(ptsInTr*numTr)  # ppm
        outTrrms = chanrms/np.sqrt(ptsOutTr*numTr)  # ppm
        deptherr = np.sqrt(inTrrms**2 + outTrrms**2)  # ppm

        info = {"Number of HST orbits": norbits,
                "Use first orbit":   useFirstOrbit,
                "WFC3 parameters: NSAMP": nsamp,
                "WFC3 parameters: SAMP_SEQ": samp_seq.upper(),
                "Scan Direction": scanDirection,
                "Recommended scan rate (arcsec/s)": scanRate,
                "Scan height (pixels)": scanHeight,
                "Maximum pixel fluence (electrons)": fluence,
                "exposure time": exptime,
                "Estimated duty cycle (outside of Earth occultation)": dutyCycle,
                "Transit depth uncertainty(ppm)": deptherr,
                "Number of channels": nchan,
                "Number of Transits": numTr}

        return {"spec_error": deptherr/1e6,
                "light_curve_rms": chanrms/1e6,
                "nframes_per_orb": ptsOrbit,
                "info": info}

    def wfc3_GuessNOrbits(self, trdur):
        '''Predict number of HST orbits

        Predict number of HST orbits for transit observation when not provided by the user.

        Parameters
        ----------
        trdur : float
            transit duration in days

        Returns
        -------
        float
            number of requested orbits per transit (including discarded thermal-settling orbit)
        '''
        # Compute # of HST orbits during transit
        # ~96 minutes per HST orbit
        orbitsTr = trdur*24.*60/96
        if orbitsTr <= 1.5:
            norbits = 4.
        elif orbitsTr <= 2.0:
            norbits = 5.
        else:
            norbits = np.ceil(orbitsTr*2+1)

        return norbits

    def wfc3_obs(self, jmag, disperser, scanDirection, subarray, nsamp, samp_seq, targetFluence=30000., hmag=None):
        '''Determine the recommended exposure time, scan rate, scan height, and overheads.

        Parameters
        ----------
        jmag : float
            J-band magnitude
        disperser : str
            Grism ('G141' or 'G102')
        scanDirection : str
            spatial scan direction ('Forward' or 'Round Trip')
        subarray : str
            Subarray aperture ('grism256' or 'grism512')
        nsamp : float
            Number of up-the-ramp samples (1..15)
        samp_seq : str
            Time between non-destructive reads ('SPARS5', 'SPARS10', or 'SPARS25')
        targetFluence : float
            (Optional) Desired fluence in electrons per pixel
        hmag : float
            (Optional) H-band magnitude

        Returns
        -------
        float
            exptime--exposure time in seconds
        float
            tottime--total frame time including overheads in seconds
        float
            scanRate--recommented scan rate in arcsec/s
        float
            scanHeight--scan height in pixels
        float
            fluence--maximum pixel fluence in electrons
        '''
        # Estimate exposure time
        if subarray == 'grism512':
            # GRISM512
            if samp_seq == 'spars5':
                exptime = 0.853 + (nsamp-1)*2.9215  # SPARS5
            elif samp_seq == 'spars10':
                exptime = 0.853 + (nsamp-1)*7.9217  # SPARS10
            elif samp_seq == 'spars25':
                exptime = 0.853 + (nsamp-1)*22.9213  # SPARS25
            else:
                print(("****HALTED: Unknown SAMP_SEQ: %s" % samp_seq))
                return
        else:
            # GRISM256
            if samp_seq == 'spars5':
                exptime = 0.280 + (nsamp-1)*2.349  # SPARS5
            elif samp_seq == 'spars10':
                exptime = 0.278 + (nsamp-1)*7.3465  # SPARS10
            elif samp_seq == 'spars25':
                exptime = 0.278 + (nsamp-1)*22.346  # SPARS25
            else:
                print(("****HALTED: Unknown SAMP_SEQ: %s" % samp_seq))
                return

        # Recommended scan rate
        if hmag == None:
            hmag = jmag
        #scanRate = np.round(1.9*10**(-0.4*(hmag-5.9)), 3)  # arcsec/s
        #scanRate = np.round(2363./targetFluence*10**(-0.4*(jmag-9.75)), 3) # arcsec/s
        scanRate = (2491./targetFluence)*10**(-0.4*(jmag-9.75)) - (161/targetFluence)*10**(-0.4*(jmag-hmag))
        if disperser == 'g102':
            # G102/G141 flux ratio is ~0.8
            scanRate *= 0.8
        # Max fluence in electrons/pixel
        #fluence = (5.5/scanRate)*10**(-0.4*(hmag-15))*2.4  # electrons
        #fluence = (2363./scanRate)*10**(-0.4*(jmag-9.75))  # electrons
        fluence = (2491./scanRate)*10**(-0.4*(jmag-9.75)) - (161/scanRate)*10**(-0.4*(jmag-hmag))
        if disperser == 'g102':
            # WFC3_ISR_2012-08 states that the G102/G141 scale factor is 0.96 DN/electron
            fluence *= 0.96
            # G102/G141 flux ratio is ~0.8
            fluence *= 0.8
        # Scan height in pixels
        scanRatep = scanRate/0.121  # pixels/s
        scanHeight = scanRatep*exptime  # pixels
        '''
        #Quadratic correlation between scanRate and read overhead
        foo     = np.array([0.0,0.1,0.3,0.5,1.0,2.0,3.0,4.0])/0.121
        bar     = np.array([ 40, 40, 41, 42, 45, 50, 56, 61])
        c       = np.polyfit(foo,bar,2)
        model   = c[2] + c[1]*foo + c[0]*foo**2
        #c = [  6.12243227e-04,   6.31621064e-01,   3.96040946e+01]
        '''
        # Define instrument overheads (in seconds)
        c = [6.12243227e-04, 6.31621064e-01, 3.96040946e+01]
        read = c[2] + c[1]*scanRatep + c[0]*scanRatep**2
        # Correlation between scanHeight/scanRate and pointing overhead was determined elsewhere
        if scanDirection == 'Round Trip':
            # Round Trip scan direction doesn't have to return to starting point, therefore no overhead
            pointing = 0.
        elif scanDirection == 'Forward':
            c = [3.18485340e+01,   3.32968829e-02,   1.65687590e-02,
                 7.65510038e-01,  -6.24504499e+01,   5.51452028e-03]
            pointing = c[0]*(1 - np.exp(-c[2]*(scanHeight-c[4]))) + \
                c[1]*scanHeight + c[3]*scanRatep + c[5]*scanRatep**2
        else:
            print(("****HALTED: Unknown scan direction: %s" % scanDirection))
            return
        # Estimate total frame time including overheads
        tottime = exptime+read+pointing  # seconds

        return exptime, tottime, scanRate, scanHeight, fluence