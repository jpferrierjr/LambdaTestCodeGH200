from ase.units import kB
from gpaw.new.ase_interface import GPAW # Using new one for GPU
from dftd4.ase import DFTD4
from gpaw import PW
import numpy as np
from ase import Atoms
import os


xc_list     = { 
    'PBE-D4':'PBE',
    #'r2SCAND4': 'MGGA_X_R2SCAN+MGGA_C_R2SCAN'
}

# Returns the requested calculator
def get_calculator( 
        xc:str                      = 'PBE',
        encut:float                 = 500.,
        initial_opt:bool            = False,
        kpointdensity:float|None    = 4.8 ):

    # NOTE: We are using the planar wave mode for consistency. Since the
    #       crystal structures tend to converge better with PW (in my experience)

    # Determine the parallelization parameters
    # PBE can run on the GPU while every other XC (currently)
    # can only run on the CPU. But we can still implement optimizations
    p_param = { 
        'sl_auto': True,
        'gpu':True,
        'use_elpa': False,
        'augment_grids': True
    }

    # if xc == 'r2SCAND4':
    #     p_param['gpu'] = False

    # Build out k-points
    k_pnts = {'density': kpointdensity, 'gamma': True}

    # Build Symmetry
    symm    = { 'point_group': False, 'time_reversal': False }
    if initial_opt:
        symm    = { 'point_group': True, 'time_reversal': True }
    
    # Set convergence (energy ~fmax/10)
    convergence = { 'energy': 0.001, 'density': 1e-4, 'eigenstates': 1e-6 }

    return  GPAW(   mode        = PW(encut),
                    basis       = 'dzp',
                    xc          = xc_list[xc],
                    kpts        = k_pnts,
                    occupations = { 'name': 'fermi-dirac', 'width': 0.05},
                    convergence = convergence,
                    symmetry    = symm,
                    parallel    = p_param )

# Checks to see if the normalized slope is below the given threshold
def check_normalized_slope( 
        X:list          = [], 
        Y:list          = [], 
        thresh:float    = 0.01, 
        flip:bool       = False, 
        minsteps:int    = 4 ):

    # Normalize X to 1
    x       = X.copy()
    max_x   = np.max( x )
    min_x   = np.min( x )
    mag_x   = max_x - min_x
    x       = (x - min_x)/mag_x
    if flip:
        x   = x[::-1]

    # Normalize Y to 1
    y       = Y.copy()
    max_y   = np.max( y )
    min_y   = np.min( y )
    mag_y   = max_y - min_y
    y       = ( y - min_y )/mag_y

    # Check slope of recent changes
    m       = np.abs( y[-1] - y[-2] )/np.abs( x[-1] - x[-2] )
    RET     = ( m <= thresh )

    # Also check averaging, as it could be fluctuating around a minima. Run on last 3 points
    if len( y ) > minsteps:
        y_slice = y[-3:]
        ave_y   = np.average( y_slice )
        std     = np.std( y_slice )

        # Check if the last point in y is between the deviations of the average value
        # If it is, compare the 'average' slope has been achieved
        ave_dv  = np.logical_and( y[-1] >= ( ave_y-std ), y[-1] <= ( ave_y+std )  )
        RET     = np.logical_or( RET, ave_dv )

    # If either is true, return True. Else, if both false, return False
    return RET

# Optimizes the Crystal Structure
def crystal_optimizer( 
        crystal:Atoms, 
        crystal_path:str   = "",
        is_bulk:bool        = False ):

    if not is_bulk:
        # Set the vacuum before starting. THis helps avoid issues when starting GPAW
        crystal.center( vacuum = 5., axis = 2 )


    # Return values in dict for XC
    return_vals = {}

    # Cycle through xcs
    for xc in xc_list:

        return_vals[xc] = {}

        xc_name         = xc
        if xc == "PBE-D4":
            xc_name = 'PBE'

        encut_file      = os.path.join( crystal_path, f"XC-{xc_name}_encut.npy" )
        encut_exists    = os.path.isfile( encut_file )

        # Optimize energy cutoff
        if encut_exists:
            return_vals[xc]['cutoff_energy'] = np.load( encut_file )
        else:
            # Do cutoff energy
            ENS         = np.linspace( 200., 800., 20 )
            pot_ens     = []
            val_list    = []

            for i, E in enumerate( ENS ):

                # Build calculator with vDW correction
                gcalc       = get_calculator( xc = xc, encut = E, initial_opt = True )
                calc        = DFTD4( method = 'pbe' ).add_calculator( gcalc )

                crystal.set_calculator( calc )
                    
                pot_ens.append( crystal.get_potential_energy() )
                val_list.append( E )

                if i > 0:
                    if check_normalized_slope( val_list, pot_ens, thresh = 0.05 ):
                        break

            # We shift 1 back, since there was no difference between the previous value and the current value.
            # This saves on computational time later.
            return_vals[xc]['cutoff_energy'] = val_list[-2]

            # Save the value
            np.save( encut_file, val_list[-2] )

        # Optimize vacuum
        if not is_bulk:
            vac_file        = os.path.join( crystal_path, f"XC-{xc_name}_Opt-vacuum.npy" )
            vac_exists      = os.path.isfile( vac_file )
            if vac_exists:
                return_vals[xc]['vacuum'] = np.load( vac_file )
            else:

                # Do vacuum
                VAC         = np.arange( 5., 20., 1. )
                pot_ens     = []
                val_list    = []

                # Cycle through the energies
                for i, V in enumerate( VAC ):
                    
                    # Build calculator with vDW correction
                    gcalc       = get_calculator( xc = xc, encut = return_vals[xc]['cutoff_energy'], initial_opt = True )
                    calc        = DFTD4( method = 'pbe' ).add_calculator( gcalc )
                    
                    crystal.set_calculator( calc )

                    crystal.center( vacuum = V, axis = 2 )

                    pot_ens.append( crystal.get_potential_energy() )
                    val_list.append( V )
                    
                    if i > 0:
                        if check_normalized_slope( val_list, pot_ens, thresh = 0.05 ):
                            break

                return_vals[xc]['vacuum'] = val_list[-2]

                # Save the value
                np.save( vac_file, val_list[-2] )

    return return_vals