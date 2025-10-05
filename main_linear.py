import os
import json
import numpy as np
from gpaw import PW
from ase import Atoms
from ase.units import kB
from dftd4.ase import DFTD4
from gpaw.mixer import Mixer
from ase.optimize import FIRE               # The algorithm for minimizing the forces
from ase.spacegroup import crystal
from gpaw.new.ase_interface import GPAW
from ase.filters import FrechetCellFilter
from ase.io import Trajectory, read, write
from ase.build import surface, mx2, graphene
from ase.constraints import FixAtoms, FixSymmetry


#### Set Initial variables
#region - Initial Variables
os.environ['GPAW_NEW']      = 'True'
os.environ['GPAW_USE_GPUS'] = 'True'

bulk_substrates    = None
materials_2d       = None
fmax               = 0.01
kpt_dens           = 4.8

bulk_calc_vals     = {}
qm_calc_vals       = {}
dE_sub2D           = {}
sub_Es             = {}
qmat_Es            = {}
het_Es             = {}

save_path          = os.path.join( os.getcwd(), "DFT Output DFTD4" )
bulk_path          = os.path.join( save_path, "bulk" )
qm_path            = os.path.join( save_path, "2D materials" )
het_path           = os.path.join( save_path, "heterostructures" )
sub_path           = os.path.join( save_path, "substrates" )
os.makedirs( save_path, exist_ok = True )
os.makedirs( bulk_path, exist_ok = True )
os.makedirs( qm_path, exist_ok = True )
os.makedirs( het_path, exist_ok = True )
os.makedirs( sub_path, exist_ok = True )

# XCs
xc_list     = { 
    'PBE-D4':'PBE',
    'r2SCAND4': 'MGGA_X_R2SCAN+MGGA_C_R2SCAN'
}

# Material specifics
bulk_substrates_thicknesses = {
    # 'Cu_111': 5,
    # 'Ni_111': 5,
    # 'Sapphire_Al2O3_0001': 2,
    # 'SiO2_alpha_quartz': 2,
    'SiO2_beta_cristobalite': 2
}

# Orienations of all structures when they're sliced
orientations  = {
    # 'Cu_111': (1,1,1),
    # 'Ni_111': (1,1,1),
    # 'Sapphire_Al2O3_0001': (0,0,1),
    # 'SiO2_alpha_quartz': (0,0,1),
    'SiO2_beta_cristobalite': (0,0,1)
}

# Devise their combinations
combinations = [
    # ('Cu_111', 'graphene'),
    # ('Ni_111', 'graphene'),
    # ('Sapphire_Al2O3_0001', 'hbn'),
    # ('Cu_111', 'hbn'),
    # ('SiO2_alpha_quartz', 'mos2'),
    ('SiO2_beta_cristobalite', 'mos2')#,
    # ('SiO2_alpha_quartz', 'mose2'),
    # ('SiO2_beta_cristobalite', 'mose2'),
    # ('SiO2_alpha_quartz', 'ws2'),
    # ('SiO2_beta_cristobalite', 'ws2'),
    # ('SiO2_alpha_quartz', 'wse2'),
    # ('SiO2_beta_cristobalite', 'wse2')
]

#### Due to a lot of frustration, the stacking information was gathered manually, as my algorithm for it didn't work properly
# First 3 is qm, Second 3 is substrate
stack_repeat = {
    'graphene':{
        'Cu_111': [2,1],
        'Ni_111': [2,1]
    },
    'hbn':{
        'Sapphire_Al2O3_0001': [2,1],
        'Cu_111': [2,1]
    },
    'mos2':{
        'SiO2_alpha_quartz': [3,2],
        'SiO2_beta_cristobalite': [3,2]
    },
    'mose2':{
        'SiO2_alpha_quartz': [3,2],
        'SiO2_beta_cristobalite': [3,2]
    },
    'ws2':{
        'SiO2_alpha_quartz': [3,2],
        'SiO2_beta_cristobalite': [3,2]
    },
    'wse2':{
        'SiO2_alpha_quartz': [3,2],
        'SiO2_beta_cristobalite': [3,2]
    }
}

# x,y shift for stacked 2D material
stack_shifts = {
    'graphene':{
        'Cu_111': [ 5.063, 1.462 ],
        'Ni_111': [ 4.921, 1.462 ]
    },
    'hbn':{
        'Sapphire_Al2O3_0001': [ 2.510, 0 ],
        'Cu_111': [ 3.786, -0.724 ]
    },
    'mos2':{
        'SiO2_alpha_quartz': [ -0.415, 0.589 ],
        'SiO2_beta_cristobalite': [ -0.415, 0.589 ]     # This one is not a great fit and will have strain
    },
    'mose2':{
        'SiO2_alpha_quartz': [ -0.415, 0.589 ],
        'SiO2_beta_cristobalite': [ -0.415, 0.589 ]     # This one is not a great fit and will have strain
    },
    'ws2':{
        'SiO2_alpha_quartz': [ -0.415, 0.589 ],
        'SiO2_beta_cristobalite': [ -0.415, 0.589 ]     # This one is not a great fit and will have strain
    },
    'wse2':{
        'SiO2_alpha_quartz': [ -0.415, 0.589 ],
        'SiO2_beta_cristobalite': [ -0.415, 0.589 ]     # This one is not a great fit and will have strain
    }
}
#endregion

#### Functions
#region functions
## General
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

## Bulks
def create_sio2_alpha_quartz():
    """Create a SiO2 alpha quartz substrate."""

    # Values taken from Materials Project mp-6930

    a = b = 4.92
    c = 5.43

    bulk = crystal(
        ('Si', 'O'), # Formula unit in the cell
        basis=[(0., 0.468798, 1./3.), (0.144538, 0.73144, 0.54782)],
        spacegroup=154,
        cellpar=[a, b, c, 90, 90, 120])

    return bulk

def create_sio2_beta_cristobalite():
    """Create a SiO2 beta cristobalite substrate."""
    
    # Values taken from Materials Project mp-6922
    # Transition to beta happens around 575C. So, this is the most common to use

    a = b = 5.06 
    c = 5.54

    bulk = crystal(
        ('Si', 'O'), # Formula unit in the cell
        basis=[(0., 1./2., 2./3.), (0.791372, 0.582743, 1./2.)],
        spacegroup=180,
        cellpar=[a, b, c, 90, 90, 120])

    return bulk

def create_copper_111():
    """Create a Cu(111) substrate."""
    # Create as bulk first
    # mp-30
    a = b = c = 3.58 

    bulk = crystal(
        ('Cu'), # Formula unit in the cell
        basis = [(0., 0., 0.)],
        spacegroup = 225,
        cellpar = [a, b, c, 90, 90, 90])

    return bulk

def create_nickel_111(size=(3, 3, 1)):
    """Create a Ni(111) substrate."""
    # mp-23
    a = b = c = 3.48 

    bulk = crystal(
        ('Ni'), # Formula unit in the cell
        basis = [(0., 0., 0.)],
        spacegroup = 225,
        cellpar = [a, b, c, 90, 90, 90] )

    return bulk

def create_sapphire_0001(size=(3, 3, 1)):
    """Create a Sapphire (Al2O3) (0001) substrate."""
    # Values taken from Materials Project mp-1143

    a = b = 4.81 
    c = 13.12

    bulk = crystal(
        ('Al', 'O'), # Formula unit in the cell
        basis=[(1./3, 2./3., 0.814571), (0.360521, 1./3., 0.583333)],
        spacegroup=167,
        cellpar=[a, b, c, 90, 90, 120]
    )

    return bulk


## QMs
def create_graphene( vacuum = 5.0 ):
    """Create a graphene sheet."""
    return graphene( vacuum = vacuum )

def create_hbn( vacuum = 5.0 ):
    """Create a hexagonal boron nitride (hBN) sheet."""
    return graphene( formula = "BN", a = 2.51, vacuum = vacuum )

def create_mos2( vacuum = 5.0 ):
    """Create a MoS2 sheet."""
    return mx2( vacuum = vacuum )
    
def create_mose2( vacuum = 5.0 ):
    """Create a MoSe2 sheet."""
    return mx2( formula = 'MoSe2', a = 3.29, thickness = 3.34, vacuum = vacuum )

def create_ws2( vacuum = 5.0 ):
    """Create a WS2 sheet."""
    return mx2( formula = 'WS2', a = 3.15, thickness = 3.14, vacuum = vacuum )
    
def create_wse2( vacuum = 5.0 ):
    """Create a WSe2 sheet."""
    return mx2( formula = 'WSe2', a = 3.28, thickness = 3.31, vacuum = vacuum )


#endregion

#### Build bulk materials
#region Build bulk
bulk_substrates = {
    # 'SiO2_alpha_quartz': create_sio2_alpha_quartz(),
    'SiO2_beta_cristobalite': create_sio2_beta_cristobalite()#,
    # 'Cu_111': create_copper_111(),
    # 'Ni_111': create_nickel_111(),
    # 'Sapphire_Al2O3_0001': create_sapphire_0001()
}
#endregion

#### Build 2D materials
#region Build 2D
materials_2d = {
    # 'graphene': create_graphene(),
    # 'hbn': create_hbn(),
    'mos2': create_mos2()#,
    # 'mose2': create_mose2(),
    # 'ws2': create_ws2(),
    # 'wse2': create_wse2()
}
#endregion

#### Optimize the bulks GPAW
#region Optimize bulks GPAW
for bulk in bulk_substrates:
    # Create the directories needed, if they don't exist.
    bulk_path = os.path.join( bulk_path, f"{bulk}" )
    os.makedirs( bulk_path, exist_ok = True )

    print( f"Optimizing crystal {bulk}:" )

    bulk_calc_vals[bulk] = crystal_optimizer( crystal = bulk_substrates[bulk], crystal_path = bulk_path, is_bulk = True )
#endregion

#### Optimize the bulk structures
#region Optimize bulk structures
for bulk in bulk_substrates:

    bulk_path   = os.path.join( bulk_path, f"{bulk}" )
    traj_path   = os.path.join( bulk_path, "trajectories" )
    os.makedirs( traj_path, exist_ok = True )

    # After this is done, we will continue with the rest of the XCs
    for xc in xc_list:

        pot_file        = os.path.join( bulk_path, f'{xc}_relaxed.json' )
        xc_traj_file    = os.path.join( traj_path, f'{xc}_relaxation.traj' )

        # If the gpaw file exists, this has been done already.
        if not os.path.isfile(pot_file):
            
            # Set the crystal 
            crystal     = bulk_substrates[bulk].copy()

            # If a relaxation has been started, just copy the last frame of the relaxation
            # file and use that for the copy.
            if os.path.isfile( xc_traj_file ):
                crystal:Atoms = read( xc_traj_file, index = "-1" )
                
            gpw_calc    = get_calculator( xc = xc, encut = bulk_calc_vals[bulk][xc]['cutoff_energy'] )
            calc        = DFTD4( method = 'PBE' ).add_calculator( gpw_calc )

            crystal.set_calculator( calc )

            # Fix the symmetry
            crystal.set_constraint( FixSymmetry( crystal ) )

            # Set the ExpCellFilter to also relax the cell parameters
            ecf         = FrechetCellFilter( crystal )

            #optimizer   = BFGSLineSearch( ecf, trajectory = xc_traj_file, append_trajectory = True )
            optimizer   = FIRE( ecf, trajectory = xc_traj_file, append_trajectory = True )
            optimizer.run( fmax = fmax )

            # Get the new cell params
            cell_params = crystal.cell.cellpar()
            # Create output file with cell parameters and potential energy
            output_dict = {
                'poten': crystal.get_potential_energy(),
                'cell_params': cell_params.tolist()
            }

            # Save the final potential energy just so that we can reference it later
            with open( pot_file, 'w' ) as f:
                json.dump( output_dict, f, indent = 4 )
#endregion

#### Optimize the QM materials GPAW
#region Optimize qm gpaw
for qmmat in materials_2d:

    # Create the directories needed, if they don't exist.
    sv_path = os.path.join( qm_path, f"{qmmat}" )
    os.makedirs( sv_path, exist_ok = True )

    print( f"Optimizing 2D material {qmmat}:" )

    qm_calc_vals[qmmat] = crystal_optimizer( crystal = materials_2d[qmmat], crystal_path = sv_path )
#endregion

#### Calculate the binding energies
#region Calculate binding energies
for bulk, qmat in combinations:

    het_name            = f"{qmat}_on_{bulk}"

    blk_path            = os.path.join( bulk_path, f"{bulk}" )
    sb_path             = os.path.join( sub_path, f"{bulk}" )
    crsy_path           = os.path.join( qm_path, f"{qmat}" )
    hetero_path         = os.path.join( het_path, het_name )
    traj_path           = os.path.join( blk_path, "trajectories" )
    os.makedirs( sb_path, exist_ok = True )
    os.makedirs( crsy_path, exist_ok = True )
    os.makedirs( hetero_path, exist_ok = True )

    # Read the cell parameters of blk as compared to the original size.
    blk_cpy             = surface( bulk_substrates[bulk].copy(), orientations[bulk], bulk_substrates_thicknesses[bulk], vacuum = 5. )
    init_cell           = blk_cpy.cell.cellpar()

    # Cycle through XCs
    for xc in xc_list:

        het_traj_path   = os.path.join( hetero_path, "trajectories" )
        het_traj_file   = os.path.join( het_traj_path, f"{xc}_relaxation.traj" )
        het_bind_en     = os.path.join( hetero_path, f"{xc}_binding_energy.npy" )
        qmat_traj       = os.path.join( crsy_path, f"{xc}_relaxed_on_{bulk}.traj" )
        os.makedirs( het_traj_path, exist_ok = True )

        # Check if sub, qm, or hetero calculations already exist
        if os.path.isfile( het_bind_en ):

            dE_sub2D[het_name][xc] = np.load( het_bind_en )
        
        else:

            xc_traj_file    = os.path.join( traj_path, f'{xc}_relaxation.traj' )

            # Read the respective trajectory file's last indice
            blk:Atoms             = read( xc_traj_file, index = "-1" )

            # Remove FixSymmetry contraints
            del blk.constraints
            
            # Slide the substrate
            sub             = surface( blk, orientations[bulk], bulk_substrates_thicknesses[bulk], vacuum = 5. )

            # Calculate the cell parameter change
            sub_cell        = sub.cell.cellpar()
            cell_ratio      = sub_cell[0]/init_cell[0]  # Only the a parameter is used because we had a fixed symmetry. So, scaling will be uniform

            # Copy the 2D material
            mat2D           = materials_2d[qmat].copy()

            # Scale the 2D material values similarly to reduce computation and strain errors when stacking
            # WHY:  Since the vdW XCs rely on long range atomic atraction, it is very likely that our 2D materials
            #       will self-attract. To reduce the relaxation time, we can "pre-relax" them a bit, using the 
            #       bulk material as a template.

            mat2D.cell      *= cell_ratio
            mat2D.positions *= cell_ratio

            # Get the stacking values.
            # n_sub_x, n_mat_x, het = place_2d_on_substrate( sub, mat2D, vacuum = qm_calc_vals[qmat][xc]['vacuum'] )
            n_sub_x     = stack_repeat[qmat][bulk][1]
            n_mat_x     = stack_repeat[qmat][bulk][0]

            # Build stacked structure
            # Unit cell needs rotated if graphene or hbn.
            # I don't know why ASE doesn't fix this yet.
            if qmat == "graphene" or qmat == 'hbn':
                mat2D.rotate( 180, 'y', rotate_cell = True )

            
            mat2D:Atoms   = mat2D*[n_mat_x,n_mat_x,1]
            # Center
            mat2D.center( vacuum = qm_calc_vals[qmat][xc]['vacuum']/2., axis = 2 )

            # Slice bulk
            sub:Atoms     = sub*[n_sub_x,n_sub_x,1]
            # Center
            sub.center( vacuum = qm_calc_vals[qmat][xc]['vacuum']/2., axis = 2 )

            # Find max z-height of bulk
            substrate_top   = max( sub.positions[:, 2] )

            # Find min z-height of qm
            qm_min          = min( mat2D.positions[:, 2] )

            # Shift qm height
            z_offset                = substrate_top + 3.4 - qm_min
            mat2D.positions[:, 2]   += z_offset
            mat2D.positions[:, 0]   += stack_shifts[qmat][bulk][0]
            mat2D.positions[:, 1]   += stack_shifts[qmat][bulk][1]

            het:Atoms = sub + mat2D

            # Add the vacuum but keep it relatively small to allow my GPU to handle it
            # and to avoid weird issues with lower eV plane waves
            het.center( vacuum = qm_calc_vals[qmat][xc]['vacuum']/2., axis = 2 )

            sub_atom_cnt = len( sub )

            # Build the an extension to reduce future calculations
            sub_sv_name = f'{n_sub_x}x{n_sub_x}'
            mat_sv_name = f'{n_mat_x}x{n_mat_x}'

            # Build file names
            sub_pot_f   = os.path.join( sb_path, f"{xc}_{sub_sv_name}_poten.npy" )
            mat_pot_f   = os.path.join( crsy_path, f"{xc}_{mat_sv_name}_poten.npy" )

            # Set the filter for the substrate
            # NOTE: This only works because none of our substrates have similar atoms to the 2D mats
            sub_symbols = set(sub.symbols)

            # Freeze the substrate atoms
            sub_scaled_pos  = sub.get_positions()
            max_unit_z      = np.max(sub_scaled_pos[:, 2])
            min_unit_z      = np.min(sub_scaled_pos[:, 2])
            cell_z_thick    = (max_unit_z - min_unit_z)/2.  # Freeze lower half of substrate

            # unless it's SiO2, since those have temperature dependent structures
            if bulk == "SiO2_alpha_quartz" or bulk == "SiO2_beta_cristobalite":
                cell_z_thick = (max_unit_z - min_unit_z)

            het_pos         = het.get_positions()
            min_hetcell_z   = np.min( het_pos[:, 2] )

            # Build mask
            bottom_thresh   = min_hetcell_z + cell_z_thick
            fixed_indices   = [ atom.index for atom in het if ( atom.symbol in sub_symbols and atom.position[2] <= bottom_thresh ) ]
            bottom_const    = FixAtoms( indices = fixed_indices )
            het.set_constraint( bottom_const )

            # Check to see if the relaxation has been run before and start from where we left off
            if os.path.isfile( het_traj_file ):

                het:Atoms   = read( het_traj_file, index = "-1" )

                # Messed up some files. Gotta fix it.
                del het.constraints
                het.set_constraint( bottom_const )

            # Set PW cutoff
            # Think of the energy cutoff as the 'resolution' of the atomic structures
            # Higher energy cutoff means higher 'resolution'. But, we don't want to go to high or too low.
            # Because of this, we just go to the highest required for the atomic structures based off of our previous optimizations
            pw_cutoff   =  bulk_calc_vals[bulk][xc]['cutoff_energy']
            if pw_cutoff < qm_calc_vals[qmat][xc]['cutoff_energy']:
                pw_cutoff = qm_calc_vals[qmat][xc]['cutoff_energy']

            # Set custom params for GPAW, if necessary
            custom_params = None
            if ('Cu' in sub.symbols) or ('Ni' in sub.symbols):
                # Implemented due to metal (free electrons)
                custom_params = {
                    'mode': PW(pw_cutoff),
                    'basis': 'dzp',
                    'xc': 'PBE',
                    'kpts': {'density': 4.8, 'gamma': True},
                    'spinpol': True,
                    'convergence': { 'energy': 0.001, 'density': 1e-4, 'eigenstates': 1e-6 },
                    'parallel': {'sl_auto': True, 'gpu': True, 'use_elpa': True, 'augment_grids': True},
                    'mixer': Mixer( beta = 0.05, nmaxold = 10, weight = 50.0 ),
                    'occupations': { 'name': 'fermi-dirac', 'width': 0.2 },
                    'symmetry': {'point_group':False, 'time_reversal':False}
                }

            # Set the calculator (follow encut for bulk material)
            gpw_calc    = get_calculator( xc = xc, encut = pw_cutoff )
            calc        = DFTD4( method = 'PBE' ).add_calculator( gpw_calc )

            het.set_calculator( calc )

            # Relax the 2D material on top of the substrate
            optimizer = FIRE( het, trajectory = het_traj_file, append_trajectory = True, dt = 0.8 )
            optimizer.run( fmax = 0.05 )
            
            # Calculate the potential energy and save the values (sub+2D)
            het_Es[het_name][xc]                = {}
            het_Es[het_name][xc]['poten']       = het.get_potential_energy()
            het_Es[het_name][xc]['mat_size']    = n_mat_x

            # Explicitly delete the gcalc to avoid memory issues
            del gpw_calc
            del optimizer
            del calc

            # Calculate the potential energy and save the values (sub)
            # Check if xc and size has already been calculated for this sub
            if xc not in sub_Es[bulk]:
                sub_Es[bulk][xc] = {}

            # Check if sub energy exists already
            if os.path.isfile(sub_pot_f):
                sub_Es[bulk][xc][sub_sv_name] = np.load(sub_pot_f)
            else:
                if sub_sv_name not in sub_Es[bulk][xc]:
                    het_copy                        = het.copy()
                    sub_copy                        = het_copy[:sub_atom_cnt]
                    gpw_calc                        = get_calculator( xc = xc, encut = bulk_calc_vals[bulk][xc]['cutoff_energy'] )
                    sub_copy.set_calculator( gpw_calc )
                    sub_Es[bulk][xc][sub_sv_name]   = sub_copy.get_potential_energy()

                    # Explicitly delete the gcalc to avoid memory issues
                    del gpw_calc

                np.save( sub_pot_f, sub_Es[bulk][xc][sub_sv_name] )

            # Remove the substrate and freeze the 2D material
            # Check if xc and size has already been calculated for this 2D material
            if xc not in qmat_Es[qmat]:
                qmat_Es[qmat][xc] = {}

            # Check if 2D mat energy exists already
            if os.path.isfile( mat_pot_f ):
                qmat_Es[qmat][xc][mat_sv_name] = np.load(mat_pot_f)
            else:
                if mat_sv_name not in qmat_Es[qmat][xc]:
                    het_copy                        = het.copy()
                    mat_copy                        = het_copy[sub_atom_cnt:]
                    gpw_calc                        = get_calculator( xc = xc, encut = qm_calc_vals[qmat][xc]['cutoff_energy'] )
                    mat_copy.set_calculator( gpw_calc )
                    qmat_Es[qmat][xc][mat_sv_name]  = mat_copy.get_potential_energy()

                    # Explicitly delete the gcalc to avoid memory issues
                    del gpw_calc

                    # Save the 2D material structure for use in calculating phonons later.
                    write( qmat_traj, mat_copy )

                np.save( mat_pot_f, qmat_Es[qmat][xc][mat_sv_name] )

            # Build the dE_elec dictionary (normalized to 2D material unit cell)
            dE_sub2D[het_name][xc] = ( het_Es[het_name][xc]['poten'] - sub_Es[bulk][xc][sub_sv_name] - qmat_Es[qmat][xc][mat_sv_name] )/( het_Es[het_name][xc]['mat_size']*het_Es[het_name][xc]['mat_size'] )

            print( f"The binding energy for {het_name} with xc {xc} is: {dE_sub2D[het_name][xc]:.2f} eV" )

            # Save the calculation
            np.save( het_bind_en, dE_sub2D[het_name][xc] )
#endregion



