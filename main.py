import os
import json
from .qm_crystals import *
from .bulk_crystals import *
from .calculator import crystal_optimizer, xc_list, get_calculator
from ase.filters import FrechetCellFilter   # Allows for scaling the unit cell to minimize the energy
from ase.constraints import FixSymmetry     # Used to maintain our unit cell dimensions
from ase.optimize import FIRE               # The algorithm for minimizing the forces
from ase.io import Trajectory, read, write
from dftd4.ase import DFTD4
from gpaw import PW
from ase import Atoms
import numpy as np
from .heterostructure_adjustments import bulk_substrates_thicknesses, orientations, combinations, stack_repeat, stack_shifts
from ase.build import surface
from ase.constraints import FixAtoms
from gpaw.mixer import Mixer


class mainGPUTest:

    # Initializes the test variables
    def __init__( self, fmax = 0.01, kpt_dens = 4.8 ):

        self.bulk_substrates    = None
        self.materials_2d       = None
        self.fmax               = fmax
        self.kpt_dens           = kpt_dens

        self.bulk_calc_vals     = {}
        self.qm_calc_vals       = {}
        self.dE_sub2D           = {}
        self.sub_Es             = {}
        self.qmat_Es            = {}
        self.het_Es             = {}

        self.save_path          = os.path.join( os.getcwd(), "DFT Output DFTD4", "bulk" )
        self.bulk_path          = os.path.join( self.save_path, "bulk" )
        self.qm_path            = os.path.join( self.save_path, "2D materials" )
        self.het_path           = os.path.join( self.save_path, "heterostructures" )
        self.sub_path           = os.path.join( self.save_path, "substrates" )
        os.makedirs( self.save_path, exist_ok = True )
        os.makedirs( self.bulk_path, exist_ok = True )
        os.makedirs( self.qm_path, exist_ok = True )
        os.makedirs( self.het_path, exist_ok = True )
        os.makedirs( self.sub_path, exist_ok = True )

    # Builds the bulk substrate materials for the simulation
    def build_bulks( self ):
        self.bulk_substrates = {
            # 'SiO2_alpha_quartz': create_sio2_alpha_quartz(),
            'SiO2_beta_cristobalite': create_sio2_beta_cristobalite()#,
            # 'Cu_111': create_copper_111(),
            # 'Ni_111': create_nickel_111(),
            # 'Sapphire_Al2O3_0001': create_sapphire_0001()
        }

    # Builds the 2D materials for the simulation
    def build_2DMats( self ):
        # Create the 2D materials
        self.materials_2d = {
            # 'graphene': create_graphene(),
            # 'hbn': create_hbn(),
            'mos2': create_mos2()#,
            # 'mose2': create_mose2(),
            # 'ws2': create_ws2(),
            # 'wse2': create_wse2()
        }

    # Optimizes the bulk crystal GPAW parameters
    def optimize_bulk_crystals( self ):

        assert self.bulk_substrates is not None, "self.bulk_substrates has not been declared. Run self.build_bulks first!"
    
        # Cycle through each bulk material and optimize
        for bulk in self.bulk_substrates:
            # Create the directories needed, if they don't exist.
            bulk_path = os.path.join( self.bulk_path, f"{bulk}" )
            os.makedirs( bulk_path, exist_ok = True )

            print( f"Optimizing crystal {bulk}:" )

            self.bulk_calc_vals[bulk] = crystal_optimizer( crystal = self.bulk_substrates[bulk], crystal_path = bulk_path, is_bulk = True )

    # Optimizes the bulk crystal structure
    def optimize_bulk_structures( self ):

        assert self.bulk_substrates is not None, "self.bulk_substrates has not been declared. Run self.build_bulks first!"

        # Cycle through all substrates
        for bulk in self.bulk_substrates:

            bulk_path   = os.path.join( self.bulk_path, f"{bulk}" )
            traj_path   = os.path.join( bulk_path, "trajectories" )
            os.makedirs( traj_path, exist_ok = True )

            # After this is done, we will continue with the rest of the XCs
            for xc in xc_list:

                pot_file        = os.path.join( bulk_path, f'{xc}_relaxed.json' )
                xc_traj_file    = os.path.join( traj_path, f'{xc}_relaxation.traj' )

                # If the gpaw file exists, this has been done already.
                if not os.path.isfile(pot_file):
                    
                    # Set the crystal 
                    crystal     = self.bulk_substrates[bulk].copy()

                    # If a relaxation has been started, just copy the last frame of the relaxation
                    # file and use that for the copy.
                    if os.path.isfile( xc_traj_file ):
                        crystal:Atoms = read( xc_traj_file, index = "-1" )
                        
                    gpw_calc    = get_calculator( xc = xc, encut = self.bulk_calc_vals[bulk][xc]['cutoff_energy'] )
                    calc        = DFTD4( method = 'PBE' ).add_calculator( gpw_calc )

                    crystal.set_calculator( calc )

                    # Fix the symmetry
                    crystal.set_constraint( FixSymmetry( crystal ) )

                    # Set the ExpCellFilter to also relax the cell parameters
                    ecf         = FrechetCellFilter( crystal )

                    #optimizer   = BFGSLineSearch( ecf, trajectory = xc_traj_file, append_trajectory = True )
                    optimizer   = FIRE( ecf, trajectory = xc_traj_file, append_trajectory = True )
                    optimizer.run( fmax = self.fmax )

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

    # Optimize the 2D crystal GPAW parameters
    def optimize_qm_crystals( self ):

        assert self.materials_2d is not None, "self.materials_2d has not been declared. Run self.build_2DMats first!"

        # Cycle through each bulk material and optimize
        for qmmat in self.materials_2d:

            # Create the directories needed, if they don't exist.
            sv_path = os.path.join( self.qm_path, f"{qmmat}" )
            os.makedirs( sv_path, exist_ok = True )

            print( f"Optimizing 2D material {qmmat}:" )

            self.qm_calc_vals[qmmat] = crystal_optimizer( crystal = self.materials_2d[qmmat], crystal_path = sv_path )

    # Build heterostructures and relax the 2D materials to calculate binding energies
    def calculate_binding_energies( self ):

        for bulk, qmat in combinations:

            het_name            = f"{qmat}_on_{bulk}"

            blk_path            = os.path.join( self.bulk_path, f"{bulk}" )
            sb_path             = os.path.join( self.sub_path, f"{bulk}" )
            crsy_path           = os.path.join( self.qm_path, f"{qmat}" )
            hetero_path         = os.path.join( self.het_path, het_name )
            traj_path           = os.path.join( blk_path, "trajectories" )
            os.makedirs( sb_path, exist_ok = True )
            os.makedirs( crsy_path, exist_ok = True )
            os.makedirs( hetero_path, exist_ok = True )

            # Read the cell parameters of blk as compared to the original size.
            blk_cpy             = surface( self.bulk_substrates[bulk].copy(), orientations[bulk], bulk_substrates_thicknesses[bulk], vacuum = 5. )
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

                    self.dE_sub2D[het_name][xc] = np.load( het_bind_en )
                
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
                    mat2D           = self.materials_2d[qmat].copy()

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
                    mat2D.center( vacuum = self.qm_calc_vals[qmat][xc]['vacuum']/2., axis = 2 )

                    # Slice bulk
                    sub:Atoms     = sub*[n_sub_x,n_sub_x,1]
                    # Center
                    sub.center( vacuum = self.qm_calc_vals[qmat][xc]['vacuum']/2., axis = 2 )

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
                    het.center( vacuum = self.qm_calc_vals[qmat][xc]['vacuum']/2., axis = 2 )

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
                    pw_cutoff   =  self.bulk_calc_vals[bulk][xc]['cutoff_energy']
                    if pw_cutoff < self.qm_calc_vals[qmat][xc]['cutoff_energy']:
                        pw_cutoff = self.qm_calc_vals[qmat][xc]['cutoff_energy']

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
                    self.het_Es[het_name][xc]                = {}
                    self.het_Es[het_name][xc]['poten']       = het.get_potential_energy()
                    self.het_Es[het_name][xc]['mat_size']    = n_mat_x

                    # Explicitly delete the gcalc to avoid memory issues
                    del gpw_calc
                    del optimizer
                    del calc

                    # Calculate the potential energy and save the values (sub)
                    # Check if xc and size has already been calculated for this sub
                    if xc not in self.sub_Es[bulk]:
                        self.sub_Es[bulk][xc] = {}

                    # Check if sub energy exists already
                    if os.path.isfile(sub_pot_f):
                        self.sub_Es[bulk][xc][sub_sv_name] = np.load(sub_pot_f)
                    else:
                        if sub_sv_name not in self.sub_Es[bulk][xc]:
                            het_copy                        = het.copy()
                            sub_copy                        = het_copy[:sub_atom_cnt]
                            gpw_calc                        = get_calculator( xc = xc, encut = self.bulk_calc_vals[bulk][xc]['cutoff_energy'] )
                            sub_copy.set_calculator( gpw_calc )
                            self.sub_Es[bulk][xc][sub_sv_name]   = sub_copy.get_potential_energy()

                            # Explicitly delete the gcalc to avoid memory issues
                            del gpw_calc

                        np.save( sub_pot_f, self.sub_Es[bulk][xc][sub_sv_name] )

                    # Remove the substrate and freeze the 2D material
                    # Check if xc and size has already been calculated for this 2D material
                    if xc not in self.qmat_Es[qmat]:
                        self.qmat_Es[qmat][xc] = {}

                    # Check if 2D mat energy exists already
                    if os.path.isfile( mat_pot_f ):
                        self.qmat_Es[qmat][xc][mat_sv_name] = np.load(mat_pot_f)
                    else:
                        if mat_sv_name not in self.qmat_Es[qmat][xc]:
                            het_copy                        = het.copy()
                            mat_copy                        = het_copy[sub_atom_cnt:]
                            gpw_calc                        = get_calculator( xc = xc, encut = self.qm_calc_vals[qmat][xc]['cutoff_energy'] )
                            mat_copy.set_calculator( gpw_calc )
                            self.qmat_Es[qmat][xc][mat_sv_name]  = mat_copy.get_potential_energy()

                            # Explicitly delete the gcalc to avoid memory issues
                            del gpw_calc

                            # Save the 2D material structure for use in calculating phonons later.
                            write( qmat_traj, mat_copy )

                        np.save( mat_pot_f, self.qmat_Es[qmat][xc][mat_sv_name] )

                    # Build the dE_elec dictionary (normalized to 2D material unit cell)
                    self.dE_sub2D[het_name][xc] = ( self.het_Es[het_name][xc]['poten'] - self.sub_Es[bulk][xc][sub_sv_name] - self.qmat_Es[qmat][xc][mat_sv_name] )/( self.het_Es[het_name][xc]['mat_size']*self.het_Es[het_name][xc]['mat_size'] )

                    print( f"The binding energy for {het_name} with xc {xc} is: {self.dE_sub2D[het_name][xc]:.2f} eV" )

                    # Save the calculation
                    np.save( het_bind_en, self.dE_sub2D[het_name][xc] )

if __name__ == "__main__":
    os.environ['GPAW_NEW']      = 'True'
    os.environ['GPAW_USE_GPUS'] = 'True'
    MGT = mainGPUTest()

    # Build materials
    MGT.build_bulks()
    MGT.build_2DMats()

    # Optimize the bulks
    MGT.optimize_bulk_crystals()
    MGT.optimize_bulk_structures()

    # Optimize the QM materials
    MGT.optimize_qm_crystals()

    # Calculate the binding energies
    MGT.calculate_binding_energies()
