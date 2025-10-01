from ase.spacegroup import crystal

## Substrates
def create_sio2_alpha_quartz():
    """Create a SiO2 alpha quartz substrate."""

    # Values taken from Materials Project mp-6930

    a = b = 4.92
    c = 5.43

    bulk = crystal(
        ('Si', 'O'), # Formula unit in the cell
        basis=[(0., 0.468798, 1./3.), (0.144538, 0.73144, 0.54782)],
        spacegroup=154,
        cellpar=[a, b, c, 90, 90, 120]
    )

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
        cellpar=[a, b, c, 90, 90, 120]
    )

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
        cellpar = [a, b, c, 90, 90, 90]
    )

    return bulk

def create_nickel_111(size=(3, 3, 1)):
    """Create a Ni(111) substrate."""
    # mp-23
    a = b = c = 3.48 

    bulk = crystal(
        ('Ni'), # Formula unit in the cell
        basis = [(0., 0., 0.)],
        spacegroup = 225,
        cellpar = [a, b, c, 90, 90, 90]
    )

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