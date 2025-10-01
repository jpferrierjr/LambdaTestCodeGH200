from ase.build import mx2, graphene

## 2D Materials
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