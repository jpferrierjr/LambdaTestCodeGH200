

# This is chosen just to reduce computational times.
# Error propagated should be minimal, as the binding energy is mostly local
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