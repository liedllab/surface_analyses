from .crippen import logp
from .eisenberg import amber_to_eisen_value

import pandas as pd


def get_propensity_mapping(scale):
    if scale == 'crippen':
        return logp
    elif scale == 'eisenberg':
        def propensity(a):
            if a.atomic_number == 1:
                return 0
            return amber_to_eisen_value(a)
        return propensity
    else:
        mapping = load_scale(scale)
        def propensity(a):
            return mapping[a.residue_label]
        return propensity


def load_scale(fname):
    scale = pd.read_csv(
        fname,
        names=['residue', 'propensity'],
        skiprows=1,
        index_col='residue',
    )['propensity']
    assert isinstance(scale, pd.Series), 'Scale needs to be a 2-column csv.'
    assert not scale.index.has_duplicates, 'Duplicate keys in scale.'
    return scale
