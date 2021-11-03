"""User specific settings.

Expected to be changed by each user
"""

import matplotlib as mpl

mpl.rcParams['font.size'] = 7
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'arial'
mpl.rcParams['mathtext.fontset'] = 'stix'

use_torch = True
cluster_path = '/share/ctn/users/gy2259/olfaction_evolution'
FILEPATH = './files/'