import numpy as np

"""Finds the max value of any item embedded in a dict of lists of items. Assumes
    the items are numerical (integer or float).

    Returns the max value found.
"""

def get_max(d       : {}    # a dict of lists of items
            ):
    
    m = -np.inf
    for item_list in d:
        for item in item_list:
            if item > m:
                m = item
    
    return m

#------------------------------------------------------------------------------
