import os

import pandas as pd


def is_pandarallel_initialized():
    '''Since `pandarallel.initialize()` adds the `parallel_apply` method to `pandas.DataFrame`, this fucntion checks for its existence.'''
    return hasattr(pd.DataFrame, 'parallel_apply')


def initialize_pandarallel(num_cores_for_pandarallel: int = os.cpu_count() // 3):
    if is_pandarallel_initialized():
        print('pandarallel has already been initialized')
    else:
        from pandarallel import pandarallel    # used to multi-thread df apply with `.parallel_apply(...)`
        pandarallel.initialize(progress_bar=False, nb_workers=num_cores_for_pandarallel)
        print(f'Initialized pandarallel with {num_cores_for_pandarallel} cores')
