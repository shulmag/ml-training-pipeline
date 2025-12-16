'''
Create date: 2024-11-16
Description: Set the random seed for all modules used throughout this code base.
'''
import os
import random
import numpy as np
import tensorflow as tf


def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
