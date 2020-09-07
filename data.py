import numpy as np
from functools import partial


class DataGenerator():

    def __init__(self, d = 4, r_0 = np.random.normal, r_1 = partial(np.random.normal, loc = 1)):
        super().__init__()