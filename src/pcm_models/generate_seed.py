from functools import partial
from random import randint
import numpy as np


def randints(count, *randint_args):
    ri = partial(randint, *randint_args)
    return [ri() for _ in range(count)]


x = randints(10, 1, 1000)

print(x)
np.save('seeds.npy', x)
