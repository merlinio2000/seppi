import numpy as np
from typing import Callable, Union

'''
FS20 1b)
'''

u = 2_000
m_0 = 10_000
q = 100
g = 9.8

def v(t):
    return u * np.log(m_0 / (m_0 - q*t)) - g*t



# TODO refactor so x is not required instead calculated from h
# TODO add prints
def Rf(h: float, x: np.ndarray, f: Callable[[np.ndarray], np.ndarray]):
    return h * np.sum(f(x[:-1] + h/2))

# TODO refactor so x is not required instead calculated from h
# TODO add prints
def Tf(a:float, b: float, x:np.ndarray, f: Callable[[Union[np.ndarray, float]], np.ndarray]):
    return h * ((f(a) + f(b)) / 2 + np.sum(f(x[1:-1])))

h = 10

T = 30

x = np.arange(0, T + h, step=h, dtype=np.float64)

print(Tf(0, 30, x, v))

'''
c)
'''
