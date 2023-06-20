import numpy as np
import sympy as sym
from src.hm2.linearisierung import newton_systeme_d
from src.util import utl

x1 = 1
y1 = 40
x2 = 1.6
y2 = 250
x3 = 2
y3 = 800

sa, sb, sc = sym.symbols('a b c')

sf1 = sa + sb*sym.exp(sc*x1) - y1
sf2 = sa + sb*sym.exp(sc*x2) - y2
sf3 = sa + sb*sym.exp(sc*x3) - y3

sf = sym.Matrix([sf1, sf2, sf3])
X = sym.Matrix([sa, sb, sc])
sDf = sf.jacobian(X)


f = sym.lambdify([(sa, sb, sc)], sf)
Df = sym.lambdify([(sa, sb, sc)], sDf)

x0 = np.array([1., 2, 3])

krit = utl.AbbruchKriteriumNIterationen(1)

faktoren = newton_systeme_d(x0, f, Df, krit)

utl.np_pprint(faktoren)
