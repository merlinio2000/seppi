import numpy as np
import sympy as sym

from src.hm2.linearisierung import newton_systeme_d
from src.util import utl

sx, sy, sa, sb = sym.symbols('x y a b')

sym.init_printing()

a = 2.
b = 4.

sf1 = 1 - sx**2 - sy**2
sf2 = (sx-2)**2 / sa + (sy - 1)**2 / sb - 1

x0 = np.array([2, -1], dtype=np.float64)
sx0 = sym.Matrix([2, -1])


# Aufgabe a)

sf = sym.Matrix([sf1, sf2])
X = sym.Matrix([sx, sy])
sDf = sf.jacobian(X)

sym.pprint(sDf)

sDf_x0 = sDf.subs([(sx, x0[0]), (sy, x0[1])])
sA = sym.simplify(sDf_x0)
sym.pprint(sA)

c = sym.simplify(-sf.subs([(sx, x0[0]), (sy, x0[1])]))
sym.pprint(c)

sdelta = sA.solve(c)
print('delta')
sym.pprint(sdelta)

print('x1 = x0 + delta')
sym.pprint(sx0 + sdelta)

# Aufgabe b)
sf = sf.subs([(sa, a),(sb, b)])
f = sym.lambdify([(sx, sy)], sf)
Df = sym.lambdify([(sx, sy)], sA.subs([(sa, a),(sb, b)]))

tol = 1e-8
krit = utl.AbbruchKriteriumFXNormKleinerTol(f, tol, np.inf)

x_nst = newton_systeme_d(x0, f, Df, krit, p_max=5)

print(x_nst)

assert np.allclose(x_nst, np.array([0.9439, -0.3302]), atol=1e-4)


# Aufgabe c)


sf1 = sf1.subs([(sa, a), (sb, b)])
sf2 = sf2.subs([(sa, a), (sb, b)])
p1 = sym.plot_implicit(sym.Eq(sf1, 0), (sx, -2, 4), (sy, -2, 4))
p2 = sym.plot_implicit(sym.Eq(sf2, 0), (sx, -2, 4), (sy, -2, 4), line_color='red')
p1.append(p2[0])
p1.show()


