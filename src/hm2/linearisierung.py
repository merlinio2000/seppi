import src.util.utl as utl
import numpy as np
from typing import Callable

# TODO HM2: unit test


def scipy_jacobi_bsp():
    ''' 
    Erstellt die Jacobimatrix für das System der Vektorfunktion f

    Aus Serie 2: Aufgabe 2: Jacobi-Matrix b) 
    '''
    import sympy as sp
    sp.init_printing(pretty_print=True)
    x, y, z = sp.symbols('x y z')
    f1 = sp.ln(x**2 + y**2) + z**2
    f2 = sp.exp(y**2 + z**2) + x**2
    f3 = 1 / (z**2 + x**2) + y**2

    f = sp.Matrix([f1, f2, f3])
    X = sp.Matrix([x, y, z])
    Df = f.jacobian(X)
    print(f'Jacobi-Matrix:')
    sp.pprint(Df)


def scipy_linearisieren_bsp():
    '''  
    Linearisiert das Gleichungssystem (f1, f2, f3) um den Punkt x0

    g(x) = f(x0) + Df(x0) * (x - x0)

    Aus Serie 2: Aufgabe 3: Linearisieren der Vektorfunktion f
    '''
    import sympy as sp
    sp.init_printing(pretty_print=True)

    x, y, z = sp.symbols('x y z')
    f1 = x + y**2 - z**2 - 13
    f2 = sp.ln(y/4) + sp.exp(0.5 * z - 1) - 1
    f3 = (y - 3)**2 - z**3 + 7

    x0 = sp.Matrix([1.5, 3, 2.5]).T
    f = sp.Matrix([f1, f2, f3])
    X = sp.Matrix([x, y, z])
    Df = f.jacobian(X)
    print(f'Jacobi-Matrix:')
    sp.pprint(Df)

    # Linearisierung
    f0 = f.subs([(x, x0[0]), (y, x0[1]), (z, x0[2])])
    f0 = f0.evalf()
    Df0 = Df.subs([(x, x0[0]), (y, x0[1]), (z, x0[2])])
    Df0 = Df0.evalf()

    # Delta Vektor (x - x0)
    p1 = x - x0[0]
    p2 = y - x0[1]
    p3 = z - x0[2]
    p = sp.Matrix([p1, p2, p3])

    g = f0 + Df0 * p
    g = g.evalf()
    print(f'g:')
    sp.pprint(g)


def newton_systeme_d(x0: np.ndarray, f:  Callable[[np.ndarray], np.ndarray], \
        Df: Callable[[np.ndarray], np.ndarray], krit: utl.AbbruchKriteriumHandler, \
        p_max: int = 0):
    '''
    Löst das Newton-Verfahren zur Nullstellenbestimmung für ein System iterativ
    vec{f}(x1, ..., x_n) = vec{0}

    Optional auch mit Dämpfung, siehe p_max

    Parameters:
        x0: Startvektor der Iteration (sollte nahe der Nullstelle sein)
        f:  Vektorfunktion f(x) (Argument muss ein Vektor sein, nicht n separate Werte)
        Df: Funktion Df(x) die die Jacobi-Matrix von f für einen gewissen Vektor
            x berechnet (Argument muss ein Vektor sein, nicht n separate Werte)
        krit: spezifische Instanz von AbbruchKriteriumHandler
        p_max = 0: maximaler Dämpfungsgrad, 0 bedeutet ungedämpft
    Returns:
        xi: Vektor x nach erreichen des Abbruchkriteriums
    '''

    utl.assert_is_vec(x0)


    # zwingend konsistent mit ergebnis von linalg.solve
    # weil addiert wird
    x0 = x0.astype(np.float64).flatten()
    i = 0
    x_curr = np.copy(x0)

    delta = None
    
    # Validieren ob mitgegebene Funktionen Sinn machen
    utl.assert_dimensions_match(Df(x0), x0)
    assert len(x0) == len(f(x0))

    print(f'Newton-Verfahren für nichtlineare Systeme mit maximaler Dämpfung={p_max}')
    print('Startvektor:')
    utl.np_pprint(x0)

    while krit.keep_going(curr_i=i, curr_x=x_curr, last_delta=delta):
        A = Df(x_curr)
        c = -f(x_curr)
        delta = np.linalg.solve(A, c).flatten()

        p_min = 0
        base_norm = np.linalg.norm(f(x_curr), 2)
        for p in range(0, p_max+1):
            # check ist die Lösung eine 'Verbesserung'?
            if np.linalg.norm(f(x_curr + delta / 2**p), 2) < base_norm:
                p_min = p
                break
        delta = delta / 2**p_min

        print(f'Iteration {i}: dämpfung={p_min}')
        print('delta:')
        utl.np_pprint(delta)
        x_curr += delta 
        print(f'x_{i+1}:')
        utl.np_pprint(x_curr)
        i += 1


    return x_curr






import unittest


class LinearisierungTest(unittest.TestCase):
    def test_newton_S3_A1(self):
        from sympy import symbols, Matrix, lambdify

        x1, x2 = symbols('x1 x2')

        f1 = 20 - 18*x1 - 2*x2**2
        f2 = -4 * x2 * (x1 - x2**2)

        sf = Matrix([f1, f2])
        X = Matrix([x1, x2])
        sDf = sf.jacobian(X)

        f = lambdify([(x1, x2)], sf, 'numpy')
        Df = lambdify([(x1, x2)], sDf, 'numpy')

        krit = utl.AbbruchKriteriumNIterationen(2)

        x0 = np.array([1.1, 0.9])
        nullstelle = newton_systeme_d(x0, f, Df, krit)

        self.assertAlmostEqual(nullstelle[0], 0.99986314)
        self.assertAlmostEqual(nullstelle[1], 1.00092549)
    
    def test_newton_S3_A3(self):
        from sympy import symbols, Matrix, lambdify, ln, exp

        x1, x2, x3 = symbols('x1 x2 x3')

        f1 = x1 + x2**2 - x3**2 - 12
        f2 = ln(x2/4) + exp(0.5*x3 - 1) - 1
        f3 = (x2 - 3)**2 - x3**3 + 7

        sf = Matrix([f1, f2, f3])
        X = Matrix([x1, x2, x3])
        sDf = sf.jacobian(X)

        f = lambdify([(x1, x2, x3)], sf, 'numpy')
        Df = lambdify([(x1, x2, x3)], sDf, 'numpy')

        tol = 1e-5
        krit = utl.AbbruchKriteriumFXNormKleinerTol(f, tol)

        x0 = np.array([1.5, 3, 2.5])
        nullstelle = newton_systeme_d(x0, f, Df, krit, p_max=5)

        self.assertAlmostEqual(nullstelle[0], 0.00000009)
        self.assertAlmostEqual(nullstelle[1], 4.)
        self.assertAlmostEqual(nullstelle[2], 2.)


