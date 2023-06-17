import sympy as sp
import utl
from utl import AbbruchKriteriumHandler
import numpy as np
from typing import Callable

sp.init_printing(pretty_print=True)

def scipy_jacobi_bsp():
    ''' 
    Erstellt die Jacobimatrix für das System der Vektorfunktion f

    Aus Serie 2: Aufgabe 2: Jacobi-Matrix b) 
    '''
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


def newton_systeme(x0: np.ndarray, Df:  Callable[[np.ndarray], np.ndarray], \
        f: Callable[[np.ndarray], np.ndarray], krit: AbbruchKriteriumHandler, \
        p_max: int = 0):
    '''
    Löst das Newton-Verfahren zur Nullstellenbestimmung für ein System iterativ
    \vec{f}(x1, ..., x_n) = \vec{0}

    Optional auch mit Dämpfung, siehe p_max

    Parameters:
        x0: Startvektor der Iteration (sollte nahe der Nullstelle sein)
        Df: Funktion Df(x) die die Jacobi-Matrix von f für einen gewissen Vektor
            x berechnet (Argument muss ein Vektor sein, nicht n separate Werte)
        f:  Vektorfunktion f(x) (Argument muss ein Vektor sein, nicht n separate Werte)
        p_max = 0: maximaler Dämpfungsgrad, 0 bedeutet ungedämpft
    Returns:
        xi: Vektor x nach erreichen des Abbruchkriteriums
    '''

    utl.assert_is_vec(x0)


    x0 = x0.astype(np.float64)
    i = 0
    x_curr = np.copy(x0)

    delta = np.zeros_like(x0)

    while krit(curr_i=i, curr_x=x_curr, delta=delta):
        A = Df(x_curr)
        utl.assert_dimensions_match(A, x0)
        c = -f(x_curr)
        utl.assert_eq_shape(x0, c)
        delta = np.linalg.solve(A, c)

        p_min = 0
        base_norm = np.linalg.norm(f(x_curr), 2)
        for p in range(1, p_max):
            # check ist die Lösung eine 'Verbesserung'?
            if np.linalg.norm(f(x_curr + delta / 2**p), 2) < base_norm:
                p_min = p
                break

        x_curr += delta / 2**p_min
        i += 1


    return x_curr


