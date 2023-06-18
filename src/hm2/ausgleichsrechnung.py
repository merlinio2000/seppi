import numpy as np
from typing import Callable
import src.util.types as types
import src.util.utl as utl


def fehlerfunktional(x: types.NPValue, y_exakt: types.NPValue, f_ansatz: types.NPValueToValueFn) \
        -> float:
    '''
    Bestimmt das Fehlerfunktional einer Ansatzfunktion f am Punkt x, f(x) nach
    der Methode der kleinsten Fehlerquadrate

    Parameters:
        x: x an dem das Fehlerfunktional bestimmt wird
        y_exakt: das EXAKTE y zu diesem x
        f_ansatz: die Ansatzfunktion f
    Returns:
        float: das Fehlerfunktional E(f) = ||y - f(x)||_2^2
    '''
    return np.linalg.norm(y_exakt - f_ansatz(x), 2)**2 # type: ignore


def lin_ausgleich(x: np.ndarray, y: np.ndarray, fs: list[types.NPValueToValueFn]) \
        -> np.ndarray:
    '''
    Bestimmt lambdas l_i der linearen Ausgleichsfunktion f(x) = l_1 * f_1(x) + 
        ... + l_m * f_m(x)
    sodass das das Fehlerfunktional der kleinsten Fehlerquadrate minimal wird.

    Parameters:
        x: x der Messpunkte, länge = n
        y: y der Messpunkte, länge = n
        fs: liste der Basisfunktionen f_1,...,f_m; länge = m
    Returns:
        Vektor: mit einzelnen Lambdas l_i
    '''
    utl.assert_is_vec(x)
    utl.assert_eq_shape(x, y)

    fs_evaluated = [f(x) for f in fs]

    base_len = len(fs_evaluated[0]) # type: ignore
    assert all([isinstance(row, np.ndarray) and len(row) == base_len \
            for row in fs_evaluated])

    A = np.column_stack(fs_evaluated)

    print('linearer Ausgleich')
    print('A-Matrix:')
    utl.np_pprint(A)
   
    # Besser konditioniert als A^T A
    Q, R = np.linalg.qr(A)
    
    return np.linalg.solve(R, Q.T @ y)


def gauss_newton_d(lam0: np.ndarray, g: Callable[[np.ndarray], np.ndarray], \
         Dg: Callable[[np.ndarray], np.ndarray], krit: utl.AbbruchKriteriumHandler, \
         p_max: int = 0) -> np.ndarray:
    '''
    Löst das Gauss-Newton-Verfahren zur nichtlinearen Ausgleichsrechnung für ein System iterativ

    Optional auch mit Dämpfung, siehe p_max

    Parameters:
        lam0: Startvektor der Iteration
        g:  Fehlerfunktion g(lam) = y - f(lam), f ist die Ausgleichsfunktion 
                (Argument muss ein Vektor sein, nicht n separate Werte)
        Dg: Funktion Dg(lam) die die Jacobi-Matrix von g für einen gewissen Vektor
            lam berechnet (Argument muss ein Vektor sein, nicht n separate Werte)
        krit: spezifische Instanz von AbbruchKriteriumHandler
        p_max = 0: maximaler Dämpfungsgrad, 0 bedeutet ungedämpft
    Returns:
        lami: Vektor lam nach erreichen des Abbruchkriteriums
    '''
    utl.assert_is_vec(lam0)
    assert len(lam0.shape) == 1


    curr_lam = lam0.astype(np.float64)

    utl.assert_is_vec(g(curr_lam))
    assert Dg(curr_lam).shape[0] ==  g(curr_lam).shape[0]
    assert len(curr_lam) == Dg(curr_lam).shape[1]

    k = 0
    delta = None 
    
    print(f'Gauss-Newton-Verfahren zur nichtlinearen Ausgleichsrechung, maximale Dämpfung={p_max}')
    print('lambda_0:')
    utl.np_pprint(curr_lam)
    while krit.keep_going(curr_i=k, curr_x=curr_lam, last_delta=delta):
        # QR-Zerlegung von Dg(lam) und delta als Lösung des lin. Gleichungssystems
        Q, R = np.linalg.qr(Dg(curr_lam))
        # flatten ZWINGEND sonst shape mismatch mit lambda -> nicht elementweise addition         
        delta = np.linalg.solve(R, -Q.T @ g(curr_lam)).flatten()

        p_min = 0
        base_norm = np.linalg.norm(g(curr_lam), 2)
        for p in range(0, p_max + 1):
            if np.linalg.norm(g(curr_lam + delta / 2**p)) < base_norm:
                p_min = p
                break
        delta = delta / 2**p_min
               
        # Update des Vektors Lambda        
        curr_lam += delta

        print(f'Iteration {k}')
        print('Dämpfung = ', p_min)
        print(f'lambda_{k+1}:')
        utl.np_pprint(curr_lam)

        increment = np.linalg.norm(delta, 2)
        print('Inkrement = ',increment)
        err_func = np.linalg.norm(g(curr_lam), 2)**2
        print('Fehlerfunktional =', err_func)

        k += 1

    return curr_lam



import unittest

class AusgleichsTest(unittest.TestCase):
    def test_lin_ausgleich_S6_A1(self):
        x = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], dtype=np.float64)
        y = np.array([999.9, 999.7, 998.2, 995.7, 992.2, 988.1, 983.2, 977.8, 971.8, 965.3, 958.4], dtype=np.float64)
        assert len(x) == len(y)

        f1 = lambda x: x**2
        f2 = lambda x: x
        f3 = lambda x: np.ones_like(x)
        fs = [f1, f2, f3]

        def f(x, lams):
            return np.sum([lam*f_i(x) for lam, f_i in zip(lams, fs)], axis=0)

        our_lambdas = lin_ausgleich(x, y, fs)

        np_lambdas = np.polyfit(x, y, 2)

        self.assertTrue(np.allclose(our_lambdas, np_lambdas))
        
        # import matplotlib.pyplot as plt
        #
        # x_int = np.linspace(0, 100, num=100_000)
        # y_np = np.polyval(np_lambdas, x_int)
        # y_ours_f = f(x_int, our_lambdas)
        #
        # plt.plot(x, y, 'bx', label='Messpunkte')
        # plt.plot(x_int, y_ours_f, 'r', label='unser')
        # plt.plot(x_int, y_np, 'g', label='numpy')
        # plt.legend()
        # plt.grid()
        # plt.show()
    
    def test_gauss_newton_S7_A2a(self):
        x = np.array([2., 2.5, 3., 3.5, 4., 4.5, 5., 5.5, 6., 6.5, 7., 7.5, 8.,
                      8.5, 9., 9.5])
        y = np.array([159.57209984, 159.8851819, 159.89378952, 160.30305273,
                      160.84630757, 160.94703969, 161.56961845, 162.31468058,
                      162.32140561, 162.88880047, 163.53234609, 163.85817086,
                      163.55339958, 163.86393263, 163.90535931, 163.44385491])
        assert len(x) == len(y)
        
        from sympy import symbols, Matrix, lambdify
        p = symbols('p0 p1 p2 p3')

        def f(x, _p): # fit Funktion
            return (_p[0] + _p[1] * 10 ** (_p[2] + _p[3] * x)) / (1 + 10 ** (_p[2] + _p[3] * x))


        lam_0 = np.array([100, 120, 3, -1], dtype=np.float64)
        tol = 1e-5

        krit = utl.AbbruchKriteriumDeltaNormKleinerToleranz(tol)

        g = Matrix([y[k] - f(x[k], p) for k in range(len(x))])
        Dg = g.jacobian(p)
        g = lambdify([p], g, 'numpy')
        Dg = lambdify([p], Dg, 'numpy')

        our_lambdas = gauss_newton_d(lam_0, g, Dg, krit, p_max=5)
        print(our_lambdas)
        expected_lambdas = \
                np.array([163.88257571, 159.47423746, 2.17222151, -0.42933953])
        print(expected_lambdas)

        self.assertTrue(np.allclose(our_lambdas, expected_lambdas))

    def test_scipy_opt_bsp(self):
        '''
        Beispiel zur Ausgleichsrechung mit scipy
        Entspricht dem gleichen Problem wie test_gauss_newton_S7_A2a
        '''
        x = np.array([2., 2.5, 3., 3.5, 4., 4.5, 5., 5.5, 6., 6.5, 7., 7.5, 8.,
                      8.5, 9., 9.5])
        y = np.array([159.57209984, 159.8851819, 159.89378952, 160.30305273,
                      160.84630757, 160.94703969, 161.56961845, 162.31468058,
                      162.32140561, 162.88880047, 163.53234609, 163.85817086,
                      163.55339958, 163.86393263, 163.90535931, 163.44385491])
        assert len(x) == len(y)
        
        from sympy import symbols, Matrix, lambdify
        p = symbols('p0 p1 p2 p3')

        def f(x, _p): # fit Funktion
            return (_p[0] + _p[1] * 10 ** (_p[2] + _p[3] * x)) / (1 + 10 ** (_p[2] + _p[3] * x))

        lam_0 = np.array([100, 120, 3, -1], dtype=np.float64)
        tol = 1e-5


        g = Matrix([y[k] - f(x[k], p) for k in range(len(x))])
        Dg = g.jacobian(p)
        g = lambdify([p], g, 'numpy')
        Dg = lambdify([p], Dg, 'numpy')
        
        def err_func(lam):
            return np.linalg.norm(g(lam), 2) ** 2

        import scipy.optimize as opt

        scipy_lambdas = opt.fmin(err_func, lam_0, ftol=tol)

        expected_lambdas = np.array([163.88256553, 159.47427156, 2.17225694, -0.4293443])
        self.assertTrue(np.allclose(scipy_lambdas, expected_lambdas))


