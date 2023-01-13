from typing import Callable
import numpy as np

def newton_step(x_n: float, f: Callable[[float], float], df: Callable[[float], float]) -> float:
    """
    Führt einen Schritt der Newton Iteration durch

    Parameters:
        x_n: momentaner x Wert
        f: Funktion deren Nullstelle gesucht wird
        df: Ableitung dieser Funktion
    Returns
        x_(n+1): der nächste Wert der Newton-Iteration
    """
    return x_n - f(x_n) / df(x_n)

def newton_iter(f: Callable[[float], float], df: Callable[[float], float], x0: float, tol: float = 1e-7) \
        -> float:
    """
    Newton-Iteration einer Funktion f zum Startwert x0

    Parameters:
        f: Funktion deren Nullstelle gesucht wird
        df: Ableitung dieser Funktion
        x0: Startwert für die Iteration
        tol: Toleranz für die Genauigkeit der Lösung
    """
    x_curr = x0

    while f(x_curr - tol) * f(x_curr + tol) > 0:
        x_curr = newton_step(x_curr, f, df)

    return x_curr

def do_banach(interval: tuple[float, float], F: Callable[[float], float], dF: Callable[[float], float]) -> tuple[bool, float | None]:
    """
    **!ACHTUNG: dieses Resultat stimmt nur wenn F stetig ist im Interval!**
    Sucht eigenständig nach dem Max/Min von F&dF im interval und überprüft das Resultat mit #check_banach
    Benötigt eine stetige Fixpunktgleichung F(x_k) = x_{k+1} im Interval
    NICHT funktion f(x) = y
    Parameters:
        interval: Grenzen des Funktionsinterval (min, max)
        F: referenz zur !STETIGEN! Fixpunktgleichung F(x_k) = x_{k+1}
        dF: referenz zur Ableitung von F
    Returns:
        bool: Ob Banachscherfixpunktsatz erfüllt ist (==check_banach)
        float | None: Lipschitzkonstante / None falls Banach nicht erfüllt
    """
    (a, b) = interval
    F_extrema = F(a), F(b)
    F_min = min(F_extrema)
    F_max = max(F_extrema)

    dF_extrema = abs(dF(a)), abs(dF(b))
    dF_max = max(dF_extrema)

    banach_ok = check_banach(interval=(a,b), min_F=F_min, max_F=F_max, max_Fdx=dF_max)

    return (True, dF_max) if banach_ok else (False, None)


def check_banach(interval: tuple[float, float], min_F: float, max_F: float, max_Fdx: float) -> bool:
    """
    Siehe auch: #do_banach
    Benötigt eine stetige Fixpunktgleichung F(x_k) = x_{k+1} im Interval
    NICHT funktion f(x) = y
    Parameters:
        interval: Grenzen des Funktionsinterval (min, max)
        min_F: Minimum der Fixpunktgleichung F
        max_F: Maximum der Fixpunktgleichung F
        max_Fdx: Maximum der Ableitung der Fixpunktgleichung F
                    Entspricht der Lipschitzkonstante
    Returns:
        bool: Ob Banachscherfixpunktsatz erfüllt ist

    """
    (a, b) = interval
    lipschitz = max_Fdx

    return (0 < lipschitz and lipschitz < 1) \
        and (max_F < b) \
        and (min_F > a)


def fixpunkt_iter(F: Callable[[float], float], x0: float, tol: float, alpha: float) -> tuple[float, int]:
    """
    Fixpunktiteration für eine gegebene Fixpunktgleichung F(x_k) = x_(k+1)

    Parameters:
        F: referenz zur Fixpunktgleichung
        x0: Startwert der Iteration
        tolerance: Fehlertoleranz (für das Abbruchkriterium)
        alpha: Nach Definition von a-priori/posteriori (für das Abbruchkriterium)
    
    Returns:
        x_k: Annäherung an den Fixpunkt / NaN falls divergent nach 10_000 iterationen 
        k: Anzahl der Iterationen
    """
    k = 0
    not_converged = True
    N = 10_000
    x_curr: float = x0

    x_next = np.nan
    while (not_converged and k < N):
        x_next = F(x_curr)
        incr = abs(x_next-x_curr)
        error = alpha/(1-alpha)*incr
        not_converged = error > tol
        k = k+1
        x_curr = x_next

    if not_converged:
        raise Exception(f'not converged after {N} iterations')
    
    return (x_next, k)



import unittest


class FixpunktTest(unittest.TestCase):
    def test_HS2020_A3a(self):
        def f(x):
            return np.exp(x) - np.sqrt(x) - 2

        def df(x):
            return np.exp(x) - (0.5 * 1 / np.sqrt(x))

        nst = newton_iter(f, df, 0.5, 1e-7)

        self.assertAlmostEqual(nst, 1.1174679154114777)

    def test_FS2020_A3a(self):
        def F(x): 
            return 1./(np.cos(x+np.pi/4)-1)+2

        lipschitz = 0.6640946355544965

        fixpkt, num_iter = fixpunkt_iter(F, x0=1, tol=1e-6, alpha=lipschitz)

        self.assertAlmostEqual(fixpkt, 1.34776506)
        self.assertEqual(num_iter, 15)
        
    
    def test_FS2020_A3b(self):
        def F(x): 
            return 1./(np.cos(x+np.pi/4)-1)+2
        def dF(x) : 
            return np.sin(x + np.pi/4)/(np.cos(x + np.pi/4) - 1)**2 
        
        banach_ok, lipschitz = do_banach((1,2), F, dF)

        self.assertTrue(banach_ok)
        self.assertIsNotNone(lipschitz)
        self.assertAlmostEqual(lipschitz, 0.6640946355544965) # type: ignore


if __name__ == '__main__':
    unittest.main()