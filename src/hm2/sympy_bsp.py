import sympy as sp

def impliziter_plot_bsp():
    '''
    Aus Serie 3 Aufgabe 2 a)
    Erstellt einen implizten Plot, nützlich für Kreisförmige/Hyperbel Kurven
    '''
    x, y = sp.symbols('x y')
    f1 = x ** 2 / 186 ** 2 - y ** 2 / (300 ** 2 - 186 ** 2) - 1
    f2 = (y - 500) ** 2 / 279 ** 2 - (x - 300) ** 2 / (500 ** 2 - 279 ** 2) - 1
    p1 = sp.plot_implicit(sp.Eq(f1, 0), (x, -2000, 2000), (y, -2000, 2000))
    p2 = sp.plot_implicit(sp.Eq(f2, 0), (x, -2000, 2000), (y, -2000, 2000))
    p1.append(p2[0])
    p1.show()

