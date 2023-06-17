import sympy as sym

sym.init_printing()

pprint = sym.pprint


x = sym.Symbol('x')


"""
Ableitung simpel f1(x) = x^2 + 1
"""

f1 = x**2 + 1

pprint(f1)

df1 = sym.diff(f1, x)

pprint(df1)

callable_f1 = sym.lambdify(x, f1)
callable_df1 = sym.lambdify(x, df1)

print(callable_f1(3))
print(callable_df1(3))

f2 = sym.log(x) * sym.sqrt(x)

"""
Ableitung mit Funktionen: f2(x) = log(x) * sqrt(x)
"""

pprint(f2)

df2 = sym.diff(f2, x)

callable_df2 = sym.lambdify(x, df2)

pprint(df2)
