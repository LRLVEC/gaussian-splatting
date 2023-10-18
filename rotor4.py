from sympy import symbols, pprint, print_latex
from galgebra.ga import Ga
from galgebra.printer import Format

# Format(Fmode=False, Dmode=True)

s4coords = (x, y, z, w) = symbols('0 1 2 3', real=True)
s4 = Ga('e', g=[1, 1, 1, 1], coords=s4coords)

a = s4.mv('a', 'vector')

s = s4.mv('s', 'scalar')
b = s4.mv('b', 'bivector')
p = s4.mv('p', 'pseudo')
rotor = s + b + p

# pprint(rotor * a)
# print_latex(rotor * a)

pprint(rotor * rotor.rev())
print_latex(rotor * rotor.rev())

# pprint(rotor * a * rotor.rev())
# print_latex(rotor * a * rotor.rev())