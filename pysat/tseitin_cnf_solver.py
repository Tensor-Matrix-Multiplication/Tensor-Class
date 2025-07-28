from pysat.formula import CNF, IDPool
from pysat.solvers import Glucose3

# Create variable pool and CNF formula
vpool = IDPool()
cnf = CNF()

def tseitin_not(x, y, cnf):
    """Encode y <-> ¬x"""
    cnf.append([-y, -x])
    cnf.append([y, x])

def tseitin_and(x, y, z, cnf):
    """Encode z <-> (x ∧ y)"""
    cnf.append([-z, x])
    cnf.append([-z, y])
    cnf.append([z, -x, -y])

def tseitin_or(x, y, z, cnf):
    """Encode z <-> (x ∨ y)"""
    cnf.append([-z, x, y])
    cnf.append([z, -x])
    cnf.append([z, -y])

def tseitin_implies(x, y, z, cnf):
    """Encode z <-> (x ⇒ y)"""
    cnf.append([-z, -x, y])
    cnf.append([z, x])
    cnf.append([z, -y])

def tseitin_iff(x, y, z, cnf):
    """Encode z <-> (x ↔ y)"""
    cnf.append([-z, -x, y])
    cnf.append([-z, x, -y])
    cnf.append([z, x, y])
    cnf.append([z, -x, -y])

# Input vars: x1 = 1, x2 = 2, x3 = 3, x4 = 4
x1 = vpool.id("x1")
x2 = vpool.id("x2")
x3 = vpool.id("x3")
x4 = vpool.id("x4")
x5 = vpool.id("x5")

# Subformula variables
y1 = vpool.id("y1")    # y1 = NOT x2
y2 = vpool.id("y2")    # y2 = x1 ^ y1
y3 = vpool.id("y3")    # y3 = y2 V x3
y4 = vpool.id("y4")    # y4 = NOT x5
y5 = vpool.id("y5")    # y5 = x4 <-> y4
y6 = vpool.id("y6")    # y6 = y3 -> y5
phi = vpool.id("phi")  # final output = y6

# CNF clauses for y1 = NOT x2
tseitin_not(x2, y1, cnf)

# CNF for y2 = x1 ^ y1
tseitin_and(x1, y1, y2, cnf)

# CNF for y3 = y2 V x3
tseitin_or(y2, x3, y3, cnf)

# CNF for y4 = NOT x5
tseitin_not(x5, y4, cnf)

# CNF for y5 = x4 <-> y4
tseitin_iff(x4, y4, y5, cnf)

# CNF for y6 = y3 ⇒ y5 = ¬y3 ∨ y5
tseitin_implies(y3,y5,y6, cnf)

# CNF for phi = y6
cnf.append([-phi, y6])
cnf.append([phi, -y6])

# Final assertion: φ = True
cnf.append([phi])

# Solve
solver = Glucose3()
solver.append_formula(cnf)

if solver.solve():
    model = solver.get_model()
    print("✅ SATISFIABLE!\n")
    print("Model:")
    for var in sorted(model, key=abs):
        name = vpool.obj(abs(var))
        print(f"  {name} = {var > 0}")
else:
    print("❌ UNSATISFIABLE.")
