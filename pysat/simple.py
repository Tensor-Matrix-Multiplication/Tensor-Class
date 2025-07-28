from pysat.formula import CNF
from pysat.solvers import Glucose3

cnf = CNF()

cnf.append([1, -2, 3])  # C1: x1 ∨ ¬x2 ∨ x3
cnf.append([-1, 2])     # C2: ¬x1 ∨ x2
cnf.append([2, -4])     # C3: x2 ∨ ¬x4
cnf.append([-3, 4])     # C4: ¬x3 ∨ x4
cnf.append([4])         # C5: x4


solver = Glucose3()
solver.append_formula(cnf)

if solver.solve():
    print("✅ Satisfiable!")
    print("Model (variable assignment):")
    model = solver.get_model()
    for var in model:
        var_name = f"x{abs(var)}"
        value = var > 0
        print(f"{var_name} = {value}")
else:
    print("❌ UNSAT: No satisfying assignment exists.")
