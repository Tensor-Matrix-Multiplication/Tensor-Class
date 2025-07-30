import collections
from pysat.solvers import Glucose3

# --- Helper Functions (mostly unchanged) ---

def convert_brent_equation_to_cnf(equation_terms, final_value, var_manager):
    """Converts a single Brent equation into CNF clauses."""
    clauses = []
    
    # Handle the product terms (AND gates)
    product_vars = []
    for term in equation_terms:
        a, b, c = term
        p = var_manager.new_aux_var()
        product_vars.append(p)
        clauses.extend([[-p, a], [-p, b], [-p, c], [p, -a, -b, -c]])

    if not product_vars:
        # This can happen if the equation is just 0=0
        return []

    # Handle the chain of XOR gates
    x_old = product_vars[0]
    if len(product_vars) > 1:
        for i in range(1, len(product_vars)):
            p_next = product_vars[i]
            x_new = var_manager.new_aux_var()
            clauses.extend([
                [-x_new, x_old, p_next], [-x_new, -x_old, -p_next],
                [x_new, -x_old, p_next], [x_new, x_old, -p_next]
            ])
            x_old = x_new
    
    final_xor_var = x_old
    
    # Set the final value
    if final_value == 1:
        clauses.append([final_xor_var])
    else:
        clauses.append([-final_xor_var])
        
    return clauses

class VariableMapper:
    """A class to map variable names to integers and back."""
    def __init__(self):
        self.var_count = 0
        self.int_to_var = {}
        self.var_to_int = {}

    def get_var(self, name, r, i, j):
        """Get or create an integer for a variable like a_ij^r."""
        var_tuple = (name, r, i, j)
        if var_tuple not in self.var_to_int:
            self.var_count += 1
            self.var_to_int[var_tuple] = self.var_count
            self.int_to_var[self.var_count] = var_tuple
        return self.var_to_int[var_tuple]

    def new_aux_var(self):
        """Get a new, temporary variable."""
        self.var_count += 1
        return self.var_count

def decode_and_print_solution(model, var_manager, matrix_size, num_products):
    """Translates the SAT solver's model back into readable formulas."""
    true_vars = {v for v in model if v > 0}
    
    print("\n" + "="*20)
    print("✅ Solution Found! Decoding Recipe...")
    print("="*20 + "\n")

    # Decode the multiplication formulas (P_r)
    print(f"--- The {num_products} Multiplications ---")
    products = collections.defaultdict(lambda: {"A": [], "B": []})
    
    for i in range(1, matrix_size + 1):
        for j in range(1, matrix_size + 1):
            for r in range(1, num_products + 1):
                if var_manager.var_to_int.get(('a', r, i, j)) in true_vars:
                    products[r]["A"].append(f"A_{i}{j}")
                if var_manager.var_to_int.get(('b', r, i, j)) in true_vars:
                    products[r]["B"].append(f"B_{i}{j}")

    for r in range(1, num_products + 1):
        a_terms = " + ".join(products[r]["A"]) if products[r]["A"] else "0"
        b_terms = " + ".join(products[r]["B"]) if products[r]["B"] else "0"
        print(f"P_{r} = ({a_terms}) * ({b_terms})")
        
    # Decode the final matrix reconstruction
    print("\n--- Reconstructing the Final Matrix C ---")
    final_c = collections.defaultdict(list)
    
    for i in range(1, matrix_size + 1):
        for j in range(1, matrix_size + 1):
            for r in range(1, num_products + 1):
                if var_manager.var_to_int.get(('c', r, i, j)) in true_vars:
                    final_c[f"C_{i}{j}"].append(f"P_{r}")
    
    for i in range(1, matrix_size + 1):
        for j in range(1, matrix_size + 1):
            c_idx = f"C_{i}{j}"
            p_terms = " + ".join(final_c[c_idx]) if final_c[c_idx] else "0"
            print(f"{c_idx} = {p_terms}")

# --- Main Execution ---
if __name__ == "__main__":
    # --- Parameters for the problem ---
    MATRIX_SIZE = 2
    NUM_PRODUCTS = 7
    
    var_manager = VariableMapper()
    all_clauses = []

    print(f"--- Finding {MATRIX_SIZE}x{MATRIX_SIZE} matrix multiplication with {NUM_PRODUCTS} products ---")

    print("1. Generating variables...")
    num_main_vars = 3 * (MATRIX_SIZE**2) * NUM_PRODUCTS
    print(f"   (Expecting {num_main_vars} main variables for a, b, c)")
    for r in range(1, NUM_PRODUCTS + 1):
        for i in range(1, MATRIX_SIZE + 1):
            for j in range(1, MATRIX_SIZE + 1):
                var_manager.get_var('a', r, i, j)
                var_manager.get_var('b', r, i, j)
                var_manager.get_var('c', r, i, j)
    print(f"   ...done. Mapped to integers 1-{var_manager.var_count}")

    print(f"\n2. Generating and converting all {MATRIX_SIZE**6} Brent equations to CNF...")
    # Loop over all combinations of indices for the Brent equations
    # This is equivalent to six nested loops from 1 to MATRIX_SIZE
    from itertools import product
    indices = product(range(1, MATRIX_SIZE + 1), repeat=6)

    for i1, i2, j1, j2, k1, k2 in indices:
        rhs = 1 if (i2 == j1 and i1 == k1 and j2 == k2) else 0
        
        equation_terms = []
        for r in range(1, NUM_PRODUCTS + 1):
            a_var = var_manager.get_var('a', r, i1, i2)
            b_var = var_manager.get_var('b', r, j1, j2)
            c_var = var_manager.get_var('c', r, k1, k2)
            equation_terms.append((a_var, b_var, c_var))
        
        clauses = convert_brent_equation_to_cnf(equation_terms, rhs, var_manager)
        all_clauses.extend(clauses)

    print(f"   ...done. Generated a total of {len(all_clauses)} clauses.")

    print("\n3. Starting SAT solver (this will likely take a very long time)...")
    with Glucose3(bootstrap_with=all_clauses) as solver:
        is_solvable = solver.solve()
        print(f"   ...solver finished. Is a solution found? {is_solvable}")
        
        if is_solvable:
            model = solver.get_model()
            decode_and_print_solution(model, var_manager, MATRIX_SIZE, NUM_PRODUCTS)
        else:
            print("\n❌ No solution found. The model is UNSAT.")