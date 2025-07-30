import collections
from pysat.solvers import Glucose3
from itertools import product

# --- Helper Functions (VariableMapper, convert_brent_equation_to_cnf) ---
# These are correct and remain unchanged.
class VariableMapper:
    """A class to map variable names to integers and back."""
    def __init__(self):
        self.var_count = 0
        self.int_to_var = {}
        self.var_to_int = {}

    def get_var(self, name, r, i, j):
        var_tuple = (name, r, i, j)
        if var_tuple not in self.var_to_int:
            self.var_count += 1
            self.var_to_int[var_tuple] = self.var_count
            self.int_to_var[self.var_count] = var_tuple
        return self.var_to_int[var_tuple]

    def new_aux_var(self):
        self.var_count += 1
        return self.var_count

def convert_brent_equation_to_cnf(equation_terms, final_value, var_manager):
    """Converts a single Brent equation into CNF clauses."""
    clauses = []
    product_vars = []
    for term in equation_terms:
        a, b, c = term
        p = var_manager.new_aux_var()
        product_vars.append(p)
        clauses.extend([[-p, a], [-p, b], [-p, c], [p, -a, -b, -c]])

    if not product_vars:
        return []

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
    
    if final_value == 1:
        clauses.append([final_xor_var])
    else:
        clauses.append([-final_xor_var])
        
    return clauses

# --- ✨ NEW: Final Decoding Function with Full Expansion Logic ---
def decode_and_print_solution(model, var_manager, matrix_size, num_products):
    """
    Decodes the SAT model into a fully expanded and simplified algorithm.
    """
    true_vars = {v for v in model if v > 0}
    
    print("\n" + "="*40)
    print("✅ Solution Found! Decoding and Simplifying Recipe...")
    print("="*40 + "\n")

    # --- Step 1 & 2: Identify and print Unique Products (P) ---
    raw_products = collections.defaultdict(lambda: {"A": [], "B": []})
    for i in range(1, matrix_size + 1):
        for j in range(1, matrix_size + 1):
            for r in range(1, num_products + 1):
                if var_manager.var_to_int.get(('a', r, i, j)) in true_vars:
                    raw_products[r]["A"].append(f"A_{i}{j}")
                if var_manager.var_to_int.get(('b', r, i, j)) in true_vars:
                    raw_products[r]["B"].append(f"B_{i}{j}")

    unique_product_defs = {}
    product_remapping = {}
    
    for r in range(1, num_products + 1):
        a_terms = tuple(sorted(raw_products[r]["A"]))
        b_terms = tuple(sorted(raw_products[r]["B"]))
        canonical_form = (a_terms, b_terms)

        if canonical_form not in unique_product_defs:
            new_name = f"P_{len(unique_product_defs) + 1}"
            unique_product_defs[canonical_form] = {"name": new_name, "def": canonical_form}
        
        product_remapping[f"P_{r}"] = unique_product_defs[canonical_form]["name"]

    print(f"--- The {len(unique_product_defs)} Unique Multiplications (P) ---")
    sorted_ups = sorted(unique_product_defs.values(), key=lambda item: int(item["name"].split('_')[1]))
    for up in sorted_ups:
        (a_terms, b_terms) = up["def"]
        a_str = " + ".join(a_terms) if a_terms else "0"
        b_str = " + ".join(b_terms) if b_terms else "0"
        print(f"{up['name']} = ({a_str}) * ({b_str})")

    # --- Step 3: Decode and simplify the C_ij formulas in terms of Ps ---
    print("\n--- Reconstructing C in terms of Unique Products (Simplified) ---")
    c_formulas_in_up = {}
    for i in range(1, matrix_size + 1):
        for j in range(1, matrix_size + 1):
            c_idx = f"C_{i}{j}"
            remapped_terms = []
            for r in range(1, num_products + 1):
                if var_manager.var_to_int.get(('c', r, i, j)) in true_vars:
                    remapped_terms.append(product_remapping[f"P_{r}"])
            
            term_counts = collections.Counter(remapped_terms)
            final_terms = sorted(
                [term for term, count in term_counts.items() if count % 2 != 0],
                key=lambda item: int(item.split('_')[1])
            )
            c_formulas_in_up[c_idx] = final_terms
            p_str = " + ".join(final_terms) if final_terms else "0"
            print(f"{c_idx} = {p_str}")

    # --- ✨ Step 4: Fully expand and simplify C_ij formulas ---
    print("\n--- Final Verification: C Matrix Fully Expanded and Simplified ---")
    # Create a reverse map from P_name to its definition
    up_defs_by_name = {up["name"]: up["def"] for up in unique_product_defs.values()}
    
    for i in range(1, matrix_size + 1):
        for j in range(1, matrix_size + 1):
            c_idx = f"C_{i}{j}"
            
            # Get the list of P terms for this C element
            up_terms_for_c = c_formulas_in_up[c_idx]
            
            # Expand every P term into its final A*B products
            all_expanded_terms = []
            for up_name in up_terms_for_c:
                a_terms, b_terms = up_defs_by_name[up_name]
                # Create the cross-product
                for a_term in a_terms:
                    for b_term in b_terms:
                        all_expanded_terms.append(f"{a_term}*{b_term}")
            
            # Count the final expanded terms and cancel pairs
            final_term_counts = collections.Counter(all_expanded_terms)
            final_simplified_terms = sorted(
                [term for term, count in final_term_counts.items() if count % 2 != 0]
            )
            
            result_str = " + ".join(final_simplified_terms) if final_simplified_terms else "0"
            print(f"{c_idx} = {result_str}")

# --- Main Execution ---
if __name__ == "__main__":
    MATRIX_SIZE = 2
    NUM_PRODUCTS = 7
    
    var_manager = VariableMapper()
    all_clauses = []

    print(f"--- Finding {MATRIX_SIZE}x{MATRIX_SIZE} matrix multiplication with {NUM_PRODUCTS} products ---")

    print("1. Generating variables...")
    for r in range(1, NUM_PRODUCTS + 1):
        for i in range(1, MATRIX_SIZE + 1):
            for j in range(1, MATRIX_SIZE + 1):
                var_manager.get_var('a', r, i, j); var_manager.get_var('b', r, i, j); var_manager.get_var('c', r, i, j)
    print(f"   ...done. Mapped {var_manager.var_count} variables.")

    print(f"\n2. Generating and converting all {MATRIX_SIZE**6} Brent equations to CNF...")
    indices = product(range(1, MATRIX_SIZE + 1), repeat=6)
    for i1, i2, j1, j2, k1, k2 in indices:
        rhs = 1 if (i2 == j1 and i1 == k1 and j2 == k2) else 0
        equation_terms = [(var_manager.get_var('a', r, i1, i2), var_manager.get_var('b', r, j1, j2), var_manager.get_var('c', r, k1, k2)) for r in range(1, NUM_PRODUCTS + 1)]
        clauses = convert_brent_equation_to_cnf(equation_terms, rhs, var_manager)
        all_clauses.extend(clauses)
    print(f"   ...done. Generated a total of {len(all_clauses)} clauses.")

    print("\n3. Starting SAT solver...")
    with Glucose3(bootstrap_with=all_clauses) as solver:
        is_solvable = solver.solve()
        print(f"   ...solver finished. Is a solution found? {is_solvable}")
        
        if is_solvable:
            model = solver.get_model()
            decode_and_print_solution(model, var_manager, MATRIX_SIZE, NUM_PRODUCTS)
        else:
            print("\n❌ No solution found. The model is UNSAT.")