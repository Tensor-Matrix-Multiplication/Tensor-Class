import collections
from pysat.solvers import Glucose3
from itertools import product
import re

# --- Helper Functions (VariableMapper, convert_brent_equation_to_cnf) ---
# These are correct and remain unchanged from your original code.
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
    """Converts a single Brent equation into CNF clauses using XOR logic."""
    clauses = []
    # Create variables representing each term a*b*c in the equation
    product_vars = []
    for term in equation_terms:
        a, b, c = term
        p = var_manager.new_aux_var()
        product_vars.append(p)
        # Clause equivalent to: p <=> (a AND b AND c)
        clauses.extend([[-p, a], [-p, b], [-p, c], [p, -a, -b, -c]])

    if not product_vars:
        return []

    # Chain the XOR operations
    x_old = product_vars[0]
    if len(product_vars) > 1:
        for i in range(1, len(product_vars)):
            p_next = product_vars[i]
            x_new = var_manager.new_aux_var()
            # Clause equivalent to: x_new <=> (x_old XOR p_next)
            clauses.extend([
                [-x_new, x_old, p_next], [-x_new, -x_old, -p_next],
                [x_new, -x_old, p_next], [x_new, x_old, -p_next]
            ])
            x_old = x_new
    
    final_xor_var = x_old
    
    # Assert the final value of the XOR chain
    if final_value == 1:
        clauses.append([final_xor_var])
    else:
        clauses.append([-final_xor_var])
        
    return clauses

def add_laderman_assumptions(var_manager, matrix_size, num_products):
    """
    Encodes the 23-product Laderman equations as unit clauses (assumptions),
    but INTENTIONALLY SKIPS the 23rd multiplication to test the solver.
    """
    assumptions = []

    # Define the equations as strings
    m_defs_str = {
        1: "(a11 + a12 + a13 - a21 - a22 - a32 - a33) * b22",
        2: "(a11 - a21) * (b12 + b22)",
        3: "a22 * (-b11 + b12 + b21 - b22 - b23 + b31 - b33)",
        4: "(-a11 + a21 + a22) * (b11 - b12 + b22)",
        5: "(a21 + a22) * (-b11 + b12)",
        6: "a11 * b11",
        7: "(-a11 + a31 + a32) * (b11 - b13 + b23)",
        8: "(-a11 + a31) * (b13 - b23)",
        9: "(a31 + a32) * (-b11 + b13)",
        10: "(a11 + a12 + a13 - a22 - a23 - a31 - a32) * b23",
        11: "a32 * (-b11 + b13 + b21 - b22 - b23 - b31 + b32)",
        12: "(-a13 + a32 + a33) * (b22 + b31 - b32)",
        13: "(a13 - a33) * (b22 - b32)",
        14: "a13 * b31",
        15: "(a32 + a33) * (-b31 + b32)",
        16: "(-a13 + a22 + a23) * (b23 + b31 - b33)",
        17: "(-a13 + a23) * (b23 + b33)",
        18: "(a22 + a23) * (b31 - b33)",
        19: "a12 * b21",
        20: "a23 * b32",
        21: "a21 * b13",
        22: "a31 * b12",
        23: "a33 * b33" # This definition will be ignored by the logic below
    }

    c_defs_str = {
        "11": "m6 + m14 + m19",
        "12": "m1 + m4 + m5 + m6 + m12 + m14 + m15",
        "13": "m6 + m7 + m9 + m10 + m14 + m16 + m18",
        "21": "m2 + m3 + m4 + m6 + m14 + m16 + m17",
        "22": "m2 + m4 + m5 + m6 + m20",
        "23": "m14 + m16 + m17 + m18 + m21",
        "31": "m6 + m7 + m8 + m11 + m12 + m13 - m14",
        "32": "m12 + m13 + m14 + m15 + m22",
        "33": "m6 + m7 + m8 + m9 + m23"
    }

    # Helper to parse terms like "a12", "b31"
    def get_terms(s):
        return {term for term in re.findall(r'[ab]\d{2}', s)}

    # Encode M products (P in your code)
    for r in range(1, num_products + 1):
        # ✨ THIS IS THE MODIFICATION: Skip M_23's definition ✨
        if r == 23:
            continue

        a_str, b_str = m_defs_str[r].split('*')
        a_terms = get_terms(a_str)
        b_terms = get_terms(b_str)
        
        for i, j in product(range(1, matrix_size + 1), repeat=2):
            var_a = var_manager.get_var('a', r, i, j)
            if f'a{i}{j}' in a_terms:
                assumptions.append([var_a])
            else:
                assumptions.append([-var_a])
            var_b = var_manager.get_var('b', r, i, j)
            if f'b{i}{j}' in b_terms:
                assumptions.append([var_b])
            else:
                assumptions.append([-var_b])

    # Encode C matrix (NO CHANGES HERE)
    for i, j in product(range(1, matrix_size + 1), repeat=2):
        c_terms = {int(term) for term in re.findall(r'\d+', c_defs_str[f'{i}{j}'])}
        for r in range(1, num_products + 1):
            var_c = var_manager.get_var('c', r, i, j)
            if r in c_terms:
                assumptions.append([var_c])
            else:
                assumptions.append([-var_c])
                
    return assumptions


def decode_and_print_solution(model, var_manager, matrix_size, num_products):
    """
    Decodes the SAT model into a fully expanded and simplified algorithm.
    This function remains largely the same, as its logic is general.
    """
    true_vars = {v for v in model if v > 0}
    
    print("\n" + "="*40)
    print("✅ Solution Verified! Decoding Recipe...")
    print("="*40 + "\n")

    # --- Step 1 & 2: Identify and print Unique Products (P) ---
    raw_products = collections.defaultdict(lambda: {"A": [], "B": []})
    for i in range(1, matrix_size + 1):
        for j in range(1, matrix_size + 1):
            for r in range(1, num_products + 1):
                if var_manager.get_var('a', r, i, j) in true_vars:
                    raw_products[r]["A"].append(f"A_{i}{j}")
                if var_manager.get_var('b', r, i, j) in true_vars:
                    raw_products[r]["B"].append(f"B_{i}{j}")

    print(f"--- The {num_products} Multiplications (M) ---")
    for r in range(1, num_products + 1):
        a_str = " + ".join(sorted(raw_products[r]["A"])) if raw_products[r]["A"] else "0"
        b_str = " + ".join(sorted(raw_products[r]["B"])) if raw_products[r]["B"] else "0"
        print(f"M_{r} = ({a_str}) * ({b_str})")

    # --- Step 3: Decode and simplify the C_ij formulas in terms of Ms ---
    print("\n--- Reconstructing C in terms of M ---")
    c_formulas = {}
    for i in range(1, matrix_size + 1):
        for j in range(1, matrix_size + 1):
            c_idx = f"C_{i}{j}"
            terms = []
            for r in range(1, num_products + 1):
                if var_manager.get_var('c', r, i, j) in true_vars:
                    terms.append(f"M_{r}")
            
            c_formulas[c_idx] = terms
            m_str = " + ".join(sorted(terms, key=lambda t: int(t.split('_')[1]))) if terms else "0"
            print(f"{c_idx} = {m_str}")

    # --- Step 4: Fully expand and simplify C_ij formulas for verification ---
    print("\n--- Final Verification: C Matrix Fully Expanded and Simplified ---")
    for i in range(1, matrix_size + 1):
        for j in range(1, matrix_size + 1):
            c_idx = f"C_{i}{j}"
            m_terms_for_c = c_formulas[c_idx]
            
            all_expanded_terms = []
            for m_name in m_terms_for_c:
                r = int(m_name.split('_')[1])
                a_terms = raw_products[r]["A"]
                b_terms = raw_products[r]["B"]
                # Create the cross-product
                for a_term in a_terms:
                    for b_term in b_terms:
                        all_expanded_terms.append(f"{a_term}*{b_term}")
            
            # Count the final expanded terms and cancel pairs (XOR logic)
            final_term_counts = collections.Counter(all_expanded_terms)
            final_simplified_terms = sorted(
                [term for term, count in final_term_counts.items() if count % 2 != 0]
            )
            
            result_str = " + ".join(final_simplified_terms) if final_simplified_terms else "0"
            print(f"{c_idx} = {result_str}")


# --- Main Execution ---
if __name__ == "__main__":
    MATRIX_SIZE = 3
    NUM_PRODUCTS = 23
    
    var_manager = VariableMapper()
    all_clauses = []

    print(f"--- Verifying {MATRIX_SIZE}x{MATRIX_SIZE} matrix multiplication with {NUM_PRODUCTS} products ---")

    print("1. Generating variables...")
    for r, i, j in product(range(1, NUM_PRODUCTS + 1), range(1, MATRIX_SIZE + 1), range(1, MATRIX_SIZE + 1)):
        var_manager.get_var('a', r, i, j)
        var_manager.get_var('b', r, i, j)
        var_manager.get_var('c', r, i, j)
    print(f"   ...done. Mapped {var_manager.var_count} variables.")

    print(f"\n2. Generating and converting all {MATRIX_SIZE**6} Brent equations to CNF...")
    indices = product(range(1, MATRIX_SIZE + 1), repeat=6)
    # The paper uses a transposed version C^T = AB, we use C = AB
    # Standard definition: C_ik = sum(j=1 to n) A_ij * B_jk
    # Brent equations for C=AB: sum(r=1 to m) alpha_ij^r * beta_jk^r * gamma_ik^r = 1 for all i,j,k
    # In our notation (i1,i2, j1,j2, k1,k2): C_k1k2 = sum A_k1j*B_jk2
    # So, need to check for delta_(i1,k1) * delta_(j2,k2) * delta_(i2,j1)
    for i1, i2, j1, j2, k1, k2 in indices:
        rhs = 1 if (i1 == k1 and j2 == k2 and i2 == j1) else 0
        equation_terms = [(var_manager.get_var('a', r, i1, i2), var_manager.get_var('b', r, j1, j2), var_manager.get_var('c', r, k1, k2)) for r in range(1, NUM_PRODUCTS + 1)]
        clauses = convert_brent_equation_to_cnf(equation_terms, rhs, var_manager)
        all_clauses.extend(clauses)
    print(f"   ...done. Generated a total of {len(all_clauses)} clauses.")
    
    print("\n3. Adding Laderman solution as assumptions...")
    assumptions = add_laderman_assumptions(var_manager, MATRIX_SIZE, NUM_PRODUCTS-5)
    all_clauses.extend(assumptions)
    print(f"   ...done. Added {len(assumptions)} unit clauses.")

    print("\n4. Starting SAT solver...")
    with Glucose3(bootstrap_with=all_clauses) as solver:
        is_solvable = solver.solve()
        print(f"   ...solver finished. Is a solution found? {is_solvable}")
        
        if is_solvable:
            model = solver.get_model()
            decode_and_print_solution(model, var_manager, MATRIX_SIZE, NUM_PRODUCTS)
        else:
            print("\n❌ No solution found. The model is UNSAT, which indicates an issue in the problem encoding or the provided equations.")