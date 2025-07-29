#3x3 Matrix Multiplication Brent Equation Generator and CNF Converter

# Steps:
# 1. Generate Variables
# 2. Generate Equations
# 3. Convert Equations to CNF
# 4. Give CNF to SAT Solver
# 5. If a soltuion is found, decode it and print it

from pysat.solvers import Glucose3
import collections

def convert_brent_equation_to_cnf(equation_terms, final_value, var_manager):
    """Converts a single Brent equation into CNF clauses."""
    clauses = []
    
    # --- Step 1: Handle the 23 product terms (AND gates) ---
    product_vars = []
    for term in equation_terms:
        a, b, c = term
        #get new auxiliary variable for the product of a b and c 
        p = var_manager.new_aux_var()
        product_vars.append(p)
        #clauses for p<-> ( a AND b AND c)
        clauses.append([-p, a])
        clauses.append([-p, b])
        clauses.append([-p, c])
        clauses.append([p, -a, -b, -c])

    # --- Step 2: Handle the chain of XOR gates ---
    # Now we have: p1 XOR p2 XOR ... XOR p7
    if not product_vars: # Handle equations that equal 0 with no terms
        return []

    x_old = product_vars[0]
    if len(product_vars) > 1:
        for i in range(1, len(product_vars)):
            p_next = product_vars[i]
            x_new = var_manager.new_aux_var()
            
            # Clauses for x_new <-> (x_old XOR p_next)
            #x_old grows as the XOR chain progresses
            clauses.append([-x_new, x_old, p_next])
            clauses.append([-x_new, -x_old, -p_next])
            clauses.append([x_new, -x_old, p_next])
            clauses.append([x_new, x_old, -p_next])
            
            x_old = x_new
    
    final_xor_var = x_old
    
    # --- Step 3: Set the final value ---
    #If RHS = 1 set final_xor_var to true(positive), else false(negative)
    if final_value == 1:
        clauses.append([final_xor_var])
    else:
        clauses.append([-final_xor_var])
        
    return clauses

def decode_and_print_solution(model, var_manager):
    """Translates the SAT solver's integer model back into readable formulas."""
    
    # Create a set of positive variables for quick lookups
    true_vars = {v for v in model if v > 0}
    
    print("\n" + "="*20)
    print("✅ Solution Found! Decoding Recipe...")
    print("="*20 + "\n")

    # --- Part 1: Decode the 7 multiplication formulas (P1 to P7) ---
    print("--- The 7 Multiplications ---")
    products = collections.defaultdict(lambda: {"A": [], "B": []})
    
    for i in range(1, 4):
        for j in range(1, 4):
            for r in range(1, 24):
                # Check A terms
                a_var = var_manager.var_to_int.get(('a', r, i, j))
                if a_var in true_vars:
                    products[r]["A"].append(f"A_{i}{j}")
                # Check B terms
                b_var = var_manager.var_to_int.get(('b', r, i, j))
                if b_var in true_vars:
                    products[r]["B"].append(f"B_{i}{j}")

    for r in range(1, 24):
        a_terms = " + ".join(products[r]["A"])
        b_terms = " + ".join(products[r]["B"])
        print(f"P_{r} = ({a_terms}) * ({b_terms})")
        
    # --- Part 2: Decode how to combine products for the final matrix C ---
    print("\n--- Reconstructing the Final Matrix C ---")
    final_c = collections.defaultdict(list)
    
    for i in range(1, 4):
        for j in range(1, 4):
            for r in range(1, 24):
                c_var = var_manager.var_to_int.get(('c', r, i, j))
                if c_var in true_vars:
                    final_c[f"C_{i}{j}"].append(f"P_{r}")
    
    for c_idx in sorted(final_c.keys()):
        p_terms = " + ".join(final_c[c_idx])
        print(f"{c_idx} = {p_terms}")



class VariableMapper: # A class to map variable names to integers and back.
    def __init__(self):

        self.var_count = 0
        self.int_to_var = {}
        self.var_to_int = {}

    def get_var(self, name, r, i, j):
        #Get or create an integer for a variable
        var_tuple = (name, r, i, j)
        if var_tuple not in self.var_to_int:
            self.var_count += 1
            self.var_to_int[var_tuple] = self.var_count
            self.int_to_var[self.var_count] = var_tuple
        return self.var_to_int[var_tuple]

    def new_aux_var(self):
        """Get a new, temporary variable for the Tseitin transformation."""
        self.var_count += 1
        return self.var_count
    
if __name__ == "__main__":
        var_manager = VariableMapper()
        num_products = 7               
        all_clauses = []     #list to hold CNF clauses 

        print("Generating variables... ")
        for r in range(1, num_products + 1):                    
            for i in range(1, 4):
                for j in range(1, 4):
                    var_manager.get_var('a', r, i, j)
                    var_manager.get_var('b', r, i, j)
                    var_manager.get_var('c', r, i, j) 
        print(f" Mapped to integers 1-{var_manager.var_count}")

        print("Generating and Converting all 729 Brent equations...")
        num_equations = 0
        for i1 in range(1, 4):   
            for i2 in range(1, 4):  
                for j1 in range(1, 4):  
                    for j2 in range(1, 4):  
                        for k1 in range(1, 4):  
                            for k2 in range(1, 4):  
                            
                            #determining trhs using the kronecker delta
                                rhs = 1 if (i2 == j1 and i1 == k1 and j2 == k2) else 0
                            
                            # get the variables for this equation
                                equation_terms = []
                                for r in range(1, num_products + 1):
                                    a_var = var_manager.get_var('a', r, i1, i2)
                                    b_var = var_manager.get_var('b', r, j1, j2)
                                    c_var = var_manager.get_var('c', r, k1, k2)
                                    equation_terms.append((a_var, b_var, c_var))
                                num_equations +=1
                            #convert this equation to CNF
                                clauses = convert_brent_equation_to_cnf(equation_terms, rhs, var_manager)
                                all_clauses.extend(clauses)
        print(f"{len(all_clauses)} clauses generated")
        print(f"{num_equations} equations generated")

        print("\n3. Starting SAT solver...")
        with Glucose3(bootstrap_with=all_clauses) as solver:
            is_solvable = solver.solve()
        
        if is_solvable:
            print("Solution found")
            model = solver.get_model()
            # The model contains all variables, including auxiliary ones.
            # We only care about the original a, b, c variables for decoding.
            decode_and_print_solution(model, var_manager)
        else:
            print("\n No solution found")
        




