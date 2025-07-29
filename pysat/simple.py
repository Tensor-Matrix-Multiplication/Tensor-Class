from pysat.solvers import Glucose3

def convert_brent_equation_to_cnf(equation_terms, final_value):
    """
    Converts a single Brent equation into CNF clauses using Tseitin transformation.

    Args:
        equation_terms (list of tuples): Each tuple represents a product term,
                                        containing the integer mapping of its variables.
                                        e.g., [ (1, 2, 3), (4, 5, 6), ... ]
        final_value (int): The right-hand side of the equation (0 or 1).

    Returns:
        list of lists: The CNF clauses for the SAT solver.
    """
    clauses = []
    # We need a way to create new, unique variables for the transformation.
    # We start counting from the highest variable number used in the equation.
    max_var = max(v for term in equation_terms for v in term)
    next_aux_var = max_var + 1

    # --- Step 1: Handle the 7 product terms (AND gates) ---
    product_vars = []
    for term in equation_terms:
        a, b, c = term
        p = next_aux_var  # This is the auxiliary variable for the product
        product_vars.append(p)
        next_aux_var += 1

        # Tseitin transformation for p <-> (a AND b AND c)
        # This translates to 4 clauses:
        # 1. (~p OR a)
        # 2. (~p OR b)
        # 3. (~p OR c)
        # 4. (p OR ~a OR ~b OR ~c)
        clauses.append([-p, a])
        clauses.append([-p, b])
        clauses.append([-p, c])
        clauses.append([p, -a, -b, -c])

    # --- Step 2: Handle the chain of XOR gates ---
    # Now we have: p1 XOR p2 XOR p3 XOR p4 XOR p5 XOR p6 XOR p7
    
    # First XOR: x1 <-> (p1 XOR p2)
    p1 = product_vars[0]
    p2 = product_vars[1]
    x1 = next_aux_var
    next_aux_var += 1
    
    # Tseitin transformation for x1 <-> (p1 XOR p2)
    # (~x1 | p1 | p2) & (~x1 | ~p1 | ~p2) & (x1 | ~p1 | p2) & (x1 | p1 | ~p2)
    clauses.append([-x1, p1, p2])
    clauses.append([-x1, -p1, -p2])
    clauses.append([x1, -p1, p2])
    clauses.append([x1, p1, -p2])

    # Chain the rest of the XORs
    # x_new <-> (x_old XOR p_next)
    x_old = x1
    for i in range(2, len(product_vars)):
        p_next = product_vars[i]
        x_new = next_aux_var
        next_aux_var += 1
        
        # Clauses for x_new <-> (x_old XOR p_next)
        clauses.append([-x_new, x_old, p_next])
        clauses.append([-x_new, -x_old, -p_next])
        clauses.append([x_new, -x_old, p_next])
        clauses.append([x_new, x_old, -p_next])
        
        x_old = x_new
    
    # The final variable in our chain represents the result of the entire LHS
    final_xor_var = x_old
    
    # --- Step 3: Set the final value ---
    # The equation must equal `final_value` (which is 1 in our example).
    if final_value == 1:
        # Add clause (final_xor_var) to force it to be True
        clauses.append([final_xor_var])
    else:
        # Add clause (~final_xor_var) to force it to be False
        clauses.append([-final_xor_var])
        
    return clauses

# --- Let's run it on your example equation ---
if __name__ == "__main__":
    # First, let's create a mapping from variable names to integers.
    # For this single equation, we have 7 terms, each with 3 variables.
    # Total of 21 unique variables.
    # a_22^1 -> 1, b_22^1 -> 2, c_22^1 -> 3
    # a_22^2 -> 4, b_22^2 -> 5, c_22^2 -> 6
    # ... and so on.
    
    var_counter = 1
    brent_eq_terms = []
    for r in range(7):
        brent_eq_terms.append((var_counter, var_counter + 1, var_counter + 2))
        var_counter += 3
    
    print("Variable mapping for our single equation:")
    print(brent_eq_terms)
    print("-" * 20)

    # Convert our single equation to CNF
    final_clauses = convert_brent_equation_to_cnf(brent_eq_terms, 1)

    print(f"Generated {len(final_clauses)} clauses.")
    print("Clauses:", final_clauses) # Uncomment to see all clauses
    
    # --- Use PySAT to solve ---
    with Glucose3(bootstrap_with=final_clauses) as solver:
        is_solvable = solver.solve()
        print(f"\nIs the single equation solvable? {is_solvable}")
        
        if is_solvable:
            model = solver.get_model()
            print("\nA possible solution (a satisfying assignment) for the variables:")
            print(model)