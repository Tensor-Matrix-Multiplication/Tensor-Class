def generate_brent_2x2_equations():
    # Each multiplication step r contributes one term
    num_products = 7

    equations = []
    rhs_counter = 0

    # Loop over all combinations of indices
    for i1 in range(1, 3):   # A row
        for i2 in range(1, 3):  # A col
            for j1 in range(1, 3):  # B row
                for j2 in range(1, 3):  # B col
                    for k1 in range(1, 3):  # C row
                        for k2 in range(1, 3):  # C col
                            # Each equation is a linear combination over 7 products
                            terms = []
                            for r in range(1, num_products + 1):
                                terms.append(f"a_{i1}{i2}^{r} * b_{j1}{j2}^{r} * c_{k1}{k2}^{r}")
                            
                            lhs = " + ".join(terms)
                            
                            # This is the key delta condition
                            rhs = 1 if (i2 == j1 and i1 == k1 and j2 == k2) else 0
                            if rhs == 1:
                                rhs_counter += 1
                            
                            equations.append((lhs, rhs))
    return equations, rhs_counter

def generate_brent_2x2_equations_pretty(pretty=True):
    num_products = 7
    equations = []
    rhs_counter = 0

    for i1 in range(1, 3):   # A row
        for i2 in range(1, 3):  # A col
            for j1 in range(1, 3):  # B row
                for j2 in range(1, 3):  # B col
                    for k1 in range(1, 3):  # C row
                        for k2 in range(1, 3):  # C col
                            # Check contraction match
                            rhs = 1 if (i2 == j1 and i1 == k1 and j2 == k2) else 0
                            if rhs == 1:
                                rhs_counter += 1
                                terms = [f"a_{i1}{i2}^{r} * b_{j1}{j2}^{r} * c_{k1}{k2}^{r}" for r in range(1, num_products+1)]
                                equations.append(((i1, i2), (j1, j2), (k1, k2), terms, rhs))

    # Print nicely
    print("\n--- Brent Equations for 2x2 Matrix Multiplication Using 7 Products ---")
    for (ai, aj), (bi, bj), (ci, cj), terms, rhs in equations:
        print(f"\nEquation for A_{ai}{aj} * B_{bi}{bj} * C_{ci}{cj}:")
        for term in terms:
            print(f"  {term}")
        print(f"= {rhs}")

    print(f"\nTotal RHS = 1 Equations: {rhs_counter}")
    return equations, rhs_counter

if __name__ == "__main__":
    # Run and print
    equations, rhs_count = generate_brent_2x2_equations()

    print("Brent equations for 2x2 matrix multiplication with 7 products:\n")
    for eq in equations:
        print(f"{eq[0]} = {eq[1]}")

    print("\nTotal RHS = 1 constraints:", rhs_count)

    generate_brent_2x2_equations()