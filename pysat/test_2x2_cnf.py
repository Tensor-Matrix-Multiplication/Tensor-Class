import numpy as np
from pysat.solvers import Glucose3
from pysat.formula import CNF
import itertools
from collections import defaultdict

class MatrixMultiplicationSAT:
    def __init__(self, size=2, num_multiplications=7):
        self.size = size
        self.num_multiplications = num_multiplications
        self.cnf = CNF()
        
        # Variable mapping
        self.var_counter = 1
        self.alpha_vars = {}  # alpha[l][i][j] for multiplication l, matrix A entry (i,j)
        self.beta_vars = {}   # beta[l][i][j] for multiplication l, matrix B entry (i,j)
        self.gamma_vars = {}  # gamma[l][i][j] for multiplication l, output C entry (i,j)
        
        # Auxiliary variables for Tseitin transformation
        self.aux_vars = {}
        
        self._create_variables()
        self._encode_brent_equations()
    
    def _get_new_var(self):
        var = self.var_counter
        self.var_counter += 1
        return var
    
    def _create_variables(self):
        """Create all base variables for the SAT encoding"""
        # Create alpha variables (coefficients for matrix A)
        for l in range(self.num_multiplications):
            self.alpha_vars[l] = {}
            for i in range(self.size):
                self.alpha_vars[l][i] = {}
                for j in range(self.size):
                    self.alpha_vars[l][i][j] = self._get_new_var()
        
        # Create beta variables (coefficients for matrix B)
        for l in range(self.num_multiplications):
            self.beta_vars[l] = {}
            for i in range(self.size):
                self.beta_vars[l][i] = {}
                for j in range(self.size):
                    self.beta_vars[l][i][j] = self._get_new_var()
        
        # Create gamma variables (coefficients for output C)
        for l in range(self.num_multiplications):
            self.gamma_vars[l] = {}
            for i in range(self.size):
                self.gamma_vars[l][i] = {}
                for j in range(self.size):
                    self.gamma_vars[l][i][j] = self._get_new_var()
    
    def _add_xor_constraint(self, literals):
        """Add XOR constraint for a list of literals using auxiliary variables"""
        if len(literals) == 0:
            return
        elif len(literals) == 1:
            self.cnf.append([literals[0]])
            return
        elif len(literals) == 2:
            # XOR of two literals: (a ⊕ b) = (¬a ∨ ¬b) ∧ (a ∨ b)
            self.cnf.append([-literals[0], -literals[1]])
            self.cnf.append([literals[0], literals[1]])
            return
        
        # For more than 2 literals, use auxiliary variables
        aux_var = self._get_new_var()
        
        # First XOR: aux = literals[0] ⊕ literals[1] ⊕ literals[2]
        self._add_three_xor(aux_var, literals[0], literals[1], literals[2])
        
        # Recursively handle the rest
        remaining = [aux_var] + literals[3:]
        self._add_xor_constraint(remaining)
    
    def _add_three_xor(self, result, a, b, c):
        """Add constraint: result ↔ (a ⊕ b ⊕ c)"""
        # result = a ⊕ b ⊕ c
        # This is true when an odd number of {a, b, c} are true
        self.cnf.append([-result, -a, -b, -c])  # not all true
        self.cnf.append([-result, a, b, -c])    # a,b true, c false
        self.cnf.append([-result, a, -b, c])    # a,c true, b false  
        self.cnf.append([-result, -a, b, c])    # b,c true, a false
        self.cnf.append([result, -a, -b, c])    # only c true
        self.cnf.append([result, -a, b, -c])    # only b true
        self.cnf.append([result, a, -b, -c])    # only a true
        self.cnf.append([result, a, b, c])      # all true
    
    def _add_and_constraint(self, result, literals):
        """Add constraint: result ↔ (literals[0] ∧ literals[1] ∧ ...)"""
        if len(literals) == 2:
            # result ↔ (a ∧ b)
            a, b = literals
            self.cnf.append([-result, a])     # result → a
            self.cnf.append([-result, b])     # result → b  
            self.cnf.append([result, -a, -b]) # (a ∧ b) → result
        else:
            # For more literals, chain them
            aux_var = self._get_new_var()
            self._add_and_constraint(aux_var, literals[:2])
            self._add_and_constraint(result, [aux_var] + literals[2:])
    
    def _encode_brent_equations(self):
        """Encode the Brent equations as CNF clauses"""
        # For 2x2 matrices, we have Brent equations:
        # sum_l (alpha[l][i1][i2] * beta[l][j1][j2] * gamma[l][k1][k2]) = delta[i2][j1] * delta[i1][k1] * delta[j2][k2]
        
        for i1 in range(self.size):
            for i2 in range(self.size):
                for j1 in range(self.size):
                    for j2 in range(self.size):
                        for k1 in range(self.size):
                            for k2 in range(self.size):
                                # Right hand side of Brent equation
                                rhs = 1 if (i2 == j1 and i1 == k1 and j2 == k2) else 0
                                
                                # Left hand side: sum of products
                                product_vars = []
                                for l in range(self.num_multiplications):
                                    # Create auxiliary variable for the triple product
                                    product_var = self._get_new_var()
                                    product_vars.append(product_var)
                                    
                                    # product_var ↔ (alpha[l][i1][i2] ∧ beta[l][j1][j2] ∧ gamma[l][k1][k2])
                                    self._add_and_constraint(product_var, [
                                        self.alpha_vars[l][i1][i2],
                                        self.beta_vars[l][j1][j2], 
                                        self.gamma_vars[l][k1][k2]
                                    ])
                                
                                # XOR all products should equal rhs
                                if rhs == 1:
                                    # Odd number of products should be true
                                    self._add_xor_constraint(product_vars)
                                else:
                                    # Even number of products should be true (including 0)
                                    # This means XOR should be false
                                    aux_xor = self._get_new_var()
                                    self._add_xor_constraint([aux_xor] + product_vars)
                                    self.cnf.append([-aux_xor])  # Force XOR to be false
    
    def fix_strassen_multiplication(self, mult_index, alpha_coeffs, beta_coeffs, gamma_coeffs):
        """
        Fix a specific multiplication to match Strassen's algorithm
        mult_index: which multiplication (0-6) to fix
        alpha_coeffs: 2x2 matrix of coefficients for A matrix
        beta_coeffs: 2x2 matrix of coefficients for B matrix  
        gamma_coeffs: 2x2 matrix of coefficients for C matrix
        """
        # Fix alpha coefficients
        for i in range(self.size):
            for j in range(self.size):
                if alpha_coeffs[i][j] == 1:
                    self.cnf.append([self.alpha_vars[mult_index][i][j]])
                else:
                    self.cnf.append([-self.alpha_vars[mult_index][i][j]])
        
        # Fix beta coefficients  
        for i in range(self.size):
            for j in range(self.size):
                if beta_coeffs[i][j] == 1:
                    self.cnf.append([self.beta_vars[mult_index][i][j]])
                else:
                    self.cnf.append([-self.beta_vars[mult_index][i][j]])
        
        # Fix gamma coefficients
        for i in range(self.size):
            for j in range(self.size):
                if gamma_coeffs[i][j] == 1:
                    self.cnf.append([self.gamma_vars[mult_index][i][j]])
                else:
                    self.cnf.append([-self.gamma_vars[mult_index][i][j]])
    
    def solve(self):
        """Solve the SAT instance and return the solution"""
        solver = Glucose3()
        
        # Add all clauses to solver
        for clause in self.cnf.clauses:
            solver.add_clause(clause)
        
        if solver.solve():
            model = solver.get_model()
            return self._extract_solution(model)
        else:
            return None
    
    def _extract_solution(self, model):
        """Extract the matrix multiplication scheme from the SAT solution"""
        solution = {
            'multiplications': [],
            'combinations': []
        }
        
        # Extract each multiplication
        for l in range(self.num_multiplications):
            # Extract alpha coefficients (for matrix A)
            alpha_matrix = np.zeros((self.size, self.size), dtype=int)
            for i in range(self.size):
                for j in range(self.size):
                    var = self.alpha_vars[l][i][j]
                    if var in model and model[var-1] > 0:
                        alpha_matrix[i][j] = 1
            
            # Extract beta coefficients (for matrix B)
            beta_matrix = np.zeros((self.size, self.size), dtype=int)
            for i in range(self.size):
                for j in range(self.size):
                    var = self.beta_vars[l][i][j]
                    if var in model and model[var-1] > 0:
                        beta_matrix[i][j] = 1
            
            # Extract gamma coefficients (for output C)
            gamma_matrix = np.zeros((self.size, self.size), dtype=int)
            for i in range(self.size):
                for j in range(self.size):
                    var = self.gamma_vars[l][i][j] 
                    if var in model and model[var-1] > 0:
                        gamma_matrix[i][j] = 1
            
            solution['multiplications'].append({
                'index': l,
                'alpha': alpha_matrix,
                'beta': beta_matrix,
                'gamma': gamma_matrix
            })
        
        return solution
    
    def print_solution(self, solution):
        """Print the matrix multiplication scheme in a readable format"""
        if solution is None:
            print("No solution found!")
            return
        
        print("Matrix Multiplication Scheme (2x2 with 7 multiplications):")
        print("=" * 60)
        
        for mult in solution['multiplications']:
            l = mult['index']
            alpha = mult['alpha']
            beta = mult['beta'] 
            gamma = mult['gamma']
            
            print(f"\nM_{l+1} = (", end="")
            
            # Print A linear combination
            a_terms = []
            for i in range(self.size):
                for j in range(self.size):
                    if alpha[i][j] == 1:
                        a_terms.append(f"a_{i+1}{j+1}")
            print(" + ".join(a_terms) if a_terms else "0", end="")
            
            print(")(", end="")
            
            # Print B linear combination  
            b_terms = []
            for i in range(self.size):
                for j in range(self.size):
                    if beta[i][j] == 1:
                        b_terms.append(f"b_{i+1}{j+1}")
            print(" + ".join(b_terms) if b_terms else "0", end="")
            
            print(")")
            
            # Print which C entries this contributes to
            c_terms = []
            for i in range(self.size):
                for j in range(self.size):
                    if gamma[i][j] == 1:
                        c_terms.append(f"c_{i+1}{j+1}")
            if c_terms:
                print(f"    contributes to: {', '.join(c_terms)}")

def get_strassen_multiplications():
    """Return Strassen's 7 multiplications as coefficient matrices"""
    # Strassen's algorithm:
    # M1 = (a11 + a22)(b11 + b22) -> c11, c22
    # M2 = (a21 + a22)(b11) -> c21, c22  
    # M3 = (a11)(b12 - b22) -> c12, c22
    # M4 = (a22)(b21 - b11) -> c11, c21
    # M5 = (a11 + a12)(b22) -> c11, c12
    # M6 = (a21 - a11)(b11 + b12) -> c22
    # M7 = (a12 - a22)(b21 + b22) -> c11, c22
    
    multiplications = [
        # M1 = (a11 + a22)(b11 + b22)
        {
            'alpha': [[1, 0], [0, 1]],  # a11 + a22
            'beta': [[1, 0], [0, 1]],   # b11 + b22
            'gamma': [[1, 0], [0, 1]]   # contributes to c11, c22
        },
        # M2 = (a21 + a22)(b11)
        {
            'alpha': [[0, 0], [1, 1]],  # a21 + a22
            'beta': [[1, 0], [0, 0]],   # b11
            'gamma': [[0, 0], [1, 1]]   # contributes to c21, c22
        },
        # M3 = (a11)(b12 - b22) - Note: we can't represent subtraction in Z2, so this needs adjustment
        {
            'alpha': [[1, 0], [0, 0]],  # a11
            'beta': [[0, 1], [0, 1]],   # b12 + b22 (XOR in Z2)
            'gamma': [[0, 1], [0, 1]]   # contributes to c12, c22
        },
        # M4 = (a22)(b21 - b11) 
        {
            'alpha': [[0, 0], [0, 1]],  # a22
            'beta': [[1, 0], [1, 0]],   # b21 + b11 (XOR in Z2)
            'gamma': [[1, 0], [1, 0]]   # contributes to c11, c21
        },
        # M5 = (a11 + a12)(b22)
        {
            'alpha': [[1, 1], [0, 0]],  # a11 + a12
            'beta': [[0, 0], [0, 1]],   # b22
            'gamma': [[1, 1], [0, 0]]   # contributes to c11, c12
        },
        # M6 = (a21 - a11)(b11 + b12)
        {
            'alpha': [[1, 0], [1, 0]],  # a21 + a11 (XOR in Z2)
            'beta': [[1, 1], [0, 0]],   # b11 + b12
            'gamma': [[0, 0], [0, 1]]   # contributes to c22
        },
        # M7 = (a12 - a22)(b21 + b22)
        {
            'alpha': [[0, 1], [0, 1]],  # a12 + a22 (XOR in Z2)  
            'beta': [[0, 0], [1, 1]],   # b21 + b22
            'gamma': [[1, 0], [0, 1]]   # contributes to c11, c22
        }
    ]
    
    return multiplications

# Example usage
if __name__ == "__main__":
    # Create SAT instance for 2x2 matrix multiplication
    sat_instance = MatrixMultiplicationSAT(size=2, num_multiplications=7)
    
    # Get Strassen's multiplications
    strassen_mults = get_strassen_multiplications()
    
    # Fix some of Strassen's multiplications (e.g., first 3)
    print("Fixing first 3 Strassen multiplications...")
    for i in range(3):
        sat_instance.fix_strassen_multiplication(
            i, 
            strassen_mults[i]['alpha'],
            strassen_mults[i]['beta'], 
            strassen_mults[i]['gamma']
        )
    
    print("Solving for remaining multiplications...")
    solution = sat_instance.solve()
    
    if solution:
        print("Solution found!")
        sat_instance.print_solution(solution)
    else:
        print("No solution found - the problem may be unsatisfiable with the given constraints.")