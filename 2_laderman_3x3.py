import collections
from pysat.solvers import Glucose3
from itertools import product
import re
import time
import random
import statistics
import multiprocessing
import queue
import json

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

# --- Refactored and New Functions for Analysis ---

def get_laderman_definitions():
    """
    Returns the complete Laderman's algorithm definitions as structured data.
    This centralizes the algorithm's definition.
    """
    m_defs_str = {
        1: "(a11 + a12 + a13 - a21 - a22 - a32 - a33) * b22", 2: "(a11 - a21) * (b12 + b22)",
        3: "a22 * (-b11 + b12 + b21 - b22 - b23 + b31 - b33)", 4: "(-a11 + a21 + a22) * (b11 - b12 + b22)",
        5: "(a21 + a22) * (-b11 + b12)", 6: "a11 * b11", 7: "(-a11 + a31 + a32) * (b11 - b13 + b23)",
        8: "(-a11 + a31) * (b13 - b23)", 9: "(a31 + a32) * (-b11 + b13)",
        10: "(a11 + a12 + a13 - a22 - a23 - a31 - a32) * b23",
        11: "a32 * (-b11 + b13 + b21 - b22 - b23 - b31 + b32)", 12: "(-a13 + a32 + a33) * (b22 + b31 - b32)",
        13: "(a13 - a33) * (b22 - b32)", 14: "a13 * b31", 15: "(a32 + a33) * (-b31 + b32)",
        16: "(-a13 + a22 + a23) * (b23 + b31 - b33)", 17: "(-a13 + a23) * (b23 + b33)",
        18: "(a22 + a23) * (b31 - b33)", 19: "a12 * b21", 20: "a23 * b32", 21: "a21 * b13",
        22: "a31 * b12", 23: "a33 * b33"
    }
    c_defs_str = {
        "11": "m6 + m14 + m19", "12": "m1 + m4 + m5 + m6 + m12 + m14 + m15",
        "13": "m6 + m7 + m9 + m10 + m14 + m16 + m18", "21": "m2 + m3 + m4 + m6 + m14 + m16 + m17",
        "22": "m2 + m4 + m5 + m6 + m20", "23": "m14 + m16 + m17 + m18 + m21",
        "31": "m6 + m7 + m8 + m11 + m12 + m13 - m14", "32": "m12 + m13 + m14 + m15 + m22",
        "33": "m6 + m7 + m8 + m9 + m23"
    }
    return m_defs_str, c_defs_str

def add_laderman_assumptions(var_manager, matrix_size, products_to_include):
    """
    Encodes a specific subset of the Laderman equations as unit clauses.
    """
    assumptions = []
    m_defs, c_defs = get_laderman_definitions()

    def get_terms(s):
        return {term for term in re.findall(r'[ab]\d{2}', s)}

    for r in products_to_include:
        a_str, b_str = m_defs[r].split('*')
        a_terms = get_terms(a_str)
        b_terms = get_terms(b_str)
        
        for i, j in product(range(1, matrix_size + 1), repeat=2):
            var_a = var_manager.get_var('a', r, i, j)
            assumptions.append([var_a] if f'a{i}{j}' in a_terms else [-var_a])
            
            var_b = var_manager.get_var('b', r, i, j)
            assumptions.append([var_b] if f'b{i}{j}' in b_terms else [-var_b])

    for i, j in product(range(1, matrix_size + 1), repeat=2):
        c_terms_full_definition = {int(term) for term in re.findall(r'\d+', c_defs[f'{i}{j}'])}
        for r in products_to_include:
            var_c = var_manager.get_var('c', r, i, j)
            if r in c_terms_full_definition:
                assumptions.append([var_c])
            else:
                assumptions.append([-var_c])
                
    return assumptions

def solver_worker(clauses, var_manager, matrix_size, num_products, result_queue):
    """
    A target function to run the SAT solver and decode the result in a separate process.
    """
    try:
        with Glucose3(bootstrap_with=clauses) as solver:
            is_solvable = solver.solve()
            if is_solvable:
                model = solver.get_model()
                # Decode the solution inside the worker process
                solution_data = decode_solution_to_dict(model, var_manager, matrix_size, num_products)
                result_queue.put((True, solution_data))
            else:
                result_queue.put((False, None))
    except Exception as e:
        result_queue.put((f"Error: {e}", None))

def decode_solution_to_dict(model, var_manager, matrix_size, num_products):
    """
    Decodes the SAT model into a structured dictionary instead of printing.
    """
    true_vars = {v for v in model if v > 0}
    
    solution_data = {
        "M_formulas": collections.defaultdict(str),
        "C_formulas": collections.defaultdict(str)
    }

    raw_products = collections.defaultdict(lambda: {"A": [], "B": []})
    for r, i, j in product(range(1, num_products + 1), range(1, matrix_size + 1), range(1, matrix_size + 1)):
        var_tuple = ('a', r, i, j)
        if var_tuple in var_manager.var_to_int and var_manager.get_var('a', r, i, j) in true_vars:
            raw_products[r]["A"].append(f"A_{i}{j}")
        
        var_tuple = ('b', r, i, j)
        if var_tuple in var_manager.var_to_int and var_manager.get_var('b', r, i, j) in true_vars:
            raw_products[r]["B"].append(f"B_{i}{j}")

    for r in range(1, num_products + 1):
        a_str = " + ".join(sorted(raw_products[r]["A"])) if raw_products[r]["A"] else "0"
        b_str = " + ".join(sorted(raw_products[r]["B"])) if raw_products[r]["B"] else "0"
        solution_data["M_formulas"][f"M_{r}"] = f"({a_str}) * ({b_str})"

    for i, j in product(range(1, matrix_size + 1), repeat=2):
        c_idx = f"C_{i}{j}"
        terms = []
        for r in range(1, num_products + 1):
            var_tuple = ('c', r, i, j)
            if var_tuple in var_manager.var_to_int and var_manager.get_var('c', r, i, j) in true_vars:
                terms.append(f"M_{r}")
        m_str = " + ".join(sorted(terms, key=lambda t: int(t.split('_')[1]))) if terms else "0"
        solution_data["C_formulas"][c_idx] = m_str
        
    return solution_data

def analyze_laderman_subset(num_to_provide, num_total, num_runs, top_k, timeout_seconds):
    """
    Main analysis function. Runs the solver multiple times with random subsets
    of Laderman's equations and reports statistics.
    """
    MATRIX_SIZE = 3
    
    print("="*60)
    print(f"Starting Analysis: Providing {num_to_provide} of {num_total} random multiplications.")
    print(f"Running {num_runs} times with a timeout of {timeout_seconds} seconds per run.")
    print("="*60 + "\n")

    print("1. Pre-generating all Brent equation clauses and the master variable map...")
    var_manager_base = VariableMapper()
    
    # This loop populates the one true variable mapper
    for r, i, j in product(range(1, num_total + 1), range(1, MATRIX_SIZE + 1), range(1, MATRIX_SIZE + 1)):
        var_manager_base.get_var('a', r, i, j)
        var_manager_base.get_var('b', r, i, j)
        var_manager_base.get_var('c', r, i, j)
        
    brent_clauses = []
    indices = product(range(1, MATRIX_SIZE + 1), repeat=6)
    for i1, i2, j1, j2, k1, k2 in indices:
        rhs = 1 if (i1 == k1 and j2 == k2 and i2 == j1) else 0
        equation_terms = [(var_manager_base.get_var('a', r, i1, i2), var_manager_base.get_var('b', r, j1, j2), var_manager_base.get_var('c', r, k1, k2)) for r in range(1, num_total + 1)]
        clauses = convert_brent_equation_to_cnf(equation_terms, rhs, var_manager_base)
        brent_clauses.extend(clauses)
    print(f"   ...done. Generated {len(brent_clauses)} base clauses with {var_manager_base.var_count} variables.\n")

    results = []
    timeouts = 0
    crashes_or_unsat = 0
    all_product_indices = list(range(1, num_total + 1))

    print(f"2. Running {num_runs} randomized experiments...")
    for i in range(num_runs):
        products_to_include = sorted(random.sample(all_product_indices, num_to_provide))
        print(f"   - Run {i+1}/{num_runs} with M set: {products_to_include}", end="", flush=True)
        
        # Add the assumptions for this specific run using the master variable map
        assumptions = add_laderman_assumptions(var_manager_base, MATRIX_SIZE, products_to_include)
        final_clauses = brent_clauses + assumptions

        start_time = time.time()
        result_queue = multiprocessing.Queue()
        # **BUG FIX**: Pass the master variable map to the worker so it can decode the solution correctly.
        solver_process = multiprocessing.Process(
            target=solver_worker, 
            args=(final_clauses, var_manager_base, MATRIX_SIZE, num_total, result_queue)
        )
        
        solver_process.start()
        solver_process.join(timeout_seconds)
        
        elapsed_time = time.time() - start_time
        
        if solver_process.is_alive():
            solver_process.terminate()
            solver_process.join()
            timeouts += 1
            print(f" -> TIMEOUT (exceeded {timeout_seconds}s)")
        else:
            try:
                is_solvable, solution_data = result_queue.get_nowait()
                if is_solvable and solution_data:
                    results.append({"time": elapsed_time, "products": products_to_include, "solution": solution_data})
                    print(f" -> Solvable ({elapsed_time:.4f}s)")
                else:
                    crashes_or_unsat += 1
                    print(f" -> UNSAT or Crashed ({is_solvable})")
            except queue.Empty:
                crashes_or_unsat += 1
                print(" -> Crashed (Queue Empty)")

    print("   ...all runs complete.\n")

    if not results:
        print(f"No solvable runs were completed. ({timeouts} runs timed out, {crashes_or_unsat} failed). Cannot perform analysis.")
        return

    print("3. Analyzing results...")
    times = [r['time'] for r in results]
    min_time = min(times)
    max_time = max(times)
    avg_time = statistics.mean(times)
    results.sort(key=lambda x: x['time'])

    print("\n" + "="*60)
    print("📊 Performance Analysis Summary 📊")
    print("="*60)
    print(f"Number of multiplications provided: {num_to_provide}/{num_total}")
    print(f"Number of runs attempted:         {num_runs}")
    print(f"Number of successful runs:        {len(results)}")
    print(f"Number of timed out runs:         {timeouts}")
    print(f"Number of UNSAT/crashed runs:     {crashes_or_unsat}")
    
    if results:
        print(f"\n🕒 Solve Time Statistics (for successful runs):")
        print(f"   - Min Time:    {min_time:.4f} seconds")
        print(f"   - Max Time:    {max_time:.4f} seconds")
        print(f"   - Average Time: {avg_time:.4f} seconds")

        print(f"\n🚀 Top {top_k} Fastest Runs (Easiest for the Solver):")
        for i in range(min(top_k, len(results))):
            run = results[i]
            print(f"   {i+1}. Time: {run['time']:.4f}s | M set used: {run['products']}")

        print(f"\n🐢 Top {top_k} Slowest Runs (Hardest for the Solver):")
        for i in range(min(top_k, len(results))):
            run = results[-(i+1)]
            print(f"   {i+1}. Time: {run['time']:.4f}s | M set used: {run['products']}")
    print("="*60)

    # --- Step 4: Save all found solutions to a JSON file ---
    print("\n4. Saving all found solutions to solutions.json...")
    solutions_to_save = [
        {"products_used": r["products"], "solution": r["solution"]} for r in results
    ]
    with open(f'solutions_{num_to_provide}_{num_runs}_{timeout_seconds}.json', "w") as f:
        json.dump(solutions_to_save, f, indent=4)
    print("   ...done.")


# --- Main Execution ---
if __name__ == "__main__":
    
    # --- CONFIGURATION ---
    NUM_PRODUCTS_TO_PROVIDE = 18
    NUM_RUNS = 50
    TOP_K_RESULTS = 5
    SOLVER_TIMEOUT_SECONDS = 20 # Set the timeout for each solver run
    # -------------------

    total_start_time = time.time()

    analyze_laderman_subset(
        num_to_provide=NUM_PRODUCTS_TO_PROVIDE,
        num_total=23,
        num_runs=NUM_RUNS,
        top_k=TOP_K_RESULTS,
        timeout_seconds=SOLVER_TIMEOUT_SECONDS
    )
    
    total_end_time = time.time()
    print(f"\nTotal analysis completed in {total_end_time - total_start_time:.2f} seconds.")

