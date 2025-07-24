def generate_brent(r):

    A_indices = [(i,j) for i in range(1,4) for j in range(1,4)]  ##Creates a list of tuples for indices
    B_indices = [(i,j) for i in range(1,4) for j in range(1,4)]
    C_indices = [(i,j) for i in range(1,4) for j in range(1,4)]


 
    equations = []
    RHS = 0
    rhs_counter=0

    for i1 in range(1,3):
        for i2 in range(1,3):
            for j1 in range(1,3):
                for j2 in range(1,3):
                    for k1 in range(1,3):
                        for k2 in range(1,3):
                            LHS = f"A_{i1}{i2} B_{j1}{j2} C_{k1}{k2}"

                            if(i2 == j1 and i1 == k1 and j2 == k2):
                                RHS = 1
                                rhs_counter += 1
                            else:
                                RHS = 0

                            equation = [LHS, RHS]
                            equations.append(equation)
                            
    return equations, rhs_counter                    
                            


x,y = generate_brent(23)

rhs1 = []
print("Brent Equations:")
for i in range (len(x)):

    if(x[i][1] == 1):
        rhs1.append(x[i])

    print(f'{x[i][0]} = {x[i][1]}')

print()
print()
print("RHS = 1 Equations:")

for i in range (len(rhs1)):
    print(f'{rhs1[i][0]} = {rhs1[i][1]}')

print()
print(f'{y}')
