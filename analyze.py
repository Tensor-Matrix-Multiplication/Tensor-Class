from collections import defaultdict
import json

# Load the JSON file
with open("solutions_18_50_20.json", "r") as file:
    solutions = json.load(file)

mul_list = [[] for _ in solutions]
print(mul_list, len(mul_list))

for i, res in enumerate(solutions):
    formulaes = res["solution"]["M_formulas"]
    for key, val in formulaes.items():
        mul_list[i].append(val)

for mul in mul_list:
    mul.sort()

map_mul_cnt = defaultdict(int) # 'multiplication tuple' -> number of times its repeated
for mul in mul_list:
    map_mul_cnt[tuple(mul)] += 1

print(map_mul_cnt)