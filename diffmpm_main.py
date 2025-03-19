from diffmpm_final import *
import subprocess
import json

parent_geno = [(0.0, 0.0, 0.1, 0.05, -1, 1, 2), (0.0, 0.0, 0.03, 0.04, 0, 1, 0), (0.1, 0.0, 0.03, 0.03, 1, 1, 1), 2]
# gens = 3
# children = 5

# gen_loss = []

# geno_str = json.dumps(parent_geno)
# print('blah')

# for i in range(3):  # Run different robot shapes
#     print('blah2')
#     result = subprocess.run(["python", "diffmpm_lab4.py", "--genotype", geno_str], capture_output=True, text=True)
#     print('blah3')

# print(result)

# Run without mutation
final, loss = run(parent_geno)

print("Genotype ",final)
print("Final Loss = ",loss)

# # Mutate genotype
# geno_mut = mutate_geno(parent_geno)

# final_mut, loss_mut = run(geno_mut)

# print("Genotype ",final_mut)
# print("Final Loss = ",loss_mut)