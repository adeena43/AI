from ortools.sat.python import cp_model

model = cp_model.CpModel()

num_vals = 3

x = model.new_int_var(0, num_vals, "x")
y = model.new_int_var(0, num_vals, "y")
z = model.new_int_var(0, num_vals, "z")

model.add(x != y)
model.add(y != z)
model.add(z != x)

solver = cp_model.CpSolver()
status = solver.solve(model)

if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
    print(f"x = {solver.value(x)}")
    print(f"y = {solver.value(y)}")
    print(f"z = {solver.value(z)}")

else:
    print("No solution found")
