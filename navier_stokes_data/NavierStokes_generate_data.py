# NavierStokes test
# ================
import firedrake as fd
import matplotlib.pyplot as plt
import numpy as np

# Save series expansion
from series_approximation import FourierExpansion2d, GaussianExpansion2d


# mesh = fd.Mesh("cylinder_mesh1.msh")
mesh = fd.Mesh("../../meshes/cylinder_050.msh")

# fig, axes = plt.subplots()
# fd.triplot(mesh, axes=axes)
# axes.axis("off")
# axes.set_aspect("equal")
# axes.legend(loc="upper right")
# plt.show()


# physical constants
nu_val = 0.001
nu = fd.Constant(nu_val)

# time step
dt = 0.01
# define a firedrake constant equal to dt so that variation forms
# not regenerated if we change the time step
k = fd.Constant(dt)


V = fd.VectorFunctionSpace(mesh, "CG", 2)
Q = fd.FunctionSpace(mesh, "CG", 1)


u = fd.TrialFunction(V)
v = fd.TestFunction(V)

p = fd.TrialFunction(Q)
q = fd.TestFunction(Q)

vortex = fd.Function(Q)

u_now = fd.Function(V)
u_next = fd.Function(V)
u_star = fd.Function(V)
p_now = fd.Function(Q)
p_next = fd.Function(Q)


# Expressions for the variational forms
n = fd.FacetNormal(mesh)
f = fd.Constant((0.0, 0.0))
u_mid = 0.5 * (u_now + u)


def sigma(u, p):
    return 2 * nu * fd.sym(fd.nabla_grad(u)) - p * fd.Identity(len(u))


x, y = fd.SpatialCoordinate(mesh)

# Define boundary conditions
bcu = [
    fd.DirichletBC(V, fd.Constant((0, 0)), (1, 4)),  # top-bottom and cylinder
    fd.DirichletBC(V, ((4.0 * 1.5 * y * (0.41 - y) / 0.41**2), 0), 2),
]  # inflow
bcp = [fd.DirichletBC(Q, fd.Constant(0), 3)]  # outflow

# # Define boundary conditions Note need to be adapted to mesh
# bcu = [
#     fd.DirichletBC(V, fd.Constant((0, 0)), (3, 4, 5)),  # top-bottom and cylinder
#     fd.DirichletBC(V, ((4.0 * 1.5 * y * (0.41 - y) / 0.41 ** 2), 0), 1),
# ]  # inflow
# bcp = [fd.DirichletBC(Q, fd.Constant(0), 2)]  # outflow


U_mean = 1.0
re_num = int(U_mean * 0.1 / nu_val)
print(f"Re = {re_num}")

# Define variational forms
F1 = (
    fd.inner((u - u_now) / k, v) * fd.dx
    + fd.inner(fd.dot(u_now, fd.nabla_grad(u_mid)), v) * fd.dx
    + fd.inner(sigma(u_mid, p_now), fd.sym(fd.nabla_grad(v))) * fd.dx
    + fd.inner(p_now * n, v) * fd.ds
    - fd.inner(nu * fd.dot(fd.nabla_grad(u_mid), n), v) * fd.ds
    - fd.inner(f, v) * fd.dx
)

a1, L1 = fd.system(F1)

a2 = fd.inner(fd.nabla_grad(p), fd.nabla_grad(q)) * fd.dx
L2 = (
    fd.inner(fd.nabla_grad(p_now), fd.nabla_grad(q)) * fd.dx
    - (1 / k) * fd.inner(fd.div(u_star), q) * fd.dx
)

a3 = fd.inner(u, v) * fd.dx
L3 = (
    fd.inner(u_star, v) * fd.dx - k * fd.inner(fd.nabla_grad(p_next - p_now), v) * fd.dx
)

# Define linear problems
prob1 = fd.LinearVariationalProblem(a1, L1, u_star, bcs=bcu)
prob2 = fd.LinearVariationalProblem(a2, L2, p_next, bcs=bcp)
prob3 = fd.LinearVariationalProblem(a3, L3, u_next)

# Define solvers
solve1 = fd.LinearVariationalSolver(
    prob1, solver_parameters={"ksp_type": "gmres", "pc_type": "sor"}
)
solve2 = fd.LinearVariationalSolver(
    prob2, solver_parameters={"ksp_type": "cg", "pc_type": "gamg"}
)
solve3 = fd.LinearVariationalSolver(
    prob3, solver_parameters={"ksp_type": "cg", "pc_type": "sor"}
)

# Prep for saving solutions
# u_save = fd.Function(V).assign(u_now)
# p_save = fd.Function(Q).assign(p_now)
# outfile_u = fd.File("outputs_sim/cylinder/u.pvd")
# outfile_p = fd.File("outputs_sim/cylinder/p.pvd")
# outfile_u.write(u_save)
# outfile_p.write(p_save)

# Time loop
t = 0.0
t_end = 4.0

total_step = int((t_end - t) / dt)
print("Beginning time loop...")

# Create output file
outfile_pressure = fd.VTKFile("NavierStokes_p.pvd")
outfile_velocity = fd.VTKFile("NavierStokes_u.pvd")

V_out_vec = fd.VectorFunctionSpace(mesh, "CG", 2)
V_out_sca = fd.FunctionSpace(mesh, "CG", 1)

while t < t_end:

    nxmodes = 8
    nymodes = 8
    expansion = GaussianExpansion2d(nxmodes, nymodes, 0, 2.2, 0, 0.41)

    expansion.approximate(u_now)
    expansion.save("navier_stokes_series_coeffs_cylinder_8terms/u_expansion{0:d}.json".format(int(t * 100)))

    expansion.approximate(p_now)
    expansion.save("navier_stokes_series_coeffs_cylinder_8terms/p_expansion{0:d}.json".format(int(t * 100)))

    # input("Press Enter to continue...")
    
    solve1.solve()
    solve2.solve()
    solve3.solve()

    t += dt

    # p_save.assign(p_next)
    # outfile.write(fd.project(u_next, V_out_vec, name="Velocity"))
    outfile_pressure.write(fd.project(p_next, V_out_sca, name="Pressure"))
    outfile_velocity.write(fd.project(u_next, V_out_vec, name="Velocity"))
    # outfile_p.write(p_save)

    # u_list.append(fd.Function(u_next))

    # update solutions
    u_now.assign(u_next)
    p_now.assign(p_next)

    # if np.abs(t - np.round(t, decimals=0)) < 1.0e-8:
    #print("time = {0:.3f}".format(t))
    print("time = {0:d}".format(int(t * 100)))


