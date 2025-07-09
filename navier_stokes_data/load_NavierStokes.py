import firedrake as fd
from series_approximation import load_expansion

mesh = fd.Mesh("../../meshes/cylinder_010.msh")

x, y = fd.SpatialCoordinate(mesh)

V = fd.VectorFunctionSpace(mesh, "CG", 2)
Q = fd.FunctionSpace(mesh, "CG", 1)

expansion_u = load_expansion("navier_stokes_series_coeffs_cylinder_010_8terms/u_expansion40.json")
expansion_p = load_expansion("navier_stokes_series_coeffs_cylinder_010_8terms/p_expansion40.json")

u_approx = fd.Function(V, name="u_approx").interpolate(expansion_u.as_expression(x, y))
p_approx = fd.Function(Q, name="p_approx").interpolate(expansion_p.as_expression(x, y))

# file = fd.VTKFile("series_expansion.pvd")
# file.write(u_approx, p_approx)

import matplotlib.pyplot as plt

V_plot = fd.FunctionSpace(mesh, "CG", 1)
p0=fd.assemble(fd.interpolate(u_approx[1],V_plot))
# Plot firedrake mesh
fig, ax = plt.subplots(nrows=1, ncols=1)
a = fd.tripcolor(p0,axes=ax)
plt.colorbar(a)
ax.set_title("Solver class solution - u0")
plt.show()