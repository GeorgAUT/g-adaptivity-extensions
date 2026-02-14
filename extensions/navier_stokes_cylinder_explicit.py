import os

import matplotlib.pyplot as plt
import numpy as np
import imageio.v2 as imageio

# from firedrake import (FacetNormal, Function, FunctionSpace, Identity, LinearVariationalProblem,
#                       LinearVariationalSolver, Mesh, SpatialCoordinate, TestFunction, TrialFunction,
#                       VectorFunctionSpace, assemble, div, dot, ds, dx, inner, nabla_grad, sym, Constant,
#                       sqrt, DirichletBC)

from firedrake import *
from firedrake.pyplot import tripcolor, triplot


mesh_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "meshes", "cylinder_050.msh"))
mesh = Mesh(mesh_path)

markers = list(mesh.topology.exterior_facets.unique_markers)
need = {1, 2, 3, 4}
if not need.issubset(set(markers)):
    raise ValueError(f"Mesh boundary markers are {sorted(markers)} but this script expects {sorted(need)}")

nu_val = 0.001
dt_val = 0.02
num_steps = 250
U_mean = 1.0

nu = Constant(nu_val)
k = Constant(dt_val)

V = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)

u_trial = TrialFunction(V)
v_test = TestFunction(V)
p_trial = TrialFunction(Q)
q_test = TestFunction(Q)

u_now = Function(V)
u_next = Function(V)
u_star = Function(V)
p_now = Function(Q)
p_next = Function(Q)

n = FacetNormal(mesh)
f = Constant((0.0, 0.0))

coords = mesh.coordinates.dat.data_ro
x0 = float(coords[:, 0].min())
x1 = float(coords[:, 0].max())
y0 = float(coords[:, 1].min())
y1 = float(coords[:, 1].max())
H = y1 - y0

x, y = SpatialCoordinate(mesh)
inflow_u = as_vector((4.0 * U_mean * (y - y0) * (y1 - y) / (H * H), 0.0))

bcu = [
    DirichletBC(V, Constant((0.0, 0.0)), (1, 4)),
    DirichletBC(V, inflow_u, 2),
]
bcp = [DirichletBC(Q, Constant(0.0), 3)]


def sigma(u, p):
    return 2 * nu * sym(nabla_grad(u)) - p * Identity(len(u))


u_mid = 0.5 * (u_now + u_trial)

F1 = (
    inner((u_trial - u_now) / k, v_test) * dx
    + inner(dot(u_now, nabla_grad(u_mid)), v_test) * dx
    + inner(sigma(u_mid, p_now), sym(nabla_grad(v_test))) * dx
    + inner(p_now * n, v_test) * ds
    - inner(nu * dot(nabla_grad(u_mid), n), v_test) * ds
    - inner(f, v_test) * dx
)

a1, L1 = system(F1)

a2 = inner(nabla_grad(p_trial), nabla_grad(q_test)) * dx
L2 = inner(nabla_grad(p_now), nabla_grad(q_test)) * dx - (1.0 / k) * inner(div(u_star), q_test) * dx

a3 = inner(u_trial, v_test) * dx
L3 = inner(u_star, v_test) * dx - k * inner(nabla_grad(p_next - p_now), v_test) * dx

problem1 = LinearVariationalProblem(a1, L1, u_star, bcs=bcu)
problem2 = LinearVariationalProblem(a2, L2, p_next, bcs=bcp)
problem3 = LinearVariationalProblem(a3, L3, u_next)

solver1 = LinearVariationalSolver(problem1, solver_parameters={"ksp_type": "gmres", "pc_type": "sor"})
solver2 = LinearVariationalSolver(problem2, solver_parameters={"ksp_type": "cg", "pc_type": "gamg"})
solver3 = LinearVariationalSolver(problem3, solver_parameters={"ksp_type": "cg", "pc_type": "sor"})

u_now.assign(Constant((0.0, 0.0)))
p_now.assign(Constant(0.0))
u_star.assign(u_now)
u_next.assign(u_now)
p_next.assign(p_now)

Vmag = FunctionSpace(mesh, "CG", 1)
u_mag_f = Function(Vmag)

p_f = Function(Q)
omega_f = Function(Vmag)

video_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "navier_stokes_cylinder.gif"))
video_path_solution_only = os.path.abspath(os.path.join(os.path.dirname(__file__), "navier_stokes_cylinder_solution.gif"))
video_path_pressure = os.path.abspath(os.path.join(os.path.dirname(__file__), "navier_stokes_cylinder_pressure.gif"))
video_path_pressure_solution_only = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "navier_stokes_cylinder_pressure_solution.gif")
)
video_path_vorticity = os.path.abspath(os.path.join(os.path.dirname(__file__), "navier_stokes_cylinder_vorticity.gif"))
video_path_vorticity_solution_only = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "navier_stokes_cylinder_vorticity_solution.gif")
)
fps = 10
frame_every = 1

fig_vid, ax_vid = plt.subplots(nrows=1, ncols=1, figsize=(6, 5))
writer = imageio.get_writer(video_path, mode="I", duration=1.0 / fps)

fig_vid_sol, ax_vid_sol = plt.subplots(nrows=1, ncols=1, figsize=(6, 5))
writer_sol = imageio.get_writer(video_path_solution_only, mode="I", duration=1.0 / fps)

fig_vid_p, ax_vid_p = plt.subplots(nrows=1, ncols=1, figsize=(6, 5))
writer_p = imageio.get_writer(video_path_pressure, mode="I", duration=1.0 / fps)

fig_vid_p_sol, ax_vid_p_sol = plt.subplots(nrows=1, ncols=1, figsize=(6, 5))
writer_p_sol = imageio.get_writer(video_path_pressure_solution_only, mode="I", duration=1.0 / fps)

fig_vid_w, ax_vid_w = plt.subplots(nrows=1, ncols=1, figsize=(6, 5))
writer_w = imageio.get_writer(video_path_vorticity, mode="I", duration=1.0 / fps)

fig_vid_w_sol, ax_vid_w_sol = plt.subplots(nrows=1, ncols=1, figsize=(6, 5))
writer_w_sol = imageio.get_writer(video_path_vorticity_solution_only, mode="I", duration=1.0 / fps)

for step in range(num_steps):
    print("Remaining Steps: ", num_steps - step, end="\r")
    solver1.solve()
    solver2.solve()
    solver3.solve()
    u_now.assign(u_next)
    p_now.assign(p_next)

    if step % frame_every == 0:
        u_mag_expr = sqrt(inner(u_now, u_now))
        u_mag_f.project(u_mag_expr)

        p_f.assign(p_now)

        omega_expr = grad(u_now[1])[0] - grad(u_now[0])[1]
        omega_f.project(omega_expr)

        ax_vid.clear()
        tripcolor(u_mag_f, axes=ax_vid)
        triplot(mesh, axes=ax_vid)
        ax_vid.set_aspect("equal")
        ax_vid.set_title(f"|u|  t={(step + 1) * float(dt_val):.3f}")
        fig_vid.tight_layout()
        fig_vid.canvas.draw()
        rgba = np.asarray(fig_vid.canvas.buffer_rgba())
        img = rgba[:, :, :3].copy()
        writer.append_data(img)

        ax_vid_sol.clear()
        tripcolor(u_mag_f, axes=ax_vid_sol)
        ax_vid_sol.set_aspect("equal")
        ax_vid_sol.set_title(f"|u|  t={(step + 1) * float(dt_val):.3f}")
        fig_vid_sol.tight_layout()
        fig_vid_sol.canvas.draw()
        rgba_sol = np.asarray(fig_vid_sol.canvas.buffer_rgba())
        img_sol = rgba_sol[:, :, :3].copy()
        writer_sol.append_data(img_sol)

        ax_vid_p.clear()
        tripcolor(p_f, axes=ax_vid_p)
        triplot(mesh, axes=ax_vid_p)
        ax_vid_p.set_aspect("equal")
        ax_vid_p.set_title(f"p  t={(step + 1) * float(dt_val):.3f}")
        fig_vid_p.tight_layout()
        fig_vid_p.canvas.draw()
        rgba_p = np.asarray(fig_vid_p.canvas.buffer_rgba())
        img_p = rgba_p[:, :, :3].copy()
        writer_p.append_data(img_p)

        ax_vid_p_sol.clear()
        tripcolor(p_f, axes=ax_vid_p_sol)
        ax_vid_p_sol.set_aspect("equal")
        ax_vid_p_sol.set_title(f"p  t={(step + 1) * float(dt_val):.3f}")
        fig_vid_p_sol.tight_layout()
        fig_vid_p_sol.canvas.draw()
        rgba_p_sol = np.asarray(fig_vid_p_sol.canvas.buffer_rgba())
        img_p_sol = rgba_p_sol[:, :, :3].copy()
        writer_p_sol.append_data(img_p_sol)

        ax_vid_w.clear()
        tripcolor(omega_f, axes=ax_vid_w)
        triplot(mesh, axes=ax_vid_w)
        ax_vid_w.set_aspect("equal")
        ax_vid_w.set_title(f"vorticity  t={(step + 1) * float(dt_val):.3f}")
        fig_vid_w.tight_layout()
        fig_vid_w.canvas.draw()
        rgba_w = np.asarray(fig_vid_w.canvas.buffer_rgba())
        img_w = rgba_w[:, :, :3].copy()
        writer_w.append_data(img_w)

        ax_vid_w_sol.clear()
        tripcolor(omega_f, axes=ax_vid_w_sol)
        ax_vid_w_sol.set_aspect("equal")
        ax_vid_w_sol.set_title(f"vorticity  t={(step + 1) * float(dt_val):.3f}")
        fig_vid_w_sol.tight_layout()
        fig_vid_w_sol.canvas.draw()
        rgba_w_sol = np.asarray(fig_vid_w_sol.canvas.buffer_rgba())
        img_w_sol = rgba_w_sol[:, :, :3].copy()
        writer_w_sol.append_data(img_w_sol)

writer.close()
plt.close(fig_vid)
writer_sol.close()
plt.close(fig_vid_sol)
writer_p.close()
plt.close(fig_vid_p)
writer_p_sol.close()
plt.close(fig_vid_p_sol)
writer_w.close()
plt.close(fig_vid_w)
writer_w_sol.close()
plt.close(fig_vid_w_sol)

u_mag_expr = sqrt(inner(u_now, u_now))
u_mag_f.project(u_mag_expr)

fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(18, 5))

tripcolor(u_mag_f, axes=axs[0])
axs[0].set_title("|u|")
axs[0].set_aspect("equal")

tripcolor(p_now, axes=axs[1])
axs[1].set_title("p")
axs[1].set_aspect("equal")

triplot(mesh, axes=axs[2])
axs[2].set_title("mesh")
axs[2].set_aspect("equal")

fig.tight_layout()
plt.show()

print(f"Wrote video to: {video_path}")
print(f"Wrote video to: {video_path_solution_only}")
print(f"Wrote video to: {video_path_pressure}")
print(f"Wrote video to: {video_path_pressure_solution_only}")
print(f"Wrote video to: {video_path_vorticity}")
print(f"Wrote video to: {video_path_vorticity_solution_only}")
