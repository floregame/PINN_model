import torch
import numpy as np
import time

from scipy.spatial import Delaunay

import scipy.sparse as sp
import scipy.sparse.linalg as spla

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Выбрано: {device}")

R_wire = 0.1
R_ext = 0.5
J = 1.0
N_points = 200 

def u_analytic(r):
    
    sol_outside = (J * R_wire**2 / 2) * torch.log(R_ext / r)
    
    sol_inside = (J / 4) * (R_wire**2 - r**2) + (J * R_wire**2 / 2) * torch.log(torch.tensor(R_ext / R_wire))
    
    return torch.where(r <= R_wire, sol_inside, sol_outside)

start = time.time()

#генерация сетки
r_vals = np.linspace(0, R_ext, N_points)
theta_vals = np.linspace(0, 2 * np.pi, N_points * 2, endpoint=False)

r_grid, theta_grid = np.meshgrid(r_vals, theta_vals)

x = (r_grid * np.cos(theta_grid)).flatten()
y = (r_grid * np.sin(theta_grid)).flatten()

points = np.unique(np.stack([x, y], axis=1), axis=0)

triangle = Delaunay(points)

nodes = torch.tensor(points, dtype=torch.float32, device=device)
elements = torch.tensor(triangle.simplices, dtype=torch.long, device=device)

num_nodes = nodes.shape[0]

#локальные матрицы
def local_matrices(nodes, elements):
    p = nodes[elements]
    p1, p2, p3 = p[:, 0], p[:, 1], p[:, 2]
    
    area = 0.5 * torch.abs((p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - 
                           (p3[:, 0] - p1[:, 0]) * (p2[:, 1] - p1[:, 1]))
    
    alpha_1 = torch.stack([p2[:, 1] - p3[:, 1], 
                           p3[:, 1] - p1[:, 1], 
                           p1[:, 1] - p2[:, 1]], dim=1)
    
    alpha_2 = torch.stack([p3[:, 0] - p2[:, 0], 
                           p1[:, 0] - p3[:, 0], 
                           p2[:, 0] - p1[:, 0]], dim=1)
     
    G_local = (1.0 / (4.0 * area))[:, None, None] * (alpha_1[:, :, None] * alpha_1[:, None, :] + 
                                                     alpha_2[:, :, None] * alpha_2[:, None, :])
    
    return G_local, area

G_local, areas = local_matrices(nodes, elements)

#вектор правой части
centers = nodes[elements].mean(dim=1)
f_mask = (torch.norm(centers, dim=1) <= R_wire).float()

F_local = (f_mask * areas / 3.0)[:, None].expand(-1, 3)

#сборка глобальной матрицы
rows_np = elements[:, :, None].expand(-1, 3, 3).reshape(-1).cpu().numpy()
cols_np = elements[:, None, :].expand(-1, 3, 3).reshape(-1).cpu().numpy()

vals_np = G_local.reshape(-1).cpu().numpy()

G_sparse = sp.coo_matrix((vals_np, (rows_np, cols_np)), shape=(num_nodes, num_nodes)).tocsr()

F = torch.zeros(num_nodes, device=device)
F.index_add_(0, elements.reshape(-1), F_local.reshape(-1))
F_np = F.cpu().numpy()

#граничные условия
r_nodes = torch.norm(nodes, dim=1)
boundary_mask = r_nodes >= R_ext - 1e-5
boundary_idx = torch.where(boundary_mask)[0].cpu().numpy().copy()

#формат LIL
G_sparse = G_sparse.tolil()
G_sparse[boundary_idx, :] = 0.0
G_sparse[boundary_idx, boundary_idx] = 1.0
G_sparse = G_sparse.tocsr()

F_np[boundary_idx] = 0.0

#решение системы (LU)
u_fem_np = spla.spsolve(G_sparse, F_np)
u_fem = torch.tensor(u_fem_np, dtype=torch.float32, device=device)

end = time.time()
print(f"Время: {end - start:.4f} сек")

u_true = u_analytic(r_nodes)

u_max_analytic = torch.max(torch.abs(u_true))
rel_err_fem = torch.mean(torch.abs(u_fem - u_true)) / u_max_analytic
print(f"Error: {rel_err_fem.item():.4e}")