import torch
import torch.nn as nn
import numpy as np
import time

N_f = 10000
N_bc = 500
epochs = 5000 

R_wire = 0.1
R_ext = 0.5
J = 1.0

epsilon = 0.005

def u_analytic(x, y):
    r = torch.sqrt(x**2 + y**2)

    sol_outside = (J * R_wire**2 / 2) * torch.log(R_ext / r)

    sol_inside = (J / 4) * (R_wire**2 - r**2) + (J * R_wire**2 / 2) * torch.log(torch.tensor(R_ext / R_wire))

    return torch.where(r <= R_wire, sol_inside, sol_outside)

torch.manual_seed(57)
np.random.seed(57)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Выбрано: {device}" )

history = {
    'total': [],
    'integral': [],
    'bc': []
}

class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_Tanh_stack = nn.Sequential(
            nn.Linear(2, 64), 
            nn.Tanh(),
            nn.Linear(64,64), 
            nn.Tanh(),
            nn.Linear(64,64), 
            nn.Tanh(),
            nn.Linear(64,64), 
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, x, y):
        x_y = torch.cat([x,y], dim=1)
        return self.linear_Tanh_stack(x_y)  
    
    # def forward(self, x, y):
    #     x_y = torch.cat([x, y], dim=1)
    #     nn_out = self.linear_Tanh_stack(x_y)
        
    #     boundary_area = R_ext**2 - (x**2 + y**2)
    #     return boundary_area * nn_out

def get_points(N_f, N_bc):

    #равномерное распределение по площади круга
    r = R_ext * torch.sqrt(torch.rand(N_f, 1, device=device))
    theta = 2 * np.pi * torch.rand(N_f, 1, device=device)

    x_f = (r * torch.cos(theta)).requires_grad_(True)
    y_f = (r * torch.sin(theta)).requires_grad_(True)

    theta_bc = 2 * np.pi * torch.rand(N_bc, 1, device=device)

    x_bc = R_ext * torch.cos(theta_bc)
    y_bc = R_ext * torch.sin(theta_bc)
    
    return x_f, y_f, x_bc, y_bc

def calculate_error(model, n_test=100):

    x = torch.linspace(-R_ext, R_ext, n_test, device=device)
    y = torch.linspace(-R_ext, R_ext, n_test, device=device)

    X, Y = torch.meshgrid(x, y, indexing='ij')

    x_flat = X.reshape(-1, 1)
    y_flat = Y.reshape(-1, 1)

    r_flat = torch.sqrt(x_flat**2 + y_flat**2)

    mask = r_flat <= R_ext

    x_test = x_flat[mask].reshape(-1, 1)
    y_test = y_flat[mask].reshape(-1, 1)

    u_pred = model(x_test, y_test)
    u_true = u_analytic(x_test, y_test)

    u_max = torch.max(torch.abs(u_true))
    
    return (torch.mean(torch.abs(u_pred - u_true)) / u_max).item()

x_f, y_f, x_bc, y_bc = get_points(N_f, N_bc)

def get_loss():

    u_f = model(x_f, y_f)

    u_x = torch.autograd.grad(u_f, x_f, grad_outputs=torch.ones_like(u_f), create_graph=True)[0]
    u_y = torch.autograd.grad(u_f, y_f, grad_outputs=torch.ones_like(u_f), create_graph=True)[0]
    
    r_f = torch.sqrt(x_f**2 + y_f**2)
    f_true = J * torch.sigmoid((R_wire - r_f) / epsilon)
    
    #энергетический функционал
    energy = 0.5 * (u_x**2 + u_y**2) - f_true * u_f
    loss_integral = torch.mean(energy) * (np.pi * R_ext**2)

    u_bc_pred = model(x_bc, y_bc)
    loss_bc = torch.mean(u_bc_pred**2)

    total_loss = loss_integral + 100 * loss_bc

    return total_loss, loss_integral, loss_bc

start = time.time()

model = PINN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3000, gamma=0.5)

for epoch in range(epochs):

    optimizer.zero_grad()

    loss, loss_integral, loss_bc = get_loss()
    loss.backward()

    optimizer.step()
    scheduler.step()

    history['total'].append(loss.item())
    history['integral'].append(loss_integral.item())
    history['bc'].append(loss_bc.item())

    if epoch % 1000 == 0:
        error = calculate_error(model)
        print(f"Epoch {epoch:5d} | Loss: {loss.item():.4e} | Loss_integral: {loss_integral.item():.4e} | Loss_bc: {loss_bc.item():.4e} | Error: {error:.4f}")

optimizer_lbfgs = torch.optim.LBFGS(model.parameters(), max_iter=1000, line_search_fn="strong_wolfe")

def lbgs():
    optimizer_lbfgs.zero_grad()
    loss, _, _ = get_loss()
    loss.backward()
    return loss

optimizer_lbfgs.step(lbgs)

end = time.time()
print(f"\nTime: {end - start:.4f} сек")
print(f"Final error: {calculate_error(model):.4e}")

torch.save(history, "train_history.pt")
torch.save(model.state_dict(), "pinn_poisson.pth")
