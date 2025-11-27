from KPZ_class_NS import *

# Define the path
import os
base_dir = os.path.abspath(os.path.dirname(__file__))
plot_path = os.path.join(base_dir, '../Plots/KPZ_test_6/')
plot_path = os.path.abspath(plot_path)

# Parameters of the model#

t_min = 0.0
t_max = 5.0

x_min = -5
x_max = +5
number_of_epochs = 250000


# Check if directory exists, and if not, create it
if not os.path.exists(plot_path):
    os.makedirs(plot_path)
    print(f"Directory '{plot_path}' created.")
else:
    print(f"Directory '{plot_path}' already exists.")
# Initialize t_min, t_max, a_min and a_max
use_Latex = True

def generate_equations(n_heads):
  eq_dict = {} # Initialize the dictionary
  # Loop to create functions
  for head in range(n_heads):
      # Define a new function using a lambda or nested function
      def make_equations(head):
        def burguers(u,x,t,nu):
            term1 = diff(u,t)
            term2 = u* diff(u,x)
            term3 = -nu* diff(u,x,order = 2)
            term4 = -torch.tensor([data['Noise_grid']]).reshape_as(term1)
            return [term1 +term2 + term3 + term4]
        return burguers
      # Store the function in the dictionary
      eq_dict.update({f'equations_{head}':make_equations(head)})  # Now at the correct indentation level
  print('Equations dictionary generated')
  print(eq_dict)
  return eq_dict

# Generate the fixed noise and save it #
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Domain parameters
nx = 101
tmax = t_max
xmin = x_min
xmax = x_max
dx = (xmax - xmin) / (nx - 1)
nu = 0.1
sigma = 1.0
dt = sigma * (tmax)/nx

# Stochastic parameters
lam = 0.1       # lambda in front of noise term
D = 1e-1         # noise amplitude

# Initial condition
x = np.linspace(xmin, xmax, nx)
u = np.exp(-x**2 / 2)
u_initial = u.copy()
t_list = np.arange(0, tmax, dt)

# Derivative operator for noise (central diff)
def grad(f):
    return (np.roll(f, -1) - np.roll(f, 1)) / (2 * dx)

def burgers_step_stochastic(u, dt, dx, nu, lam, D):
    u_new = u.copy()
    
    # White noise η at this time step
    eta = np.random.randn(nx) * np.sqrt(2 * D / dt)  # eta ~ Gaussian
    eta_x = grad(eta)                                # ∂η/∂x
    noise_term = -lam * eta_x
    # Update using finite difference scheme
    for i in range(1, nx - 1):
        convection = -dt / dx * u[i] * (u[i] - u[i - 1])
        diffusion = nu * dt / dx**2 * (u[i + 1] - 2 * u[i] + u[i - 1])
        u_new[i] = u[i] + convection + diffusion + dt * noise_term[i]

    return u_new,noise_term

# Time loop
n = 0
Ugrid = []
Noise_grid = []
total_steps = int(tmax / dt)
pbar = tqdm(total=total_steps, desc="Solving Stochastic Burgers", ncols=100)
while n * dt < tmax:
    Ugrid.append(u.copy())
    u,noise = burgers_step_stochastic(u, dt, dx, nu, lam, D)
    Noise_grid.append(noise.copy())
    pbar.update(1)
    n += 1
    
    # Plot every 10%
    if (n / (total_steps + 1) * 100) % 10 == 0:
        plt.plot(x, u, label=f"t={n * dt:.2f}")
        plt.title(f"Stochastic Burgers | ν={nu}, λ={lam}")
        plt.xlabel("x")
        plt.ylabel("u(x,t)")
        plt.legend()
        plt.show()

data = {
    "Ugrid": Ugrid,
    "Noise_grid": Noise_grid,
    "x": x,
    "t": np.arange(0, tmax, dt),
    "nu": nu,
}

np.savetxt(plot_path + '/noise.txt',data['Noise_grid'])


#nu_min = -3
#nu_max = 0


# Create the generators #
nu_list = torch.logspace(-1, 0, 5)  # List of nu values for the PDEs
t_list = torch.linspace(t_min, t_max,nx)  # Time values
x_list = torch.linspace(x_min, x_max, nx)  # Spatial values
T,X,N = torch.meshgrid(t_list, x_list, nu_list, indexing='ij')  # Create a meshgrid for t, x, and nu
data['Noise_grid'],__ = np.meshgrid(data['Noise_grid'],nu_list.cpu().detach().numpy()
,indexing = 'ij')

train_generator = PredefinedGenerator(X,T,N)

init_cond = []
for i in range(1):
    init_cond.append([BundleIBVP1D(t_min=t_min, t_min_val=lambda x: init_gauss(x,0.5,1.0),
            x_min = x_min*2 ,x_min_val= lambda t: 0,
            x_max = x_max*2 ,x_max_val= lambda t: 0),
        BundleIBVP1D(t_min=t_min, t_min_val=lambda x: init_sine(x,1.0,1),
            x_min = x_min*2 ,x_min_val= lambda t: 0,
            x_max = x_max*2 ,x_max_val= lambda t: 0),
        BundleIBVP1D(t_min=t_min, t_min_val=lambda x: init_N_wave(x,1.0,1.0),
            x_min = x_min*2 ,x_min_val= lambda t: 0,
            x_max = x_max*2 ,x_max_val= lambda t: 0),
        BundleIBVP1D(t_min=t_min, t_min_val=lambda x: morlet_wavelet(x, f=0.25, sigma=1.25)+0.8,
            x_min = x_min*2 ,x_min_val= lambda t: 0,
            x_max = x_max*2 ,x_max_val= lambda t: 0)
    ])


init_cond = np.array(init_cond)
print(init_cond.shape)

n_heads = int(len(init_cond[0,:]))

body_units = [256,256,256,256,256,256]
basis_length = 32

# Create the body and the nets #
H = [FCNN(n_input_units=3, hidden_units= body_units, n_output_units = basis_length) for _ in range(1)]

heads = [[nn.Linear(in_features=basis_length,out_features=1,bias=False) for _ in range(n_heads)]for _ in 
         range(len(H))]

nets = np.ones([len(H),n_heads],dtype=nn.Module) # i -> equation, j -> head #
for i in range(len(H)):
    #print(i)
    for j in range(n_heads):
        nets[i,j] = NET(H[i],heads[i][j])

# Create the equations #
eq_dict = {}
eq_dict = generate_equations(n_heads = n_heads)
#print(self.eq_dict)
equation_list = []
for head in range(n_heads):
    equations = eq_dict[f'equations_{head}']
    #print(equations)
    equation_list.append(equations)

print('Equation list ready')

print(nets.shape)

#gen = Generator1D(48,t_min,t_max,'equally-spaced-noisy')

adam = torch.optim.Adam([ p for q in range(n_heads) for net in list(nets[:,q]) for p in net.parameters()],
                        lr=1e-3)

# Warm-up for 5 epochs (linear increase from 0 to 0.001)
warmup_scheduler = torch.optim.lr_scheduler.LinearLR(adam, start_factor=1e-3, total_iters=1000)
main_scheduler = torch.optim.lr_scheduler.StepLR(adam, step_size=1000, gamma=0.985)
scheduler = torch.optim.lr_scheduler.SequentialLR(adam, schedulers=[warmup_scheduler, main_scheduler], milestones=[1100])
scheduler_cb = DoSchedulerStep(scheduler=scheduler)

# Initialize the solver
solver = MHSolver2D(pde_system=equation_list,
                   conditions= init_cond,
                    all_nets= nets,
                    conditions_list = init_cond,
                    train_generator = train_generator,
                    valid_generator = None,
                    xy_min= (x_min,t_min), xy_max=(x_max,t_max),
                    eq_param_index = (0,),
                    n_batches_valid = 0,
                    n_batches_train = 1,
                    n_samplings =100,
                    optimizer = adam
                   )

# Ensure all networks are on the same device as your data (cuda:0)
device = torch.device("cuda:0")
for i in range(nets.shape[0]):
	for j in range(nets.shape[1]):
		nets[i, j] = nets[i, j].to(device)

solver.fit(max_epochs=number_of_epochs-solver.global_epoch, callbacks=[scheduler_cb],tqdm_file = None)
save_model(solver, plot_path, nu_list,body_units,basis_length)

# Plot the loss #
plt.figure()
plt.plot(solver.metrics_history['r2_loss'])
#plt.plot(solver.metrics_history['add_loss'])
plt.yscale('log')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.savefig(plot_path + '/loss.pdf')
#adam.param_groups[0]['lr'] = 9e-5
print('Learning rate ',adam.param_groups[0]['lr'])

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
# Show solution #
for head in range(solver.n_heads):
    solver.best_nets = solver.best_nets_list[head] # Fix the nets to the best nets #
    solver.conditions = solver.all_conditions[:,head] # Fix the conditions to the best conditions #
    sol = solver.get_solution(best= True)     # Get solution #
    x = torch.linspace(-5,5,1000)
    t = torch.linspace(0,5,1000)
    xx,tt = torch.meshgrid(x,t)
    nu = nu_list[0]*torch.ones_like(xx)
    u = sol(xx,tt,nu,to_numpy = True)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Create a surface plot
    surf = ax.plot_surface(xx.cpu().detach().numpy(),
                           tt.cpu().detach().numpy(),
                           u, 
                           cmap=cm.rainbow, edgecolor='none')

    # Add a color bar to show the height
    #fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
    ax.set_title('$\\nu$ = ' + str(nu[0,0].item()))
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_zlabel('u(x,t)')
    plt.savefig(plot_path + '/solution_nu_' + str(round(nu[0,0].item(),3))+ '.pdf')

# Show latent space basis #
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
for head in range(1):
    solver.best_nets = solver.best_nets_list[head] # Fix the nets to the best nets #
    solver.conditions = solver.all_conditions[:,head]
    x = torch.linspace(-5,5,500)
    t = torch.linspace(0,5,503)
    xx,tt = torch.meshgrid(x,t)
    xx = xx.unsqueeze(dim = 2)
    tt = tt.unsqueeze(dim = 2)
    nu = nu_list[0]*torch.ones_like(xx)
    omega = torch.stack([xx,tt,nu],dim = 2).squeeze()
    u = solver.best_nets[0].H_model(omega)

    xx = xx.squeeze()
    tt = tt.squeeze()
    # Create a surface plot
    for i in range(u.shape[2]):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(xx.cpu().detach().numpy(),
                               tt.cpu().detach().numpy(),
                               u[:,:,i].cpu().detach().numpy(), 
                               cmap=cm.rainbow, edgecolor='none')

    # Add a color bar to show the height
        ax.set_xlabel('x')
        ax.set_ylabel('t')
        ax.set_zlabel('u(x,t)')
        ax.set_title('$\gamma$ = ' + str(nu[0,0].item()))
        plt.savefig(plot_path + '/H_' + str(i) + '_nu_' + str(round(nu[0,0].item(),3))+ '.pdf')

# Show weights #
plt.figure()
for heads in range(solver.n_heads):
    h_weights = nets[0,heads].head_model.weight
    plt.plot(h_weights.cpu().detach().numpy()[0],label = 'Head ' + str(heads + 1) + ', $\gamma = $ ' + str(nu_list[0]),marker = '.')#,linestyle = 'none')
plt.legend()
plt.xlabel('Neuron')
plt.ylabel('Weights')
plt.savefig(plot_path+'/head_weight.pdf')
plt.show()

body_layers = nets[0,0].H_model.NN
for i in range(0,len(body_layers),2):
    plt.figure(figsize=(8,6))
    plt.imshow(body_layers[i].cpu().weight.detach().numpy(),cmap = 'rainbow',aspect='auto')
    if i==0:
        plt.xlim(0,2)
    plt.colorbar(label = 'Value')
    plt.xlabel('Neuron')
    plt.ylabel('Weight')
    plt.savefig(plot_path+'/body_weight_layer_' + str(i+1) + '.pdf')
    plt.show()

# Plot residuals #
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
# Show solution #
for head in range(solver.n_heads):
    solver.best_nets = solver.best_nets_list[head] # Fix the nets to the best nets #
    solver.conditions = solver.all_conditions[:,head]
    #sol = solver.get_residuals(xbest= True)    # Get solution #
    # Use the original generators for x, t, nu
    xx = X[:, :, :]
    tt = T[:, :, :]
    nu = N[:, :, :]
    u =  solver.get_residuals(xx,tt,nu,best= True)
    plt.figure()
    # Create a surface plot
    plt.pcolormesh(xx[:,:,0].cpu().detach().numpy(),tt[:,:,0].cpu().detach().numpy(),u[:,:,0].cpu().detach().numpy(),cmap = 'rainbow')

    # Add a color bar to show the height
    plt.colorbar(label = 'Residuals')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title('$\gamma$ = ' + str(nu[0,0,0].item()))
    plt.savefig(plot_path + '/res_nu_' + str(round(nu[0,0,0].item(),3))+ '.pdf')
    plt.show()

# Plot the latent space components #
# Ensure all networks are on the same device as your data (cuda:0)
device = torch.device("cuda:0")
for i in range(nets.shape[0]):
	for j in range(nets.shape[1]):
		nets[i, j] = nets[i, j].to(device)
xx = X[:, :, :]
tt = T[:, :, :]
nu = N[:, :, :]
xx = xx.unsqueeze(dim = 3)
tt = tt.unsqueeze(dim = 3)
nu = nu.unsqueeze(dim = 3)
data_latent = np.ones([xx.shape[0],xx.shape[1],xx.shape[2],basis_length+3])
data_latent[::,::,::,0] = xx.squeeze().cpu().detach().numpy()
data_latent[::,::,::,1] = tt.squeeze().cpu().detach().numpy()
data_latent[::,::,::,2] = nu.squeeze().cpu().detach().numpy()
for head in range(1):
    omega = torch.stack([xx, tt, nu], dim=3).squeeze()
    u = H[0](omega)
    data_latent[::, ::, ::, 3::] = u.squeeze().cpu().detach().numpy()
np.save(plot_path + '/data_latent.npy', data_latent)