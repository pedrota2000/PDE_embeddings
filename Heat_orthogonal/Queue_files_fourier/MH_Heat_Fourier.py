from MHBundleSolver2D_heat_orthogonal import *

# Define the path
import os
base_dir = os.path.abspath(os.path.dirname(__file__))
plot_path = os.path.join(base_dir, '../Plots/Heat_25NU_Fourier_Orthogonal/')
plot_path = os.path.abspath(plot_path)


# Check if directory exists, and if not, create it
if not os.path.exists(plot_path):
    os.makedirs(plot_path)
    print(f"Directory '{plot_path}' created.")
else:
    print(f"Directory '{plot_path}' already exists.")
# Initialize t_min, t_max, a_min and a_max
use_Latex = True
# Parameters of the model#

t_min = 0.0
t_max = 5.0

x_min = -5
x_max = +5
number_of_epochs = 300_000


#nu_min = -3
#nu_max = 0
# Create the generators #
nu_list = torch.logspace(-2, 0, 25)  # List of nu values for the PDEs
t_list = torch.linspace(t_min, t_max,100)  # Time values
x_list = torch.linspace(x_min, x_max, 100)  # Spatial values
T,X,N = torch.meshgrid(t_list, x_list, nu_list, indexing='ij')  # Create a meshgrid for t, x, and nu

train_generator = PredefinedGenerator(X,T,N)

init_cond = []
for i in range(1):
    init_cond.append([BundleIBVP1D(t_min=t_min, t_min_val=lambda x: init_sine(x,0.5,1.0),
            x_min = x_min*2 ,x_min_val= lambda t: 0,
            x_max = x_max*2 ,x_max_val= lambda t: 0),
        BundleIBVP1D(t_min=t_min, t_min_val=lambda x: init_sine(x,1.0,1),
            x_min = x_min*2 ,x_min_val= lambda t: 0,
            x_max = x_max*2 ,x_max_val= lambda t: 0),
        BundleIBVP1D(t_min=t_min, t_min_val=lambda x: init_sine(x,1.0,2.0),
            x_min = x_min*2 ,x_min_val= lambda t: 0,
            x_max = x_max*2 ,x_max_val= lambda t: 0),
        BundleIBVP1D(t_min=t_min, t_min_val=lambda x: init_cos_norm(x,0.5,1.0),
            x_min = x_min*2 ,x_min_val= lambda t: 0,
            x_max = x_max*2 ,x_max_val= lambda t: 0),
        BundleIBVP1D(t_min=t_min, t_min_val=lambda x: init_cos_norm(x,1.0,1.0),
            x_min = x_min*2 ,x_min_val= lambda t: 0,
            x_max = x_max*2 ,x_max_val= lambda t: 0),
        BundleIBVP1D(t_min=t_min, t_min_val=lambda x: init_cos_norm(x,1.0,2.0),
            x_min = x_min*2 ,x_min_val= lambda t: 0,
            x_max = x_max*2 ,x_max_val= lambda t: 0)
    ])


init_cond = np.array(init_cond)
print(init_cond.shape)

n_heads = int(len(init_cond[0,:]))

body_units = [128, 128, 128, 128, 128]
basis_length = n_heads

# Create the body and the nets #
H = [FCNN(n_input_units=3, hidden_units= body_units, n_output_units = basis_length) for _ in range(1)]

heads = [[nn.Linear(in_features=basis_length,out_features=1,bias=False) for _ in range(n_heads)]for _ in 
         range(len(H))]

nets = np.ones([len(H),n_heads],dtype=nn.Module) # i -> equation, j -> head #
for i in range(len(H)):
    #print(i)
    for j in range(n_heads):
        nets[i,j] = NET(H[i],heads[i][j])
print('Computing orthogonality')
calc_weights_orthogonality(nets)
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
plt.plot(solver.metrics_history['add_loss'])
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
    # Select 5 uniformly spaced indices from nu_list
    nu_indices = np.linspace(0, len(nu_list) - 1, 5, dtype=int)
    for idx in nu_indices:
        nu_val = nu_list[idx]
        nu = nu_val * torch.ones_like(xx)
        u = sol(xx, tt, nu, to_numpy=True)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Create a surface plot
        surf = ax.plot_surface(xx.cpu().detach().numpy(),
                               tt.cpu().detach().numpy(),
                               u,
                               cmap=cm.rainbow, edgecolor='none')

        ax.set_title('$\\nu$ = ' + str(round(nu_val.item(), 3)))
        ax.set_xlabel('x')
        ax.set_ylabel('t')
        ax.set_zlabel('u(x,t)')
        plt.savefig(plot_path + f'/solution_head_{head}_nu_{round(nu_val.item(),3)}.pdf')
        plt.close(fig)

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
    # Select 5 uniformly spaced indices from nu_list
    nu_indices = np.linspace(0, len(nu_list) - 1, 5, dtype=int)
    for idx in nu_indices:
        nu_val = nu_list[idx]
        nu = nu_val * torch.ones_like(xx)
        omega = torch.stack([xx, tt, nu], dim=2).squeeze()
        u = solver.best_nets[0].H_model(omega)

        xx_s = xx.squeeze()
        tt_s = tt.squeeze()
        # Create a surface plot for each latent basis component
        for i in range(u.shape[2]):
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            surf = ax.plot_surface(xx_s.cpu().detach().numpy(),
                                   tt_s.cpu().detach().numpy(),
                                   u[:, :, i].cpu().detach().numpy(),
                                   cmap=cm.rainbow, edgecolor='none')

            ax.set_xlabel('x')
            ax.set_ylabel('t')
            ax.set_zlabel('u(x,t)')
            ax.set_title('$\\nu$ = ' + str(round(nu_val.item(), 3)))
            plt.savefig(plot_path + f'/H_{i}_nu_{round(nu_val.item(),3)}.pdf')
            plt.close(fig)

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
    x = torch.linspace(-5,5,200)
    t = torch.linspace(0,5,200)
    xx,tt = torch.meshgrid(x,t)
    # Select 5 uniformly spaced indices from nu_list
    nu_indices = np.linspace(0, len(nu_list) - 1, 5, dtype=int)
    for idx in nu_indices:
        nu_val = nu_list[idx]
        nu = nu_val * torch.ones_like(xx)
        u = solver.get_residuals(xx, tt, nu, best=True)
        plt.figure()
        plt.pcolormesh(xx.cpu().detach().numpy(), tt.cpu().detach().numpy(), u.cpu().detach().numpy(), cmap='rainbow')
        plt.colorbar(label='Residuals')
        plt.xlabel('x')
        plt.ylabel('t')
        plt.title('$\\gamma$ = ' + str(round(nu_val.item(), 3)))
        plt.savefig(plot_path + f'/res_head_{head}_nu_{round(nu_val.item(),3)}.pdf')
        plt.close()
    plt.show()

# Plot the latent space components #
# Ensure all networks are on the same device as your data (cuda:0)
device = torch.device("cuda:0")
for i in range(nets.shape[0]):
	for j in range(nets.shape[1]):
		nets[i, j] = nets[i, j].to(device)
x = torch.linspace(-5,5,101)
t = torch.linspace(0,5,103)
nu = nu_list
xx,tt,nunu = torch.meshgrid(x,t,nu,indexing = 'ij')
xx = xx.unsqueeze(dim = 3)
tt = tt.unsqueeze(dim = 3)
nunu = nunu.unsqueeze(dim = 3)
data_latent = np.ones([xx.shape[0],xx.shape[1],xx.shape[2],basis_length+3])
data_latent[::,::,::,0] = xx.squeeze(dim = 3).cpu().detach().numpy()
data_latent[::,::,::,1] = tt.squeeze(dim = 3).cpu().detach().numpy()
data_latent[::,::,::,2] = nunu.squeeze(dim = 3).cpu().detach().numpy()
for head in range(1):
    omega = torch.stack([xx, tt, nunu], dim=3).squeeze(dim = 4)
    print(omega.shape)
    u = H[0](omega)
    print(u.shape)
    data_latent[::, ::, ::, 3::] = u.squeeze(dim = 3).cpu().detach().numpy()
np.save(plot_path + '/data_latent.npy', data_latent)