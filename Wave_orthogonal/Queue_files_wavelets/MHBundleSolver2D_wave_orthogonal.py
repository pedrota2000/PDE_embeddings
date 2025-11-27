import torch
import numpy as np
from tqdm.notebook import tqdm
from scipy.integrate import odeint

import neurodiffeq
from neurodiffeq import diff
from neurodiffeq.conditions import IVP, DirichletBVP, DirichletBVP2D, BundleIVP, NoCondition, BundleDirichletBVP
from neurodiffeq.solvers import *
from neurodiffeq.networks import FCNN, SinActv
from neurodiffeq.monitors import Monitor1D
from neurodiffeq.generators import Generator1D, Generator2D, PredefinedGenerator, BaseGenerator, PredefinedGenerator, MeshGenerator
from neurodiffeq.callbacks import ActionCallback
from neurodiffeq import diff

import copy as copy
import cmath as cmath
import types

def _requires_closure(optimizer):
    # starting from torch v1.13, simple optimizers no longer have a `closure` argument
    closure_param = inspect.signature(optimizer.step).parameters.get('closure')
    return closure_param and closure_param.default == inspect._empty

# BundleIBVP1D #
class BundleIBVP1D(BaseCondition):
    r"""An initial & boundary condition on a 1-D range where :math:`x\in[x_0, x_1]` and time starts at :math:`t_0`.
    The conditions should have the following parts:

    - :math:`u(x,t_0)=u_0(x)`,
    - :math:`u(x_0,t)=g(t)` or :math:`u'_x(x_0,t)=p(t)`,
    - :math:`u(x_1,t)=h(t)` or :math:`u'_x(x_1,t)=q(t)`,

    where :math:`\displaystyle u'_x=\frac{\partial u}{\partial x}`.

    :param x_min: The lower bound of x, the :math:`x_0`.
    :type x_min: float
    :param x_max: The upper bound of x, the :math:`x_1`.
    :type x_max: float
    :param t_min: The initial time, the :math:`t_0`.
    :type t_min: float
    :param t_min_val: The initial condition, the :math:`u_0(x)`.
    :type t_min_val: callable
    :param x_min_val: The Dirichlet boundary condition when :math:`x = x_0`, the :math:`u(x_0, t)`, defaults to None.
    :type x_min_val: callable, optional
    :param x_min_prime: The Neumann boundary condition when :math:`x = x_0`, the :math:`u'_x(x_0, t)`, defaults to None.
    :type x_min_prime: callable, optional
    :param x_max_val: The Dirichlet boundary condition when :math:`x = x_1`, the :math:`u(x_1, t)`, defaults to None.
    :type x_max_val: callable, optional
    :param x_max_prime: The Neumann boundary condition when :math:`x = x_1`, the :math:`u'_x(x_1, t)`, defaults to None.
    :type x_max_prime: callable, optional
    :raises NotImplementedError: When unimplemented boundary conditions are configured.

    .. note::
        This condition cannot be passed to ``neurodiffeq.conditions.EnsembleCondition`` unless both boundaries uses
        Dirichlet conditions (by specifying only ``x_min_val`` and ``x_max_val``) and ``force`` is set to True in
        EnsembleCondition's constructor.
    """

    def __init__(
            self, x_min, x_max, t_min, t_min_val,
            x_min_val=None, x_min_prime=None,
            x_max_val=None, x_max_prime=None,
    ):
        super().__init__()
        n_conditions = sum(c is not None for c in [x_min_val, x_min_prime, x_max_val, x_max_prime])
        if n_conditions != 2 or (x_min_val and x_min_prime) or (x_max_val and x_max_prime):
            raise NotImplementedError('Sorry, this boundary condition is not implemented.')
        self.x_min, self.x_min_val, self.x_min_prime = x_min, x_min_val, x_min_prime
        self.x_max, self.x_max_val, self.x_max_prime = x_max, x_max_val, x_max_prime
        self.t_min, self.t_min_val = t_min, t_min_val

    def enforce(self, net, *coordinates):
        r"""Enforces this condition on a network with inputs `x` and `t`

        :param net: The network whose output is to be re-parameterized.
        :type net: `torch.nn.Module`
        :param x: The :math:`x`-coordinates of the samples; i.e., the spatial coordinates.
        :type x: `torch.Tensor`
        :param t: The :math:`t`-coordinates of the samples; i.e., the temporal coordinates.
        :type t: `torch.Tensor`
        :return: The re-parameterized output, where the condition is automatically satisfied.
        :rtype: `torch.Tensor`

        .. note::
            This method overrides the default method of ``neurodiffeq.conditions.BaseCondition`` .
            In general, you should avoid overriding ``enforce`` when implementing custom boundary conditions.
        """
        x = coordinates[0]
        t = coordinates[1]
        

        def ANN(coordinates):
            #out = net(torch.cat(coordinates, dim=1))
            out = net(torch.stack(coordinates,dim = 0).squeeze().t())
            if self.ith_unit is not None:
                out = out[:, self.ith_unit].view(-1, 1)
            return out

        uxt = ANN(coordinates)
        if self.x_min_val and self.x_max_val:
            return self.parameterize(uxt, x, t)
        elif self.x_min_val and self.x_max_prime:
            x1 = self.x_max * torch.ones_like(x, requires_grad=True)
            ux1t = ANN(x1, t)
            return self.parameterize(uxt, x, t, ux1t, x1)
        elif self.x_min_prime and self.x_max_val:
            x0 = self.x_min * torch.ones_like(x, requires_grad=True)
            ux0t = ANN(x0, t)
            return self.parameterize(uxt, x, t, ux0t, x0)
        elif self.x_min_prime and self.x_max_prime:
            x0 = self.x_min * torch.ones_like(x, requires_grad=True)
            x1 = self.x_max * torch.ones_like(x, requires_grad=True)
            ux0t = ANN(x0, t)
            ux1t = ANN(x1, t)
            return self.parameterize(uxt, x, t, ux0t, x0, ux1t, x1)
        else:
            raise NotImplementedError('Sorry, this boundary condition is not implemented.')

    def parameterize(self, u, x, t, *additional_tensors):
        r"""Re-parameterizes outputs such that the initial and boundary conditions are satisfied.

        The Initial condition is always :math:`u(x,t_0)=u_0(x)`. There are four boundary conditions that are
        currently implemented:

        - For Dirichlet-Dirichlet boundary condition :math:`u(x_0,t)=g(t)` and :math:`u(x_1,t)=h(t)`:

          The re-parameterization is
          :math:`\displaystyle u(x,t)=A(x,t)+\tilde{x}\big(1-\tilde{x}\big)\Big(1-e^{-\tilde{t}}\Big)\mathrm{ANN}(x,t)`,
          where :math:`\displaystyle A(x,t)=u_0(x)+
          \tilde{x}\big(h(t)-h(t_0)\big)+\big(1-\tilde{x}\big)\big(g(t)-g(t_0)\big)`.

        - For Dirichlet-Neumann boundary condition :math:`u(x_0,t)=g(t)` and :math:`u'_x(x_1, t)=q(t)`:

          The re-parameterization is
          :math:`\displaystyle u(x,t)=A(x,t)+\tilde{x}\Big(1-e^{-\tilde{t}}\Big)
          \Big(\mathrm{ANN}(x,t)-\big(x_1-x_0\big)\mathrm{ANN}'_x(x_1,t)-\mathrm{ANN}(x_1,t)\Big)`,
          where :math:`\displaystyle A(x,t)=u_0(x)+\big(x-x_0\big)\big(q(t)-q(t_0)\big)+\big(g(t)-g(t_0)\big)`.

        - For Neumann-Dirichlet boundary condition :math:`u'_x(x_0,t)=p(t)` and :math:`u(x_1, t)=h(t)`:

          The re-parameterization is
          :math:`\displaystyle u(x,t)=A(x,t)+\big(1-\tilde{x}\big)\Big(1-e^{-\tilde{t}}\Big)
          \Big(\mathrm{ANN}(x,t)-\big(x_1-x_0\big)\mathrm{ANN}'_x(x_0,t)-\mathrm{ANN}(x_0,t)\Big)`,
          where :math:`\displaystyle A(x,t)=u_0(x)+\big(x_1-x\big)\big(p(t)-p(t_0)\big)+\big(h(t)-h(t_0)\big)`.

        - For Neumann-Neumann boundary condition :math:`u'_x(x_0,t)=p(t)` and :math:`u'_x(x_1, t)=q(t)`

          The re-parameterization is
          :math:`\displaystyle u(x,t)=A(x,t)+\left(1-e^{-\tilde{t}}\right)
          \Big(
          \mathrm{ANN}(x,t)-\big(x-x_0\big)\mathrm{ANN}'_x(x_0,t)
          +\frac{1}{2}\tilde{x}^2\big(x_1-x_0\big)
          \big(\mathrm{ANN}'_x(x_0,t)-\mathrm{ANN}'_x(x_1,t)\big)
          \Big)`,
          where :math:`\displaystyle A(x,t)=u_0(x)
          -\frac{1}{2}\big(1-\tilde{x}\big)^2\big(x_1-x_0\big)\big(p(t)-p(t_0)\big)
          +\frac{1}{2}\tilde{x}^2\big(x_1-x_0\big)\big(q(t)-q(t_0)\big)`.

        Notations:

        - :math:`\displaystyle\tilde{t}=\frac{t-t_0}{t_1-t_0}`,
        - :math:`\displaystyle\tilde{x}=\frac{x-x_0}{x_1-x_0}`,
        - :math:`\displaystyle\mathrm{ANN}` is the neural network,
        - and :math:`\displaystyle\mathrm{ANN}'_x=\frac{\partial ANN}{\partial x}`.

        :param output_tensor: Output of the neural network.
        :type output_tensor: `torch.Tensor`
        :param x: The :math:`x`-coordinates of the samples; i.e., the spatial coordinates.
        :type x: `torch.Tensor`
        :param t: The :math:`t`-coordinates of the samples; i.e., the temporal coordinates.
        :type t: `torch.Tensor`
        :param additional_tensors: additional tensors that will be passed by ``enforce``
        :type additional_tensors: `torch.Tensor`
        :return: The re-parameterized output of the network.
        :rtype: `torch.Tensor`
        """

        t0 = self.t_min * torch.ones_like(t, requires_grad=True)
        x_tilde = (x - self.x_min) / (self.x_max - self.x_min)
        t_tilde = t - self.t_min

        if self.x_min_val and self.x_max_val:
            return self._parameterize_dd(u, x, t, x_tilde, t_tilde, t0)
        elif self.x_min_val and self.x_max_prime:
            return self._parameterize_dn(u, x, t, x_tilde, t_tilde, t0, *additional_tensors)
        elif self.x_min_prime and self.x_max_val:
            return self._parameterize_nd(u, x, t, x_tilde, t_tilde, t0, *additional_tensors)
        elif self.x_min_prime and self.x_max_prime:
            return self._parameterize_nn(u, x, t, x_tilde, t_tilde, t0, *additional_tensors)
        else:
            raise NotImplementedError('Sorry, this boundary condition is not implemented.')

    # When we have Dirichlet boundary conditions on both ends of the domain:
    def _parameterize_dd(self, uxt, x, t, x_tilde, t_tilde, t0):
        Axt = self.t_min_val(x) + \
              x_tilde * (self.x_max_val(t) - self.x_max_val(t0)) + \
              (1 - x_tilde) * (self.x_min_val(t) - self.x_min_val(t0))
        return Axt + x_tilde * (1 - x_tilde) * (1 - torch.exp(-t_tilde)) * uxt

    # When we have Dirichlet boundary condition on the left end of the domain
    # and Neumann boundary condition on the right end of the domain:
    def _parameterize_dn(self, uxt, x, t, x_tilde, t_tilde, t0, ux1t, x1):
        Axt = (self.x_min_val(t) - self.x_min_val(t0)) + self.t_min_val(x) + \
              x_tilde * (self.x_max - self.x_min) * (self.x_max_prime(t) - self.x_max_prime(t0))
        return Axt + x_tilde * (1 - torch.exp(-t_tilde)) * (
                uxt - (self.x_max - self.x_min) * diff(ux1t, x1) - ux1t
        )

    # When we have Neumann boundary condition on the left end of the domain
    # and Dirichlet boundary condition on the right end of the domain:
    def _parameterize_nd(self, uxt, x, t, x_tilde, t_tilde, t0, ux0t, x0):
        Axt = (self.x_max_val(t) - self.x_max_val(t0)) + self.t_min_val(x) + \
              (x_tilde - 1) * (self.x_max - self.x_min) * (self.x_min_prime(t) - self.x_min_prime(t0))
        return Axt + (1 - x_tilde) * (1 - torch.exp(-t_tilde)) * (
                uxt + (self.x_max - self.x_min) * diff(ux0t, x0) - ux0t
        )

    # When we have Neumann boundary conditions on both ends of the domain:
    def _parameterize_nn(self, uxt, x, t, x_tilde, t_tilde, t0, ux0t, x0, ux1t, x1):
        Axt = self.t_min_val(x) \
              - 0.5 * (1 - x_tilde) ** 2 * (self.x_max - self.x_min) * (self.x_min_prime(t) - self.x_min_prime(t0)) \
              + 0.5 * x_tilde ** 2 * (self.x_max - self.x_min) * (self.x_max_prime(t) - self.x_max_prime(t0))
        return Axt + (1 - torch.exp(-t_tilde)) * (
                uxt
                - x_tilde * (self.x_max - self.x_min) * diff(ux0t, x0)
                + 0.5 * x_tilde ** 2 * (self.x_max - self.x_min) * (
                        diff(ux0t, x0) - diff(ux1t, x1)
                )
        )

##################

# Define weights matrix times its transpose calculation given nets[i,j] #
def calc_weights_orthogonality(nets):
    __,n_heads = nets.shape
    weights_matrix = torch.zeros(n_heads,n_heads)
    for j in range(n_heads):
        # Extract weights from the first layer of head_model (assumed linear)
        # Directly extract weights from the first layer of head_model (assumed to be nn.Linear)
        weights_matrix[:, j] = nets[0, j].head_model.weight.squeeze()
    orth_condition = torch.mm(weights_matrix.t(),weights_matrix)- torch.eye(n_heads)
    orth_condition_reverse = torch.mm(weights_matrix,weights_matrix.t())- torch.eye(n_heads)
    add_loss = torch.sum(orth_condition**2) + torch.sum(orth_condition_reverse**2)
    return add_loss

class SolutionBundle2D(BaseSolution):
    def _compute_u(self, net, condition, *coordinates):
        return condition.enforce(net, *coordinates)


class MHSolver2D(Solver2D):
    r"""A solver class for solving ODEs (single-input differential equations)
    , or a bundle of ODEs for different values of its parameters and/or conditions

    :param ode_system:
        The ODE system to solve, which maps a torch.Tensor or a tuple of torch.Tensors, to a tuple of ODE residuals,
        both the input and output must have shape (n_samples, 1).
    :type ode_system: callable
    :param conditions:
        List of conditions for each target function.
    :type conditions: list[`neurodiffeq.conditions.BaseCondition`]
    :param t_min:
        Lower bound of input (start time).
        Ignored if ``train_generator`` and ``valid_generator`` are both set.
    :type t_min: float, optional
    :param t_max:
        Upper bound of input (start time).
        Ignored if ``train_generator`` and ``valid_generator`` are both set.
    :type t_max: float, optional
    :param theta_min:
        Lower bound of input (parameters and/or conditions). If conditions are included in the bundle,
        the order should match the one inferred by the values of the ``bundle_param_lookup`` input
        in the ``neurodiffeq.conditions.BundleIVP``.
        Defaults to None.
        Ignored if ``train_generator`` and ``valid_generator`` are both set.
    :type theta_min: float or tuple, optional
    :param theta_max:
        Upper bound of input (parameters and/or conditions). If conditions are included in the bundle,
        the order should match the one inferred by the values of the ``bundle_param_lookup`` input
        in the ``neurodiffeq.conditions.BundleIVP``.
        Defaults to None.
        Ignored if ``train_generator`` and ``valid_generator`` are both set.
    :type theta_max: float or tuple, optional
    :param eq_param_index:
        Index (or indices) of bundle parameter that appears in the equation.
        E.g., if there are 5 bundle parameters generated and the first (index 0) and last (index 4) parameters are used
        in the equation, then eq_param_index should be (0, 4).
        The signature of the original equation should have, in addition to functions to solve for and coordinates,
        2 more parameters, corresponding to bundle parameters indexed at 0 and 4, in that order.
    :type eq_param_index: int or tuple[int], optional
    :param nets:
        List of neural networks for parameterized solution.
        If provided, length of ``nets`` must equal that of ``conditions``
    :type nets: list[torch.nn.Module], optional
    :param train_generator:
        Generator for sampling training points,
        which must provide a ``.get_examples()`` method and a ``.size`` field.
        ``train_generator`` must be specified if ``t_min`` and ``t_max`` are not set.
        If provided, the generator must generate bundle parameters too.
    :type train_generator: `neurodiffeq.generators.BaseGenerator`, optional
    :param valid_generator:
        Generator for sampling validation points,
        which must provide a ``.get_examples()`` method and a ``.size`` field.
        ``valid_generator`` must be specified if ``t_min`` and ``t_max`` are not set.
        If provided, the generator must generate bundle parameters too.
    :type valid_generator: `neurodiffeq.generators.BaseGenerator`, optional
    :param analytic_solutions:
        Analytical solutions to be compared with neural net solutions.
        It maps a torch.Tensor to a tuple of function values.
        Output shape should match that of ``nets``.
    :type analytic_solutions: callable, optional
    :param optimizer:
        Optimizer to be used for training.
        Defaults to a ``torch.optim.Adam`` instance that trains on all parameters of ``nets``.
    :type optimizer: ``torch.nn.optim.Optimizer``, optional
    :param loss_fn:
        The loss function used for training.

        - If a str, must be present in the keys of `neurodiffeq.losses._losses`.
        - If a `torch.nn.modules.loss._Loss` instance, just pass the instance.
        - If any other callable, it must map
          A) a residual tensor (shape `(n_points, n_equations)`),
          B) a function values tuple (length `n_funcs`, each element a tensor of shape `(n_points, 1)`), and
          C) a coordinate values tuple (length `n_coords`, each element a tensor of shape `(n_coords, 1)`
          to a tensor of empty shape (i.e. a scalar). The returned tensor must be connected to the computational graph,
          so that backpropagation can be performed.

    :type loss_fn:
        str or `torch.nn.moduesl.loss._Loss` or callable
    :param n_batches_train:
        Number of batches to train in every epoch, where batch-size equals ``train_generator.size``.
        Defaults to 1.
    :type n_batches_train: int, optional
    :param n_batches_valid:
        Number of batches to validate in every epoch, where batch-size equals ``valid_generator.size``.
        Defaults to 4.
    :type n_batches_valid: int, optional
    :param metrics:
        Additional metrics to be logged (besides loss). ``metrics`` should be a dict where

        - Keys are metric names (e.g. 'analytic_mse');
        - Values are functions (callables) that computes the metric value.
          These functions must accept the same input as the differential equation ``ode_system``.

    :type metrics: dict[str, callable], optional
    :param n_output_units:
        Number of output units for each neural network.
        Ignored if ``nets`` is specified.
        Defaults to 1.
    :type n_output_units: int, optional
    :param batch_size:
        **[DEPRECATED and IGNORED]**
        Each batch will use all samples generated.
        Please specify ``n_batches_train`` and ``n_batches_valid`` instead.
    :type batch_size: int
    :param shuffle:
        **[DEPRECATED and IGNORED]**
        Shuffling should be performed by generators.
    :type shuffle: bool
    """

    #def __init__(self,  pde_system, conditions, nu_list, all_nets,
    #             xy_min=None, xy_max=None, method = 'equally-spaced-noisy', n_samplings = 32,
    #             theta_min=None, theta_max=None, eq_param_index=(),
    #             nets=None, train_generator=None, valid_generator=None, analytic_solutions=None, optimizer=None,
    #             loss_fn=None, n_batches_train=1, n_batches_valid=4, metrics=None, n_output_units=1,
                 # deprecated arguments are listed below
    #             batch_size=None, shuffle=None,*args,**kwargs):
    def __init__(self,  pde_system, conditions_list, all_nets,theta_min=None, theta_max=None, eq_param_index=(),
                 n_samplings = 32,method = 'equally-spaced-noisy', *args, **kwargs):
        
        
        # Pop and set all kwargs as attributes of the instance
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        self.equations = pde_system
        

        #if train_generator is None or valid_generator is None:
        #    if xy_min is None or xy_max is None:
        #        raise ValueError(f"Either generator is not provided, xy_min and xy_max should be both provided: \n"
        #                         f"got xy_min={xy_min}, xy_max={xy_max}, "
        #                         f"train_generator={train_generator}, valid_generator={valid_generator}")

        #if isinstance(theta_min, (float, int)):
        #    theta_min = (theta_min,)
        #elif theta_min is None:
        #    theta_min = ()

        #if isinstance(theta_max, (float, int)):
        #    theta_max = (theta_max,)
        #elif theta_max is None:
        #    theta_max = ()

        #if len(theta_min) != len(theta_max):
        #    raise ValueError(
        #        f"length of theta_min and theta_max must be equal, " f"got {len(theta_min)} != {len(theta_max)}"
        #    )

        #r_min = self.xy_min + tuple(theta_min)
        #r_max = self.xy_max + tuple(theta_max)

        #n_input_units = len(r_min)

#         if train_generator is None:
#             train_generator = Generator2D((32, 32), xy_min=xy_min, xy_max=xy_max, method='equally-spaced-noisy')
#             for i in range(n_input_units - 1):
#                 train_generator ^= Generator1D(32, t_min=r_min[i + 1], t_max=r_max[i + 1],
#                                                method='equally-spaced-noisy')
#         if valid_generator is None:
#             valid_generator = Generator2D((32, 32), xy_min=xy_min, xy_max=xy_max, method='equally-spaced')
#             for i in range(n_input_units - 1):
#                 valid_generator ^= Generator1D(32, t_min=r_min[i + 1], t_max=r_max[i + 1], method='equally-spaced')

        #self.r_min, self.r_max = r_min, r_max

        # number of functions equals number of conditions supplied
        self.all_conditions = conditions_list
        self.conditions = self.all_conditions[:,0]
        N_FUNCTIONS = len(self.conditions)
        # there is only 1 coordinate (usually time `t`) for ODEs
        N_COORDS = 2      # We will change this from 1 to 2

        # Note: It is intentionally design in this way where `eq_param_index` and `self.eq_param_index`
        # both contain values offset by `N_FUNCTIONS + N_COORDS`. The first one (not bound to `self`) is used for the
        # `eq_param_filter` closure, while the second one (bound to `self`) is used for `_get_internal_variables()`.
        eq_param_index = tuple(N_FUNCTIONS + N_COORDS + idx for idx in eq_param_index)
        self.eq_param_index = eq_param_index
        
        def _diff_eqs_wrapper(*variables):
            funcs_and_coords = variables[:N_FUNCTIONS + N_COORDS]
            eq_params = tuple(variables[idx] for idx in eq_param_index)
            return pde_system(*funcs_and_coords, *eq_params)


        # TODO: check and warn if there are variables neither used by conditions nor by equations
        
        # Prepare the variables for the Multihead #
        self.n_heads = len(self.all_conditions[0,:])  # Define the number of heads #
        self.best_nets_list = np.ones(self.n_heads,dtype = object)
        self.all_nets = all_nets            # Define all the nets to iterate #
        self.pde_list = []
        for head in range(self.n_heads):  # Define the generators and append them on the list #
        #     nu_elem = torch.tensor(nu_list[head])
        #     # Define the generator list to iterate #
        #     if self.train_generator is None:
        #         xy_gen = Generator2D((n_samplings, n_samplings), xy_min=self.xy_min, xy_max=self.xy_max, method=method)
        #     pg = PredefinedGenerator(nu_elem*torch.ones(n_samplings**2))
        #     self.all_generators.append(xy_gen * pg)
             print('head',head)
        #     #ode_system = kwargs.pop('ode_system')
             pde_system = self.equations[head]
             print('Inside solver __init()__:', pde_system)
             super().__init__(pde_system=pde_system,*args,**kwargs)
             self.pde_list.append(self.diff_eqs)
        

    #self.xy_min, self.xy_max = xy_min, xy_max  # Maybe this line is not necessary since it is in r_min & r_max (I think)

       # super(MHSolver2D, self).__init__(
       #     pde_system=self.pde_list,
       #     nu_list=nu_list,
       #     all_nets = all_nets,
       #     theta_min = theta_min, theta_max = theta_max, eq_param_index=eq_param_index,
       #          n_samplings = n_samplings,method = method, *args, **kwargs
            
       # )

        self.metrics_history['r2_loss'] = []
        self.metrics_history['add_loss'] = []
    def GA_loss_fn(self, r,f,x):
        # Compute the derivative of the function wrt to x #
        ux = diff(f[0],x[0])
        alpha = 2
        beta = 5 
        Landa = 1/(1+alpha *torch.abs(ux)**beta)
        loss = Landa * (r**2) 
        return torch.mean(loss)

    def additional_loss(self,r,f,x):
        # Add ortogonality relation #
        add_loss = calc_weights_orthogonality(self.all_nets)
        return 1e-3*add_loss
            


    def custom_epoch(self, key):
        r"""Run an epoch on train/valid points, update history, and perform an optimization step if key=='train'.

        :param key: {'train', 'valid'}; phase of the epoch
        :type key: str

        .. note::
            The optimization step is only performed after all batches are run.
        """
        if self.n_batches[key] <= 0:
            # XXX maybe we should append NaN to metric history?
            return
        self._phase = key

        tot_epoch_loss = 0.0
        tot_epoch_add_loss = 0.0
       # tot_epoch_DE_loss = 0.0

        #batch_loss = 0.0


        loss = torch.tensor([0.0]) #, requires_grad=True) #added by me for multihead
        #add_loss = torch.tensor([0.0])
        #DE_loss = torch.tensor([0.0])


        metric_values = {name: 0.0 for name in self.metrics_fn}

        # Zero the gradient only once, before running the batches. Gradients of different batches are accumulated.
        if key == 'train' and not _requires_closure(self.optimizer):
            self.optimizer.zero_grad()

        # perform forward pass for all batches: a single graph is created and release in every iteration
        # see https://discuss.pytorch.org/t/why-do-we-need-to-set-the-gradients-manually-to-zero-in-pytorch/4903/17

        for head in range(self.n_heads):
            #print('head', head)
            head_epoch_loss = 0.0
            head_epoch_add_loss = 0.0
           # head_epoch_DE_loss = 0.0

            self.nets = self.all_nets[:,head]
            self.diff_eqs = self.pde_list[head]
            self.conditions = self.all_conditions[:,head]
           # y = self.generator[key].get_examples()[1].detach().numpy()
           # plt.hist(y)
            #print(len(self.generator['train'].get_examples()))
            
            #print(self.nets)
            #print(self.diff_eqs)
            #print(self.generator[key].get_examples())
            
            for batch_id in range(self.n_batches[key]):
                batch = self._generate_batch(key)

                #print(len(batch[0]))
               # print(self.n_batches[key])

                batch_loss = 0.0
                batch_add_loss = 0.0
                #batch_DE_loss = 0.0

                def closure(zero_grad=True):
                    nonlocal batch_loss, batch_add_loss

                    if key == 'train' and zero_grad:
                        self.optimizer.zero_grad()
                    funcs = [
                        self.compute_func_val(n, c, *batch) for n, c in zip(self.nets, self.conditions)
                    ]

                    for name in self.metrics_fn:
                        value = self.metrics_fn[name](*funcs, *batch).item()
                        metric_values[name] += value

                    #CALLING THE EQUATIONS

                    residuals = self.diff_eqs(*funcs, *batch)
                    residuals = torch.cat(residuals, dim=1)
                   # print(residuals)

                    try:
                        #DE_loss = self.GA_loss_fn(residuals, funcs, batch)
                        DE_loss = self.GA_loss_fn(residuals, funcs, batch)
                        if head == self.n_heads-1:
                            add_loss = self.additional_loss(residuals, funcs, batch)
                        else:
                            add_loss = torch.tensor([0.0])
                        loss = DE_loss + add_loss

                        #self.metrics_history['add_loss'].append()

                        #print('Head' + str(head) + ' loss ' + str(loss))
                        #if self.global_epoch %100:
                            #print('loss ',head,'=',loss)

                    except TypeError as e:
                        warnings.warn(
                            "You might need to update your code. "
                            "Since v0.4.0; both `criterion` and `additional_loss` requires three inputs: "
                            "`residual`, `funcs`, and `coords`. See documentation for more.", FutureWarning)
                        raise e

                    # accumulate gradients before the current graph is collected as garbage

                    #DOING BACKPROPAGATION
                    if key == 'train':
                        loss.backward()
                        batch_loss = loss.item()
                        batch_add_loss = add_loss.item()
                       # batch_DE_loss = DE_loss.item()
                        #print('0',batch_loss)

                   # print(loss,add_loss,DE_loss)

                    return loss, add_loss#, DE_loss

                if key == 'train':
                    #print('Before closure')
                        # Optimizer step will be performed only once outside the for-loop (i.e. after all batches).
                    closure(zero_grad=False)    #closure(zero_grad=False) was inside else in initial code
                    #print('After closure')
                    #print('key 2.batch')

                    head_epoch_loss += batch_loss
                    head_epoch_add_loss += batch_add_loss
                   # head_epoch_DE_loss += batch_DE_loss
                    #print('head_loss', head_epoch_loss)
                    #head_epoch_loss += closure().item()

            if key == 'train':
                if _requires_closure(self.optimizer):
                    self._do_optimizer_step(closure=closure)
                    #print('optimizer step, key 1 head')

                ##else:
                    #closure(zero_grad=False)
                    #print('key 2.head')

                tot_epoch_loss += head_epoch_loss
                tot_epoch_add_loss += head_epoch_add_loss
            #    tot_epoch_DE_loss += head_epoch_DE_loss
             #   print('epoch losses: ', tot_epoch_loss,tot_epoch_add_loss,tot_epoch_DE_loss)

            else:
                tot_epoch_loss += closure()[0].item()
                tot_epoch_add_loss += closure()[1].item()
               # tot_epoch_DE_loss += closure()[2].item()
                #print('key 3.head')

            # If validation is performed, update the best network with the validation loss
            # Otherwise, try to update the best network with the training loss
        #print(tot_epoch_loss)
        self.metrics_history['r2_loss'].append(tot_epoch_loss)
        self.metrics_history['add_loss'].append(tot_epoch_add_loss)


        if key == 'valid' or self.n_batches['valid'] == 0:
            self._update_best(key)

        # perform the optimizer step after all heads are run (if optimizer.step doesn't require `closure`)
        if key == 'train' and not _requires_closure(self.optimizer):
            self._do_optimizer_step()
            #print('optimizer step , key 4 head')

            #tot_epoch_loss += (head_epoch_loss / self.n_batches[key])

        # calculate the sum of all losses (one per head) and register to history

        self._update_history(tot_epoch_loss, 'loss', key)

     #   self.metrics_history['add_loss'].append(tot_epoch_add_loss)
      #  self.metrics_history['DE_loss'].append(tot_epoch_DE_loss)

        # calculate total metrics across heads (and averaged across batches) and register to history
        for name in self.metrics_fn:
            print(name)
            self._update_history(
                metric_values[name], name, key)


    def run_custom_epoch(self):
      r"""Run a training epoch, update history, and perform gradient descent."""
      self.custom_epoch('train')
    def _update_best(self, key):
        """Update ``self.lowest_loss`` and ``self.best_nets``
        if current training/validation loss is lower than ``self.lowest_loss``
        """
        current_loss = self.metrics_history['r2_loss'][-1]
        if (self.lowest_loss is None) or current_loss < self.lowest_loss:
            self.lowest_loss = current_loss
            for i in range(self.n_heads):
              self.best_nets_list[i] = deepcopy(self.all_nets[:,i])

    def fit(self, max_epochs, callbacks=(), tqdm_file='default', **kwargs):
      r"""Run multiple epochs of training and validation, update best loss at the end of each epoch.

      If ``callbacks`` is passed, callbacks are run, one at a time,
      after training, validating and updating best model.

      :param max_epochs: Number of epochs to run.
      :type max_epochs: int
      :param callbacks:
          A list of callback functions.
          Each function should accept the ``solver`` instance itself as its **only** argument.
      :rtype callbacks: list[callable]
      :param tqdm_file:
          File to write tqdm progress bar. If set to None, tqdm is not used at all.
          Defaults to ``sys.stderr``.
      :type tqdm_file: io.StringIO or _io.TextIOWrapper

      .. note::
          1. This method does not return solution, which is done in the ``.get_solution()`` method.
          2. A callback ``cb(solver)`` can set ``solver._stop_training`` to True to perform early stopping.
      """
      self._stop_training = False
      self._max_local_epoch = max_epochs

      self.callbacks = callbacks

      monitor = kwargs.pop('monitor', None)
      if monitor:
          warnings.warn("Passing `monitor` is deprecated, "
                        "use a MonitorCallback and pass a list of callbacks instead")
          callbacks = [monitor.to_callback()] + list(callbacks)
      if kwargs:
          raise ValueError(f'Unknown keyword argument(s): {list(kwargs.keys())}')  # pragma: no cover

      flag=False
      if str(tqdm_file) == 'default':
          bar = tqdm(
              total = max_epochs,
              desc='Training Progress',
              colour='blue',
              dynamic_ncols=True,
          )
      elif tqdm_file is not None:
          bar = tqdm_file
      else:
          flag=True



      for local_epoch in range(max_epochs):
            #stop training if self._stop_training is set to True by a callback
          if self._stop_training:
              break

          # register local epoch (starting from 1 instead of 0) so it can be accessed by callbacks
          self.local_epoch = local_epoch + 1
          #self.run_train_epoch()
          self.run_custom_epoch()
          self.run_valid_epoch()
          for cb in callbacks:
              cb(self)
          if not flag:
              bar.update(1)
                

    def get_solution(self, copy=True, best=True):
        r"""Get a (callable) solution object. See this usage example:

        .. code-block:: python3

            solution = solver.get_solution()
            point_coords = train_generator.get_examples()
            value_at_points = solution(point_coords)

        :param copy:
            Whether to make a copy of the networks so that subsequent training doesn't affect the solution;
            Defaults to True.
        :type copy: bool
        :param best:
            Whether to return the solution with lowest loss instead of the solution after the last epoch.
            Defaults to True.
        :type best: bool
        :return:
            A solution object which can be called.
            To evaluate the solution on certain points,
            you should pass the coordinates vector(s) to the returned solution.
        :rtype: BaseSolution
        """
        nets = self.best_nets if best else self.nets
        conditions = self.conditions
        if copy:
            nets = deepcopy(nets)
            conditions = deepcopy(conditions)

        return SolutionBundle2D(nets, conditions)

    def _get_internal_variables(self):
        available_variables = super(BundleSolver2D, self)._get_internal_variables()
        available_variables.update({
            'r_min': self.r_min,
            'r_max': self.r_max,
            'eq_param_index': self.eq_param_index,
        })
        return available_variables


# Import more packages #
import neurodiffeq
from neurodiffeq.conditions import DirichletBVP2D
from neurodiffeq.solvers import Solver2D, BaseSolution
from neurodiffeq.monitors import Monitor2D
from generators import *
from neurodiffeq.conditions import IBVP1D,BaseCondition
from neurodiffeq.networks import FCNN
from neurodiffeq import diff
from neurodiffeq.callbacks import ActionCallback 
import torch
from torch.optim import lr_scheduler
from neurodiffeq.pde import make_animation
#from test_pre_Zurich.MH_BC_Burgers.Queue_files_orthogonal.MHBundleSolver2D_orthogonal import *
from torch import nn as nn
import tqdm as tqdm
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np
import pickle

class DoSchedulerStep(ActionCallback):
    def __init__(self, scheduler):
        super().__init__()
        self.scheduler = scheduler

    def __call__(self, solver):
        self.scheduler.step()
        
def _requires_closure(optimizer):
    # starting from torch v1.13, simple optimizers no longer have a `closure` argument
    closure_param = inspect.signature(optimizer.step).parameters.get('closure')
    return closure_param and closure_param.default == inspect._empty

class NET(nn.Module):
    def __init__(self, H_model, head_model):
        super(NET, self).__init__()
        self.H_model = H_model
        self.head_model = head_model

    def forward(self, x):
        x = self.H_model(x)
        x = self.head_model(x)
        #x = gaussian_smooth_2d(x, sigma=20,kernel_size=7)
        return x
    
class NET_Gaussian(nn.Module):
    def __init__(self, H_model, head_model):
        super(NET, self).__init__()
        self.H_model = H_model
        self.head_model = head_model

    def forward(self, x):
        x = self.H_model(x)
        x = self.head_model(x)
        
        return x

# MULTIHEAD NN FREEZE #
class NET_FREEZE(nn.Module):
    def __init__(self, H_model, head_model):
        super(NET_FREEZE, self).__init__()

        for param in H_model.parameters():
            param.requires_grad = False
        self.H_model = H_model
        self.head_model = head_model
        # Freeze the parameters of H_model
        #for param in self.H_model.parameters():
            #param.requires_grad = False

    def forward(self, x):
        x = self.H_model(x)
        x = self.head_model(x)
        return x
    
def save_model(solver,path,nu_list,body_units,basis_length):
    H_state = solver.best_nets_list[0][0].H_model.state_dict()
    head_state = []
    #nu_list = solver.nu_list
    loss = solver.metrics_history['r2_loss']
    global_epoch = len(loss)
    optim_state = solver.optimizer.state_dict()
    for i in range(len(solver.best_nets_list)):
        head_state.append(solver.best_nets_list[i][0].head_model.state_dict())
    mega_data = {
        "H_state": H_state,
        "head_state": head_state,
        "Loss": loss,
        "optim": optim_state,
        "body_units": body_units,
        "basis": basis_length,
        'nu_list': nu_list
                 }
    torch.save(mega_data, path+ '/Burguers_'+ str(global_epoch))

def load_nets(path,nets,optim,solver):
  master_dict = torch.load(path,map_location=torch.device('cpu'))
  H_state = master_dict['H_state']
  head_state = master_dict['head_state']
  optim_state = master_dict['optim']
  loss = master_dict['Loss']
  global_epoch = len(loss)
  #solver.global_epoch = global_epoch
  solver.metrics_history['train_loss'] = loss
  solver.metrics_history['valid_loss'] = loss
  solver.metrics_history['r2_loss'] = loss
  for i in range(len(nets[:,0])):
    nets[i,0].H_model.load_state_dict(H_state)
    for j in range(n_heads):
      print(j)
      nets[i,j].head_model.load_state_dict(head_state[j])
      #solver.best_nets_list[i,j] = nets[i,j]
  #optim.load_state_dict(optim_state)

# For the smoothenin #
def gaussian_2d_kernel(size=5, sigma=1.0, device="cpu", dtype=torch.float32):
    """Creates a 2D Gaussian kernel."""
    x = torch.arange(-size//2 + 1, size//2 + 1, dtype=dtype, device=device)
    y = torch.arange(-size//2 + 1, size//2 + 1, dtype=dtype, device=device)
    xx, yy = torch.meshgrid(x, y, indexing="ij")

    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel /= kernel.sum()  # Normalize

    return kernel.view(1, 1, size, size)  # Shape: (out_channels, in_channels, H, W) # Normalize


def gaussian_smooth_2d(u, kernel_size=5, sigma=1.0):
    """Smooths a 2D function u(t, x) using a 2D Gaussian filter with reflection padding."""
    device, dtype = u.device, u.dtype
    kernel = gaussian_2d_kernel(kernel_size, sigma, device, dtype)

    # Add batch and channel dimensions
    u = u.unsqueeze(0).unsqueeze(0)  

    # Apply reflection padding before convolution
    pad = kernel_size // 2
    u_padded = F.pad(u, (pad, pad, pad, pad), mode="reflect")  

    smoothed = F.conv2d(u_padded, kernel)
    return smoothed.squeeze()

# THIS IS WAVE EQUATION #
def generate_equations(n_heads):
  eq_dict = {} # Initialize the dictionary
  # Loop to create functions
  for head in range(n_heads):
      # Define a new function using a lambda or nested function
      def make_equations(head):
        def burguers(u,x,t,nu):
            term1 = diff(u, t, order=2)
            term2 = - (nu ** 2) * diff(u, x, order=2)
            return [term1 + term2]
        return burguers
      # Store the function in the dictionary
      eq_dict.update({f'equations_{head}':make_equations(head)})  # Now at the correct indentation level
  print('Equations dictionary generated')
  print(eq_dict)
  return eq_dict

# Define the boundary condition #
def init_gauss(x,a,sigma):
    return a/(np.sqrt(2*np.pi)*sigma)*torch.exp(-x**2 /(2*sigma**2))

def init_sine(x,a,n):
    return a*(torch.sin(x * n * np.pi / 5))

def init_N_wave(x,a,b):
    return a*torch.exp(-(x-1)**2 / 2) -b*torch.exp(-(x+1)**2 / 2)

def init_cos(x,a,n):
    return a*(torch.cos(x * n * np.pi /10))**2
def square_wave(x, a):
    """
    Generate a square wave with amplitude a.
    x: torch.Tensor
    a: amplitude (float or tensor)
    Returns: torch.Tensor
    """
    return a * (torch.sign(torch.sin(x))+1)/2
def sinc_func(x):
    return (torch.sin(2*np.pi*x)-torch.sin(np.pi*x))/(np.pi*x)

def morlet_wavelet(t, f=1.0, sigma=1.0):
    """
    Generates a Morlet wavelet.
    
    Parameters:
    - t: array of time values
    - f: frequency of the wavelet
    - sigma: standard deviation of the Gaussian envelope
    
    Returns:
    - wavelet: array of wavelet values
    """
    # Morlet wavelet = Gaussian envelope * complex exponential
    return torch.exp(-t**2 / (2 * sigma**2)) * torch.cos(2 * np.pi * f * t)

# Ricker Wavelet function #
def ricker(x, sigma=0.3):
    factor1 = 2 / (np.sqrt(3 * sigma) * np.pi ** (1/4))
    factor2 = (1 - (x / sigma) ** 2)
    factor3 = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    return factor1 * factor2 * factor3

def R_wavelet(x, s, b):
    return 1 / np.sqrt(s) * ricker((x - b) / s)

def R_wavelet_POS(x, s, b):
    return 1 / np.sqrt(s) * ricker((x - b) / s)+0.8

def init_cos_norm(x,a,n):
    return a*(torch.cos(x * n * np.pi /10))