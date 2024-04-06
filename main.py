import numpy as np
import matplotlib.pyplot as plt
import torch
tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
}
import sys
import inspect
import test_functions
import pybenchfunction as bench
from test_functions import Forrester, Ackley, Branin, BraninModified, AlpineN1
import botorch
import gpytorch
from gpytorch.models import ExactGP
from botorch.models.gpytorch import GPyTorchModel
from botorch.models import SingleTaskGP, ModelListGP
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch import fit_gpytorch_model
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.acquisition import ExpectedImprovement
from botorch.optim import optimize_acqf

MIN_INFERRED_NOISE_LEVEL = 1e-3


target_function = bench.function.AlpineN1(9)
dim = target_function.d
X_min, minimum = target_function.get_global_minimum(dim)
lower_bounds = target_function.input_domain[:,0]
upper_bounds = target_function.input_domain[:,1]
bounds = torch.tensor([lower_bounds, upper_bounds]).double()




def generate_initial_data(dim, n=3):
    x_init = np.random.default_rng().uniform(low = lower_bounds, high = upper_bounds, size = (n,dim))
    y_init = []
    for i in range(n):
        y_init.append(target_function(x_init[i,:]))

    y_init = torch.tensor(y_init).unsqueeze(-1).double()
    best_val = y_init.min()
    return torch.tensor(x_init).double(), y_init, best_val.item()




def get_next_point(x_init, y_init, best_val, bounds, n_points=1):
    model = SingleTaskGP(x_init, y_init, covar_module=gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()))
    model.to(dtype=torch.double)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)

    fit_gpytorch_model(mll)

    EI = ExpectedImprovement(model=model, best_f=best_val, maximize=False)

    candidates, _ = optimize_acqf(
        acq_function=EI,
        bounds=bounds,
        q=n_points,
        num_restarts=200,
        raw_samples=256,
        options={"batch_limit": 5, "maxiter": 200})

    return candidates


epochs = 100
chains = 5
all_hist = []

for j in range(chains):
    print(F"Starting Series {j+1} of {chains}")
    x_init, y_init, best_val = generate_initial_data(dim, 3)
    best_vals = []
    g_min = minimum

    best_vals.append(best_val)

    for i in range(epochs):
        x_new = get_next_point(x_init, y_init, best_val, bounds, n_points=1)
        y_new = torch.tensor(target_function(x_new.numpy())).reshape((1,1)).double()

        x_init = torch.cat([x_init, x_new])
        y_init = torch.cat([y_init, y_new])

        best_val = y_init.min().item()
        best_vals.append(best_val)
        if best_val < minimum + 0.01:
            best_vals[i:epochs] = [best_val for t in range(i, epochs)]
            break

        #print(F"Iteration: {i} \t Current Optimum: {best_val:.5} \t Best Possible: {g_min}")
    all_hist.append(best_vals)


temp = np.array(all_hist)
print(temp.shape)
#np.save('Alpinen1_9dim_MLE_100.npy', temp)
ind = np.arange(1, temp.shape[1]+1)

means = np.mean(temp, axis=0)
med = np.median(temp, axis=0)

ub = np.quantile(temp, 0.95, axis=0)
lb = np.quantile(temp, 0.05, axis=0)

plt.plot(ind, med, 'k', label='Mean Behavior')
plt.fill_between(ind, ub, lb, alpha=0.4, color='gray', label='90% Quantile')
plt.xlabel('Iter')
plt.ylabel('Min')
plt.grid()
plt.legend()
#plt.savefig('Alpinen1_9dim_MLE_100.pdf', bbox_inches='tight')





#
# #
# # if __name__ == '__main__':
# #     available_functions = inspect.getmembers(test_functions, inspect.isclass)
# #     functions = [cls for clsname, cls in available_functions]
# #     for sn in functions:
# #         f = sn()
# #         print(f.d)
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
