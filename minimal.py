import deepinv as dinv
import matplotlib.pyplot as plt
import torch

from utils.physics import BlurMatrix
from utils.convolution_matrix import create_gaussian_kernel_1d
from multilevel.block import BlockCoordinateDescent

PSNR = dinv.metric.PSNR()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
seed = torch.manual_seed(0)  # Random seed for reproducibility

x_true = dinv.utils.load_example("butterfly.png", device=device)

# Wavelet parameters
J = 3
wv_type = 'haar'

# Physics
filter_row = create_gaussian_kernel_1d(sigma=1.0, device=device, dtype=torch.float32)
filter_col = create_gaussian_kernel_1d(sigma=1.0, device=device, dtype=torch.float32)
physics = BlurMatrix(filter_row, filter_col, padding="circular", device=device)
sigma = 0.1
noise_model = dinv.physics.GaussianNoise(sigma=sigma)

physics.noise_model = dinv.physics.GaussianNoise(sigma=sigma)

# Observation
y = physics(x_true).to(device)

# Objective function
data_fidelity = dinv.optim.L2()
prior = dinv.optim.L1Prior()
reg_weight = 0.5

# Parameters
n_iter = 50
n_iter_coarse = 10
Anorm2 = physics.compute_norm(x_true).item()
stepsize = 0.1 / Anorm2
update_mode = 'MLFBcond'  # 'MLFB', 'FB' or 'MLFBcond'*
print(f'Using device: {device}')
print(f"Stepsize: {stepsize}")

params = {
    "image": "butterfly.png",
    "physics": "Inpainting + Gaussian noise",
    "sigma": sigma,
    "reg_weight": reg_weight,
    "n_iter": n_iter,
    "stepsize": stepsize,
    "J": J,
    "wavelet": wv_type,
    "update_mode": update_mode,
    "comment": "With approx thresholding in BCD"
}

bcd = BlockCoordinateDescent(x_true.shape, wv_type=wv_type, physics=physics, data_fidelity=data_fidelity, prior=prior, max_levels=J, stepsize=stepsize, device=device)

x0 = y.clone()
print("Running BCD MLFB...")
x_recon_mlfb, loss_mlfb, times_mlfb, cycles_mlfb, psnr_mlfb = bcd.run(y, x0, x_true=x_true, n_iter=n_iter, n_iter_coarse=n_iter_coarse, reg_weight=reg_weight, update_mode='MLFB', metrics=True)

x0 = y.clone()
x_recon_fb, loss_fb, times_fb, cycles_fb, psnr_fb = bcd.run(y, x0, x_true=x_true, n_iter=n_iter, n_iter_coarse=n_iter_coarse, reg_weight=reg_weight, update_mode='FB', metrics=True)

x0 = y.clone()
x_recon_cyclic, loss_cyclic, times_cyclic, cycles_cyclic, psnr_cyclic = bcd.run(y, x0, x_true=x_true, n_iter=n_iter, n_iter_coarse=n_iter_coarse, reg_weight=reg_weight, update_mode='cyclic', metrics=True)

x_recon_mlfbdetails, loss_mlfbdetails, times_mlfbdetails, cycles_mlfbdetails, psnr_mlfbdetails = bcd.run(y, x0, x_true=x_true, n_iter=n_iter, n_iter_coarse=n_iter_coarse, reg_weight=reg_weight, update_mode='MLFBdetails', metrics=True)

psnrs = [PSNR(y, x_true).item(), PSNR(x_recon_mlfb, x_true).item(), PSNR(x_recon_fb, x_true).item(), PSNR(x_recon_cyclic, x_true).item(), PSNR(x_recon_mlfbdetails, x_true).item()]
psnrs = [f"{p:.2f}" for p in psnrs]

cycles_mlfb.insert(0, 0)
cycles_fb.insert(0, 0)
cycles_cyclic.insert(0, 0)
cycles_mlfbdetails.insert(0, 0)

dinv.utils.plot([x_true, y, x_recon_mlfb, x_recon_fb, x_recon_cyclic, x_recon_mlfbdetails], titles=['Original', 'Observation', 'BCD MLFB', 'BCD FB', 'BCD Cyclic', 'BCD MLFB Details'], subtitles=['PSNR:', f"{psnrs[0]}", f"{psnrs[1]}", f"{psnrs[2]}", f"{psnrs[3]}", f"{psnrs[4]}"])

plt.figure()
plt.plot(loss_mlfb, label='BCD MLFB')
plt.plot(loss_fb, label='BCD FB')
plt.plot(loss_cyclic, label='BCD Cyclic', linestyle='--')
plt.plot(loss_mlfbdetails, label='BCD MLFB Details', linestyle='--')
plt.xlabel('Iterations')
#plt.xscale('log')
plt.ylabel('Objective function')
plt.title('Objective function vs Iterations')
plt.legend()
plt.grid()
plt.show()

# Time curves
plt.figure()
plt.plot(times_mlfb, loss_mlfb, label='BCD MLFB')
plt.plot(times_fb, loss_fb, label='BCD FB')
plt.plot(times_cyclic, loss_cyclic, label='BCD Cyclic', linestyle='--')
plt.plot(times_mlfbdetails, loss_mlfbdetails, label='BCD MLFB Details', linestyle='--')
plt.xlabel('Time (s)')
#plt.xscale('log')
plt.ylabel('Objective function')
plt.title('Objective function vs Time')
plt.legend()
plt.grid()
plt.show()