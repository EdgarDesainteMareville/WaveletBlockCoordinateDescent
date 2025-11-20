import deepinv as dinv
import torch
import torch.nn.functional as F
import pywt
import time
import copy
from tqdm import tqdm

PSNR = dinv.metric.PSNR()

from multilevel.information_transfer import WaveletInformationTransferMatrices
from utils.priors import WaveletPriorCustom

class BlockCoordinateDescent():
    def __init__(self, img_size, wv_type, physics, data_fidelity, prior, max_levels, stepsize=1e-3, device='cpu'):
        self.img_size = img_size
        self.wv_type = wv_type
        self.physics = physics
        self.data_fidelity = data_fidelity
        self.prior = prior
        self.max_levels = max_levels
        self.device = device

        self.stepsize = stepsize
        self.reg_weight = None
        self.y = None

        self.grad_matrices = {}       # Matrices for the gradient computation
        self.grad_wavelet_terms = {}  # Intermediate terms for the gradient computation
        self.grad_wavelet = {}        # Gradient in the wavelet domain
        self.proj_ATy = {}            # Projections of A^T y onto the wavelet subspaces

        #self.wavelet_prior = dinv.optim.WaveletPrior(level=self.max_levels, wv=self.wv_type, device=self.device)
        self.wavelet_prior = WaveletPriorCustom(level=self.max_levels, wv=self.wv_type, device=self.device)
        self.information_transfer = WaveletInformationTransferMatrices(wv=self.wv_type, mode='periodization', level=self.max_levels, device=self.device)
        self.Pi_ops = self.information_transfer.compute_Pi_operators(N=self.img_size[-1], levels=self.max_levels, wavelet=self.wv_type, mode='periodization')

    def update_blocks(self, x_wavelet, n_iter_coarse, updated_blocks, reg_weight, use_conditional_thresholding=False):
        # Update list is a list of tuples (level, mode) where mode is 'approx' or 'details'

        for scale, mode in updated_blocks:
            level = self.max_levels - scale
            if mode == 'approx':
                coeff = x_wavelet[f"A{self.max_levels}"]
                #print(f'Coeff {mode} {level} shape: {coeff.shape}')

                for i in range(n_iter_coarse):
                    # Only a gradient step on the approximation coefficients
                    coeff = coeff - self.stepsize * self.grad_wavelet[f'A{self.max_levels}']

                    self.current_iter += 1
                    self.compute_metrics(x_wavelet)

                x_wavelet[f"A{self.max_levels}"] = coeff

            elif mode == 'details':
                for char in ['V', 'H', 'D']:
                    coeff = x_wavelet[f"{char}{level}"]
                    #print(f'Coeff {mode} {level} {char} shape: {coeff.shape}')

                    if use_conditional_thresholding:
                            # Compute global threshold
                            level_threshold = (self.stepsize * reg_weight) / 2**level

                            # Reconstruct approximation at current level
                            approx_coeffs = {}
                            approx_coeffs = x_wavelet # Get all current wavelet coeffs
                            for l in range(level, 0, -1):
                                approx_coeffs[f'V{l}'] = torch.zeros_like(approx_coeffs[f'V{l}'])
                                approx_coeffs[f'H{l}'] = torch.zeros_like(approx_coeffs[f'H{l}'])
                                approx_coeffs[f'D{l}'] = torch.zeros_like(approx_coeffs[f'D{l}'])

                            # approx_at_level's shape is the same as the original image but contains the approximation at 'level'
                            approx_at_level = self.information_transfer.idwt(self.Pi_ops, approx_coeffs)

                            # Downsample to get the correct size for the current level
                            factor = 2 ** level
                            approx_downsampled = approx_at_level[..., ::factor, ::factor]
                            approx_downsampled = approx_downsampled.contiguous()

                            # Get wavelet filters
                            low, high = pywt.Wavelet(self.wv_type).filter_bank[:2]
                            low, high = torch.tensor(low, device=self.device, dtype=coeff.dtype), torch.tensor(high, device=self.device, dtype=coeff.dtype)

                            filt_hl = torch.outer(low, high).unsqueeze(0).unsqueeze(0)
                            filt_lh = torch.outer(high, low).unsqueeze(0).unsqueeze(0)
                            filt_hh = torch.outer(high, high).unsqueeze(0).unsqueeze(0)

                            # Compute the undecimated wavelet coefficients of the approximation (details only)
                            B, C, Hx, Wx = approx_downsampled.shape # C = 3
                            kH, kW = filt_hl.shape[-2:]

                            # duplicate filter for each channel
                            filt_hl_expanded = filt_hl.expand(C, 1, kH, kW) # shape (C, 1, kH, kW)
                            filt_lh_expanded = filt_lh.expand(C, 1, kH, kW)
                            filt_hh_expanded = filt_hh.expand(C, 1, kH, kW)

                            H = F.conv2d(approx_downsampled, filt_hl_expanded, padding='same', groups=C)
                            V = F.conv2d(approx_downsampled, filt_lh_expanded, padding='same', groups=C)
                            D = F.conv2d(approx_downsampled, filt_hh_expanded, padding='same', groups=C)

                            thresh_H = level_threshold / (torch.abs(H) + 1e-8)
                            thresh_V = level_threshold / (torch.abs(V) + 1e-8)
                            thresh_D = level_threshold / (torch.abs(D) + 1e-8)

                    for i in range(n_iter_coarse):
                        coeff = coeff - self.stepsize * self.grad_wavelet[f'{char}{level}']

                        if use_conditional_thresholding:
                            if char == 'H':
                                coeff = self.prior.prox(coeff, gamma=thresh_H)
                            elif char == 'V':
                                coeff = self.prior.prox(coeff, gamma=thresh_V)
                            elif char == 'D':
                                coeff = self.prior.prox(coeff, gamma=thresh_D)
                        else:
                            coeff = self.prior.prox(coeff, gamma=self.stepsize * reg_weight)

                        self.current_iter += 1
                        self.compute_metrics(x_wavelet)

                    x_wavelet[f'{char}{level}'] = coeff

            else:
                raise ValueError("Invalid mode. Choose 'details' or 'approx'.")

        #print('Updating gradient...\n')
        self.update_gradient(updated_blocks, x_wavelet)

        return x_wavelet

    def compute_gradient_matrices(self):
        A_row_TA_row = self.physics.A_row.T @ self.physics.A_row
        A_col_TA_col = self.physics.A_col.T @ self.physics.A_col
        types = ['A', 'V', 'H', 'D']

        for X in types:
            # Niveau max pour ce type
            if X == 'A':
                level_X = [self.max_levels]  # A uniquement au niveau max
            else:
                level_X = range(self.max_levels, 0, -1)  # V,H,D aux niveaux max, max-1, max-2

            for lX in level_X:
                for Y in types:
                    # Niveau cible : tous les niveaux possibles pour V,H,D, mais seulement max pour A
                    if Y == 'A':
                        level_Y = [self.max_levels]
                    else:
                        level_Y = range(self.max_levels, 0, -1)

                    for lY in level_Y:
                        key_row = f'{X}{lX}_{Y}{lY}_row'
                        key_col = f'{X}{lX}_{Y}{lY}_col'

                        PiX_row = self.Pi_ops[f'{X}{lX}_row']
                        PiY_row = self.Pi_ops[f'{Y}{lY}_row']
                        PiX_col = self.Pi_ops[f'{X}{lX}_col']
                        PiY_col = self.Pi_ops[f'{Y}{lY}_col']

                        self.grad_matrices[key_row] = PiX_row @ A_row_TA_row @ PiY_row.T
                        self.grad_matrices[key_col] = PiX_col @ A_col_TA_col @ PiY_col.T

        # Projections A^T y
        ATy = self.physics.A_row.T @ self.y @ self.physics.A_col
        for X in types:
            if X == 'A':
                levels = [self.max_levels]
            else:
                levels = range(self.max_levels, 0, -1)

            for l in levels:
                PiX_row = self.Pi_ops[f'{X}{l}_row']
                PiX_col = self.Pi_ops[f'{X}{l}_col']
                self.proj_ATy[f'{X}{l}'] = PiX_row @ ATy @ PiX_col.T

    def compute_gradient_terms(self, x_wavelet, Xchar, Xlevel, Ychar, Ylevel):
        # Compute intermediate terms for gradient computation
        # i.e. computes \Pi_X A^T A \Pi_Y^T x_wavelet[Y]

        key_row = f'{Xchar}{Xlevel}_{Ychar}{Ylevel}_row'
        key_col = f'{Xchar}{Xlevel}_{Ychar}{Ylevel}_col'

        term = self.grad_matrices[key_row] @ x_wavelet[f'{Ychar}{Ylevel}'] @ self.grad_matrices[key_col].T
        self.grad_wavelet_terms[f'{Xchar}{Xlevel}_{Ychar}{Ylevel}'] = term

    def compute_gradient(self, x_wavelet, Xchar, Xlevel):
        # Compute the gradient for a given block
        # i.e. computes \sum_Y ( \Pi_X A^T A \Pi_Y^T x_wavelet[Y] ) - \Pi_X A^T y
        keyX = f'{Xchar}{Xlevel}'

        grad_X = - self.proj_ATy[keyX]

        for Ychar in ['A', 'V', 'H', 'D']:
            keyY_levels = [self.max_levels] if Ychar == 'A' else range(self.max_levels, 0, -1)
            for lY in keyY_levels:
                key_row = f'{Xchar}{Xlevel}_{Ychar}{lY}_row'
                key_col = f'{Xchar}{Xlevel}_{Ychar}{lY}_col'

                self.compute_gradient_terms(x_wavelet, Xchar, Xlevel, Ychar, lY)
                grad_X += self.grad_wavelet_terms[f'{Xchar}{Xlevel}_{Ychar}{lY}']

        self.grad_wavelet[keyX] = grad_X

    def update_gradient(self, updated_blocks, x_wavelet):
        for scale, mode in updated_blocks:
            level = self.max_levels - scale

            if mode == 'approx':
                # If we just updated the approximation
                for Xchar in ['A', 'V', 'H', 'D']:
                    Xlevel = [self.max_levels] if Xchar == 'A' else range(self.max_levels, 0, -1)
                    for lX in Xlevel:
                        # Re-compute the terms \Pi_X A^T A \Pi_A^T x_wavelet[A] for every block X
                        # (the only thing that changed is x_wavelet[A])
                        self.compute_gradient_terms(x_wavelet, Xchar, lX, 'A', self.max_levels)
                        self.compute_gradient(x_wavelet, Xchar, lX)

            elif mode == 'details':
                # If we just updated the details at level 'level'
                for t in ['V', 'H', 'D']:
                    Ychar = t
                    Ylevel = level
                    for Xchar in ['A', 'V', 'H', 'D']:
                        Xlevel = [self.max_levels] if Xchar == 'A' else range(self.max_levels, 0, -1)
                        for lX in Xlevel:
                            # Re-compute the terms \Pi_X A^T A \Pi_Y^T x_wavelet[Y] for every block X
                            # (the only thing that changed is x_wavelet[Y] for Y=(t,level))
                            self.compute_gradient_terms(x_wavelet, Xchar, lX, Ychar, Ylevel)
                            self.compute_gradient(x_wavelet, Xchar, lX)

            else:
                raise ValueError("Invalid mode. Choose 'details' or 'approx'.")

    def run(self, y, x0, x_true, n_iter, n_iter_coarse, reg_weight, update_mode='MLFB', use_conditional_thresholding=False, metrics=False):
        self.reg_weight = reg_weight
        self.y = y
        self.x_true = x_true

        self.ATy = self.physics.A_adjoint(self.y)

        self.losses = []
        self.times = []
        self.psnrs = []
        self.current_iter = 0
        self.cycles = []

        self.compute_gradient_matrices()

        #print(self.grad_matrices[f'A{self.max_levels}_A{self.max_levels}_row'].shape)
        xk_wavelet = self.information_transfer.dwt(x0)

        for Xchar in ['A', 'V', 'H', 'D']:
            Xlevel = [self.max_levels] if Xchar == 'A' else range(self.max_levels, 0, -1)
            for lX in Xlevel:
                self.compute_gradient(x_wavelet=xk_wavelet, Xchar=Xchar, Xlevel=lX)

        print(f'Number of gradient matrices: {len(self.grad_matrices)}')

        # Initial gradient computation
        self.update_gradient([(0, 'approx')] + [(level, 'details') for level in range(self.max_levels)], xk_wavelet)

        # Compute initial metrics
        self.compute_metrics(x_wavelet=None, x_img=x0)

        # Update list
        update_list = UpdateList(self.max_levels).get_list(type=update_mode)

        if metrics:
            start = time.process_time()

        with torch.no_grad():
            with tqdm(range(n_iter), desc=f"BCD {update_mode}") as t:
                for it in t:
                    if metrics:
                        x_recon = self.information_transfer.idwt(self.Pi_ops, xk_wavelet)
                        t.set_postfix_str(f"loss={self.data_fidelity.fn(x_recon, y, self.physics).item() + self.reg_weight * self.wavelet_prior.fn(x_recon).item():.2f}")

                    # Update detail coefficients from coarse to fine
                    for updated_blocks in update_list:
                        #print(f'Updating blocks: {updated_blocks}')
                        xk_wavelet = self.update_blocks(xk_wavelet, n_iter_coarse=n_iter_coarse, updated_blocks=updated_blocks, reg_weight=reg_weight, use_conditional_thresholding=use_conditional_thresholding)
                    self.cycles.append(self.current_iter)

        x_recon = self.information_transfer.idwt(self.Pi_ops, xk_wavelet)

        if metrics:
            self.times = [t - start for t in self.times]  # Start at 0
            return x_recon, self.losses, self.times, self.cycles, self.psnrs
        return x_recon

    def compute_metrics(self, x_wavelet, x_img=None):
        if x_img is None:
            x_img = self.information_transfer.idwt(self.Pi_ops, x_wavelet)
        crit = self.data_fidelity.fn(x_img, self.y, self.physics).item() + self.reg_weight * self.wavelet_prior.fn(x_img).item()
        self.losses.append(crit)
        self.times.append(time.process_time())
        self.psnrs.append(PSNR(x_img, self.x_true).item())

def conditional_thresholding(
        details,
        approx,
        global_threshold
        ):
    """
    Conditional thresholding of the wavelet coefficients based on the gradient of the approximation.
    This function is used in the MultilevelWavelets class.

    Parameters
    ----------
    details : dict
        Dictionary containing the wavelet coefficients of the details (LH, HL, HH).
    approx : torch.Tensor
        The approximation coefficients of the wavelet transform.
    global_threshold : float
        The global threshold value for the wavelet coefficients, used to compute the other thresholds.
    """
    device, dtype = approx.device, approx.dtype

    l1_prior = dinv.optim.L1Prior()

    if isinstance(details, dict):
        LH = details["LH"]
        HL = details["HL"]
        HH = details["HH"]
    elif isinstance(details, torch.Tensor):
        LH, HL, HH = details[0], details[1], details[2]
    else:
        raise ValueError("Format de 'details' inattendu")

    # Wavelet transform of the approximation without downsampling
    stationnary_transform = pywt.swt2(approx.cpu().numpy(), wavelet="db8", level=1)
    grad_LH, grad_HL, grad_HH = (
        stationnary_transform[0][1][0],
        stationnary_transform[0][1][1],
        stationnary_transform[0][1][2],
    )

    grad_LH = torch.tensor(grad_LH, device=device, dtype=dtype)
    grad_HL = torch.tensor(grad_HL, device=device, dtype=dtype)
    grad_HH = torch.tensor(grad_HH, device=device, dtype=dtype)

    # Thresholds are inversely proportional to the gradient
    LH = l1_prior.prox(LH, global_threshold / grad_LH)
    HL = l1_prior.prox(HL, global_threshold / grad_HL)
    HH = l1_prior.prox(HH, global_threshold / grad_HH)

    return LH, HL, HH

class UpdateList():
    def __init__(self, max_levels):
        self.max_levels = max_levels

    def get_list(self, type='MLFB'):
        if type == 'MLFB':
            return self.create_update_list_MLFB()
        elif type == 'FB':
            return self.create_update_list_FB()
        elif type == 'MLFBdetails':
            return self.create_update_list_MLFBdetails()
        elif type == 'cyclic':
            return self.create_update_list_cyclic()
        else:
            raise ValueError("Invalid type. Choose 'MLFB', 'FB', 'MLFBdetails' or 'cyclic'.")

    def create_update_list_MLFB(self):
        update_list = [[(0, 'approx')]]
        updated_blocks = [(0, 'approx')]
        for i in range(self.max_levels):
            updated_blocks.append((i, 'details'))
            update_list.append(copy.deepcopy(updated_blocks))
        return update_list

    def create_update_list_FB(self):
        update_list = []
        updated_blocks = [(0, 'approx')] + [(level, 'details') for level in range(self.max_levels)]
        update_list = [copy.deepcopy(updated_blocks)]
        return update_list

    def create_update_list_MLFBdetails(self):
        update_list = [[(0, 'approx')]]
        updated_blocks = [(0, 'approx')]
        for i in range(self.max_levels):
            update_list.append([(i, 'details')])
            updated_blocks.append((i, 'details'))
            update_list.append(copy.deepcopy(updated_blocks))
        return update_list

    def create_update_list_cyclic(self):
        update_list = [[(0, 'approx')]]
        for i in range(self.max_levels):
            update_list.append([(i, 'details')])
        return update_list