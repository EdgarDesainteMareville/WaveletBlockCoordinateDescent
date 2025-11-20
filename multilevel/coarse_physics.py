import deepinv as dinv
import torch

from deepinv.physics import Physics

from custom_deepinv.physics import BlurMatrix
from multilevel.information_transfer import InformationTransferMatrix

class CoarsePhysicsOperator(Physics):
    def __init__(self, A_row, A_col):
        super().__init__()
        self.A_row = A_row
        self.A_col = A_col

    def A(self, x):
        return self.A_row @ x @ self.A_col.T

    def A_adjoint(self, y):
        return self.A_row.T @ y @ self.A_col

class CoarseBlurMatrix(BlurMatrix):
    """
    Coarse blur operator using matrix representation and information transfer operators.
    """

    def __init__(self, filter_row, filter_col, it_op: InformationTransferMatrix, level=1, padding="circular", device="cpu", **kwargs):
        super().__init__(filter_row, filter_col, padding=padding, device=device, **kwargs)
        self.it_op = it_op
        self.level = level

    def create_AH_matrices(self, signal_size):
        """
        Create the matrix representation of the coarse blur operator. Signal size is the size of the fine level.
        """
        AH_row, AH_col = self.create_A_matrix(signal_size=signal_size)
        for j in range(self.level):
            W_row, W_col = self.it_op.get_W_matrices(signal_size=AH_row.shape[0])
            AH_row = W_row @ AH_row @ W_col.T
            AH_col = W_row @ AH_col @ W_col.T
        self.A_row = AH_row
        self.A_col = AH_col

    def get_AH_matrices(self, signal_size):
        if (
            self.A_row is None
            or self.A_col is None
            or self.A_row.shape[0] != signal_size // (2 ** self.level)
            or self.A_col.shape[0] != signal_size // (2 ** self.level)
        ):
            self.create_AH_matrices(signal_size)
        return self.A_row, self.A_col

    def A(self, x_coarse):
        A_row, A_col = self.get_AH_matrices(signal_size = (2**self.level) * x_coarse.shape[-1])
        return A_row @ x_coarse @ A_col.T

    def A_adjoint(self, y_coarse):
        A_row, A_col = self.get_AH_matrices(signal_size = (2**self.level) * y_coarse.shape[-1])
        return A_row.T @ y_coarse @ A_col


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from code.utils.matrix import create_gaussian_kernel_1d

    # Example usage
    x = dinv.utils.load_example("butterfly.png", grayscale=False)

    filter_row = create_gaussian_kernel_1d(size=9, sigma=2.0, device='cpu', dtype=torch.float32)
    filter_col = create_gaussian_kernel_1d(size=9, sigma=2.0, device='cpu', dtype=torch.float32)

    it_op = InformationTransferMatrix(wv="haar", mode="periodization", device="cpu")
    coarse_blur = CoarseBlurMatrix(filter_row, filter_col, it_op=it_op, padding="circular", device="cpu")

    AH_row, AH_col = coarse_blur.get_AH_matrices(signal_size=x.shape[-1])
    print(f"Coarse blur matrix shape: {AH_row.shape[0]} x {AH_row.shape[1]}")

    x_coarse = it_op.downsample(x)
    print(f"Downsampled shape: {x_coarse.shape}")
    y_coarse = coarse_blur.A(x_coarse)
    print(f"Coarse blurred shape: {y_coarse.shape}")

    dinv.utils.plot([x, x_coarse, y_coarse], titles=["Original", "Downsampled", "Coarse Blurred"])