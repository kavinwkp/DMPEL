import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Optional, Union, Callable
from abc import ABCMeta, abstractmethod
import math

activation_t = Union[Callable[[torch.Tensor], torch.Tensor], torch.nn.Module]


class AnalyticLinear(torch.nn.Linear, metaclass=ABCMeta):
    def __init__(
        self,
        in_features: int,
        out_features: int = 0,
        gamma: float = 1e-1,
        bias: bool = False,
        device: Optional[Union[torch.device, str, int]] = None,
        dtype=torch.double,
    ):
        super(torch.nn.Linear, self).__init__()  # Skip the Linear class
        factory_kwargs = {"device": device, "dtype": dtype}
        self.gamma: float = gamma
        self.bias: bool = bias
        self.dtype = dtype

        # Linear Layer
        # if bias:
        #     in_features += 1

        # TODO: 17 subtasks
        weight = torch.zeros((in_features, out_features), **factory_kwargs)
        self.register_buffer("weight", weight)

    @torch.inference_mode()
    def forward(self, X):
        # X = X.to(self.weight)
        # if self.bias:
        #     X = torch.cat((X, torch.ones(X.shape[0], 1).to(X)), dim=-1)
        return X @ self.weight

    @property
    def in_features(self):
        if self.bias:
            return self.weight.shape[0] - 1
        return self.weight.shape[0]

    @property
    def out_features(self):
        return self.weight.shape[1]

    def reset_parameters(self):
        # Following the equation (4) of ACIL, self.weight is set to \hat{W}_{FCN}^{-1}
        self.weight = torch.zeros((self.weight.shape[0], 0)).to(self.weight)

    @abstractmethod
    def fit(self, X, Y):
        raise NotImplementedError()

    def update(self):
        assert torch.isfinite(self.weight).all(), (
            "Pay attention to the numerical stability! "
            "A possible solution is to increase the value of gamma. "
            "Setting self.dtype=torch.double also helps."
        )


class RecursiveLinear(AnalyticLinear):
    def __init__(
        self,
        in_features: int,
        out_features: int = 0,
        gamma: float = 1e-1,
        bias: bool = False,
        device: Optional[Union[torch.device, str, int]] = None,
        dtype=torch.double,
    ) -> None:
        super().__init__(in_features, out_features, gamma, bias, device, dtype)
        factory_kwargs = {"device": device, "dtype": dtype}

        # Regularized Feature Autocorrelation Matrix (RFAuM)
        self.R: torch.Tensor
        R = torch.eye(self.weight.shape[0], **factory_kwargs) / self.gamma
        self.register_buffer("R", R)

    @torch.no_grad()
    def fit(self, X, Y):
        """The core code of the ACIL and the G-ACIL.
        This implementation, which is different but equivalent to the equations shown in [1],
        is proposed in the G-ACIL [4], which supports mini-batch learning and the general CIL setting.
        """
        # X, Y = X.to(self.weight), Y.to(self.weight)
        # if self.bias:
        #     X = torch.cat((X, torch.ones(X.shape[0], 1).to(X)), dim=-1)

        num_targets = Y.shape[1]    # 6
        if num_targets > self.out_features:     # init 6 > 0
            increment_size = num_targets - self.out_features    # 6
            print(f"increment_size: {increment_size}")
            tail = torch.zeros((self.weight.shape[0], increment_size)).to(self.weight)  # (8192, 17)
            self.weight = torch.cat((self.weight, tail), dim=1)     # (8192, 17)
        elif num_targets < self.out_features:
            increment_size = self.out_features - num_targets
            tail = torch.zeros((Y.shape[0], increment_size)).to(Y)
            Y = torch.cat((Y, tail), dim=1)

        # Please update your PyTorch & CUDA if the `cusolver error` occurs.
        # If you insist on using this version, doing the `torch.inverse` on CPUs might help.
        # >>> K_inv = torch.eye(X.shape[0]).to(X) + X @ self.R @ X.T
        # >>> K = torch.inverse(K_inv.cpu()).to(self.weight.device)

        K = torch.inverse(torch.eye(X.shape[0]).to(X) + X @ self.R @ X.T)   # (bs, bs)
        # Equation (10) of ACIL
        self.R -= self.R @ X.T @ K @ X @ self.R     # (8192, 8192)
        # Equation (9) of ACIL
        self.weight += self.R @ X.T @ (Y - X @ self.weight)


class Buffer(torch.nn.Module, metaclass=ABCMeta):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()


class RandomBuffer(torch.nn.Linear, Buffer):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        device=None,
        dtype=torch.float,
        activation=None,
    ) -> None:
        super(torch.nn.Linear, self).__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.activation = (
            # torch.nn.Identity() if activation is None else activation
            torch.nn.ReLU() if activation is None else activation
        )

        W = torch.empty((out_features, in_features), **factory_kwargs)
        b = torch.empty(out_features, **factory_kwargs) if bias else None

        # Using buffer instead of parameter
        self.register_buffer("weight", W)
        self.register_buffer("bias", b)

        # Random Initialization
        self.reset_parameters()

    @torch.no_grad()
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = X.to(self.weight)
        return self.activation(super().forward(X))


# class GaussianKernel(Buffer):
#     def __init__(
#         self, mean: torch.Tensor, sigma: float = 1, device=None, dtype=torch.float
#     ) -> None:
#         super().__init__()
#         self.device = device
#         self.dtype = dtype
#         factory_kwargs = {"device": device, "dtype": dtype}
#         assert len(mean.shape) == 2, "The mean should be a 2D tensor."
#         mean = mean[None, :, :].to(**factory_kwargs)
#         beta = 1 / (2 * (sigma**2))
#         self.register_buffer("mean", mean)
#         self.register_buffer("beta", torch.tensor(beta, **factory_kwargs))
#
#     @torch.no_grad()
#     def forward(self, X: torch.Tensor) -> torch.Tensor:
#         X = torch.square_(torch.cdist(X.to(self.mean), self.mean))
#         return torch.exp_(X.mul_(-self.beta))
#
#     def init(self, X: torch.Tensor, size: Optional[int] = None) -> None:
#         if size is not None:
#             if size <= X.shape[0]:
#                 idx = torch.randperm(size).to(X.device)
#                 X = X[idx]
#             else:
#                 # The buffer size is suggested to be greater than the number of initial samples.
#                 # Generate center vectors randomly
#                 n_require = size - X.shape[0]
#                 W_proj = torch.normal(mean=0, std=1, size=(n_require, X.shape[0])).to(X)
#                 W_proj /= torch.sum(W_proj, dim=0)
#                 X = torch.cat([X, W_proj @ X], dim=0)
#         self.mean = X.to(self.mean)

class ACIL(torch.nn.Module):
    def __init__(
        self,
        backbone_output_size,
        backbone=None,
        buffer_size=8192,
        out_features=0,
        gamma=1e-3,
        device=None,
        dtype=torch.double,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        # self.backbone = backbone
        # self.backbone_output_size = backbone_output_size
        # self.buffer_size = buffer_size
        self.buffer = RandomBuffer(backbone_output_size, buffer_size, **factory_kwargs)     # (64, 8192)
        # self.buffer = GaussianKernel(torch.zeros((buffer_size, backbone_output_size)), sigma=1e-3, **factory_kwargs)
        self.analytic_linear = RecursiveLinear(buffer_size, out_features, gamma, **factory_kwargs)    # (8192, 17)
        self.eval()

    @torch.no_grad()
    def feature_expansion(self, data):
        out = self.buffer(data)     # (bs, 8192)
        return out

    @torch.no_grad()
    def forward(self, data):
        features = self.feature_expansion(data)     # (bs, 8192)
        out = self.analytic_linear(features)        # (bs, 17)
        return out

    @torch.no_grad()
    def fit(self, X, Y):
        X = self.feature_expansion(X)   # (bs, 8192)
        self.analytic_linear.fit(X, Y)

    @torch.no_grad()
    def update(self):
        self.analytic_linear.update()



class DSAL(torch.nn.Module):
    def __init__(
        self,
        backbone_output_size: int,
        backbone: Callable[[torch.Tensor], torch.Tensor] = torch.nn.Flatten(),
        buffer_size: int = 8192,
        out_features: int = 0,
        gamma_main: float = 1e-3,
        gamma_comp: float = 1e-3,
        C: float = 1,
        activation_main: activation_t = torch.relu,
        activation_comp: activation_t = torch.tanh,
        device=None,
        dtype=torch.double,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        # self.backbone = backbone
        # self.buffer_size = buffer_size
        self.buffer = RandomBuffer(
            backbone_output_size,
            buffer_size,
            activation=torch.nn.ReLU(),
            **factory_kwargs
        )
        # The main stream
        self.activation_main = activation_main
        self.main_stream = RecursiveLinear(buffer_size, out_features, gamma_main, **factory_kwargs)
        # The compensation stream
        self.C = C
        self.activation_comp = activation_comp
        self.comp_stream = RecursiveLinear(buffer_size, out_features, gamma_comp, **factory_kwargs)
        self.eval()

    @torch.no_grad()
    def forward(self, data):
        X = self.buffer(data)
        X_main = self.main_stream(self.activation_main(X))
        X_comp = self.comp_stream(self.activation_comp(X))
        return X_main + self.C * X_comp

    @torch.no_grad()
    def fit(self, X, Y, increase_size=0):
        # num_classes = max(self.main_stream.out_features, int(y.max().item()) + 1)
        # Y_main = torch.nn.functional.one_hot(y, num_classes=num_classes)

        X = self.buffer(X)
        Y_main = Y

        # Train the main stream
        X_main = self.activation_main(X)
        self.main_stream.fit(X_main, Y_main)
        self.main_stream.update()

        # Previous label cleansing (PLC)
        Y_comp = Y_main - self.main_stream(X_main)
        Y_comp[:, :-increase_size] = 0

        # Train the compensation stream
        X_comp = self.activation_comp(X)
        self.comp_stream.fit(X_comp, Y_comp)

    @torch.no_grad()
    def update(self):
        self.main_stream.update()
        self.comp_stream.update()



if __name__ == '__main__':

    # backbone = nn.Linear(100, 50)
    # nn.init.kaiming_uniform_(backbone.weight, a=math.sqrt(5))

    model = ACIL(backbone_output_size=10, buffer_size=5000)
    print(model)

    data1 = {}
    data1["obs"] = torch.randn(3, 10)
    # data1["id"] = torch.randint(0, 10, (32,))

    data1["id"] = torch.rand(3, 10)
    # normalized_tensor = random_tensor / random_tensor.sum(dim=1, keepdim=True)
    # data1["id"] = normalized_tensor

    # for GaussianKernel
    # model.buffer.init(data1["obs"], size=5000)

    model.fit(data1["obs"], data1["id"])

    # data2 = {}
    # data2["obs"] = torch.randn(3, 10)
    # data2["id"] = torch.rand(3, 10)
    #
    # model.fit(data2["obs"], data2["id"])

    out = model(data1["obs"])
    print(out)
    # out1 = torch.argmax(out, dim=1)
    # print(out1)
    print(data1["id"])

    # out = model(data2["obs"])
    # print(out)
    # out2 = torch.argmax(out, dim=1)
    # print(out2)
    # print(data2["id"])

    # data3 = {}
    # data3["obs"] = torch.cat((data1["obs"][:16], data2["obs"][16:]), dim=0)
    # data3["id"] = torch.cat((data1["id"][:16], data2["id"][16:]), dim=0)
    # out = model(data3)
    # out3 = torch.argmax(out, dim=1)
    # print(out3)
    # print(data3["id"])

