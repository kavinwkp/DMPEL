import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Optional, Union
from abc import ABCMeta, abstractmethod
import math



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

        # num_targets = Y.shape[1]    # 17
        # if num_targets > self.out_features:     # init 17 > 0
        #     increment_size = num_targets - self.out_features    # 17
        #     print(f"increment_size: {increment_size}")
        #     tail = torch.zeros((self.weight.shape[0], increment_size)).to(self.weight)  # (8192, 17)
        #     self.weight = torch.cat((self.weight, tail), dim=1)     # (8192, 17)
        # elif num_targets < self.out_features:
        #     increment_size = self.out_features - num_targets
        #     tail = torch.zeros((Y.shape[0], increment_size)).to(Y)
        #     Y = torch.cat((Y, tail), dim=1)

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


class ACIL(torch.nn.Module):
    def __init__(
        self,
        backbone_output_size,
        backbone=torch.nn.Flatten(),
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
        # Y = torch.nn.functional.one_hot(y)  # (bs, 50)
        # Y = torch.nn.functional.one_hot(data["id"], 50)  # (bs, 1, 17) -> (bs, 17)
        # X = self.feature_expansion(data["obs"])   # (bs, 8192)
        # Y = data["id"]

        # Y = data["obs"]["id_vector"].squeeze()  # (bs, 1, 17) -> (bs, 17)
        # skill_id = data["obs"]["id"].squeeze()
        # Y = torch.nn.functional.one_hot(skill_id, skill_id.max()+1)   # (bs,) -> (bs, 50)
        X = self.feature_expansion(X)   # (bs, 8192)
        self.analytic_linear.fit(X, Y)

    @torch.no_grad()
    def update(self):
        self.analytic_linear.update()

    def reset(self):
        """
        Clear all "history" of the policy if there exists any.
        """
        self.latent_queue = []
        self.current_subtask_id = 0
        self.counter = 0
        self.prev_subgoal_embedding = None
        for policy in self.backbone.skill_policies.values():
            policy.reset()

    @torch.no_grad()
    def predict(self, data):
        subtask_vector = self.forward(data)
        subtask_id = torch.argmax(subtask_vector, dim=1)    # (bs,)
        subtask_id = subtask_id.cpu().detach().numpy()

        return {"subtask_id": subtask_id}

    @torch.no_grad()
    def get_action(self, data):
        # self.model.eval()
        with torch.no_grad():
            data = self.backbone.preprocess_input(data, train_mode=False)
            # if self.counter % self.freq == 0:
            if self.counter % 10 == 0:
                predict = self.predict(data)["subtask_id"]
                print(predict)
                subtask_id = predict[0]

                # if self.counter < 150:
                #     subtask_id_set = 3
                # else:
                #     subtask_id_set = 2
                #
                # print(subtask_id_set)
                #
                # if self.current_subtask_id != subtask_id_set:
                #     self.current_subtask_id = subtask_id_set
                if self.current_subtask_id != subtask_id:
                    self.current_subtask_id = subtask_id
                    self.backbone.skill_policies[self.current_subtask_id].reset()  # reset the subskill Transformer policy
            self.counter += 1
            data["task_id"] = self.current_subtask_id    # (bs,)
            data["obs"]["joint_states"] = data["obs"]["joint_states"].squeeze(0)
            data["obs"]["agentview_rgb"] = data["obs"]["agentview_rgb"].squeeze(0)   #TODO: change for skill
            data["obs"]["eye_in_hand_rgb"] = data["obs"]["eye_in_hand_rgb"].squeeze(0)  # TODO: change for skill
            action = self.backbone.skill_policies[self.current_subtask_id].get_action(data)

        return action


if __name__ == '__main__':

    backbone = nn.Linear(100, 50)
    nn.init.kaiming_uniform_(backbone.weight, a=math.sqrt(5))

    model = ACIL(backbone_output_size=50, backbone=backbone)
    print(model)

    data1 = {}
    data1["obs"] = torch.randn(3, 100)
    # data1["id"] = torch.randint(0, 10, (32,))

    random_tensor = torch.rand(3, 10)
    # normalized_tensor = random_tensor / random_tensor.sum(dim=1, keepdim=True)
    # data1["id"] = normalized_tensor

    model.fit(data1["obs"], random_tensor)

    # data2 = {}
    # data2["obs"] = torch.randn(32, 100)
    # data2["id"] = torch.randint(0, 17, (32,))
    #
    # model.fit(data2)

    out = model(data1["obs"])
    print(out)
    # out1 = torch.argmax(out, dim=1)
    # print(out1)
    print(random_tensor)

    # out = model(data2)
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

