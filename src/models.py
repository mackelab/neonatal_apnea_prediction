import torch
import torch.nn as nn


class GlobalAveragePooling(nn.Module):
    """
    Simple time series classifier with convolutions and global average pooling
    adapted from https://arxiv.org/abs/1611.06455
    """

    def __init__(
        self,
        in_channel=1,
        hidden_channel=20,
        conv_out_channel=None,
        kernel_size=29,
        dilation=1,
    ):
        super().__init__()
        self.in_channel = in_channel
        self.hidden_channel = hidden_channel
        self.conv_out_channel = (
            hidden_channel if conv_out_channel is None else conv_out_channel
        )
        self.kernel_size = kernel_size
        assert self.kernel_size % 2 == 1
        self.dilation = dilation
        self.padding = (self.dilation * (self.kernel_size - 1)) // 2

        self.conv_pool = nn.Sequential(
            nn.Conv1d(
                in_channels=self.in_channel,
                out_channels=self.hidden_channel,
                kernel_size=self.kernel_size,
                dilation=self.dilation,
                padding=self.padding,
            ),
            nn.BatchNorm1d(self.hidden_channel),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=self.hidden_channel,
                out_channels=self.hidden_channel,
                kernel_size=self.kernel_size,
                dilation=self.dilation,
                padding=self.padding,
            ),
            nn.BatchNorm1d(self.hidden_channel),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=self.hidden_channel,
                out_channels=self.hidden_channel,
                kernel_size=self.kernel_size,
                dilation=self.dilation,
                padding=self.padding,
            ),
            nn.BatchNorm1d(self.hidden_channel),
            nn.ReLU(),
        )
        self.final_conv = nn.Conv1d(
            in_channels=self.hidden_channel,
            out_channels=self.conv_out_channel,
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            padding=self.padding,
        )
        self.lin_comb = nn.Linear(self.hidden_channel, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, sig):
        temp = self.conv_pool(sig)
        last_act = self.final_conv(temp)
        downpool = torch.mean(last_act, dim=2)
        return self.sigmoid(self.lin_comb(downpool))

    def forward_logit(self, sig):
        temp = self.conv_pool(sig)
        last_act = self.final_conv(temp)
        downpool = torch.mean(last_act, dim=2)
        return self.lin_comb(downpool)

    def class_act(self, sig):
        temp = self.conv_pool(sig)
        last_act = self.final_conv(temp)
        last_act = last_act.transpose(1, 2)
        class_act = self.lin_comb(last_act)
        return class_act.transpose(1, 2)

    def only_conv(self, sig):
        temp = self.conv_pool(sig)
        last_act = self.final_conv(temp)
        return last_act


class NeuralAdditiveModel(nn.Module):
    """
    Creates a Neural Additive Model (NAM) given a list of submodules.
    """

    def __init__(self, module_list):
        super().__init__()

        self.module_list = nn.ModuleList(module_list)
        self.multiplier_list = nn.ParameterList(
            [nn.Parameter(torch.Tensor([1.0])) for _mod in module_list]
        )
        self.bias = nn.Parameter(torch.Tensor([0.0]))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_ls):
        assert len(x_ls) == len(self.module_list)
        stacked = torch.stack(
            [
                multiplier * module.forward_logit(x)
                for module, multiplier, x in zip(
                    self.module_list, self.multiplier_list, x_ls
                )
            ]
        )
        return self.sigmoid(stacked.sum(dim=0) + self.bias)

    def compute_logits(self, x_ls):
        assert len(x_ls) == len(self.module_list)
        logit_ls = [
            multiplier * module.forward_logit(x)
            for module, multiplier, x in zip(
                self.module_list, self.multiplier_list, x_ls
            )
        ]
        return logit_ls, self.bias


class WeightedMultilabel(torch.nn.Module):
    """
    Weighted loss for imbalanced data.
    """

    def __init__(self, weights: torch.Tensor):
        super().__init__()
        self.loss = torch.nn.BCELoss(reduction="none")
        self.weights = weights

    def forward(self, outputs, targets):
        weigths_ = self.weights[targets.data.view(-1).long()].view_as(targets)
        loss = self.loss(outputs, targets)
        loss_class_weighted = loss * weigths_
        return loss_class_weighted.mean()


if __name__ == "__main__":
    pass
