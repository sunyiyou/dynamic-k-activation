import torch.nn as nn

class RouteFcMaxAct(nn.Linear):

    def __init__(self, in_features, out_features, bias=True, topk=5):
        super(RouteFcMaxAct, self).__init__(in_features, out_features, bias)
        self.topk = topk

    def forward(self, input):
        vote = input[:, None, :] * self.weight
        if self.bias is not None:
            out = vote.topk(self.topk, 2)[0].sum(2) + self.bias
        else:
            out = vote.topk(self.topk, 2)[0].sum(2)
        return out
