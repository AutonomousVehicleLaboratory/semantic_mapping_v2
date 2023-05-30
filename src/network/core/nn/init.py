from torch import nn


def xavier_uniform(module):
    if module.weight is not None:
        nn.init.xavier_uniform_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)


def xavier_normal(module):
    if module.weight is not None:
        nn.init.xavier_normal_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)


def kaiming_uniform(module):
    if module.weight is not None:
        nn.init.kaiming_uniform_(module.weight, nonlinearity='leaky_relu')
    if module.bias is not None:
        nn.init.zeros_(module.bias)


def kaiming_normal(module):
    if module.weight is not None:
        nn.init.kaiming_normal_(module.weight, nonlinearity='leaky_relu')
    if module.bias is not None:
        nn.init.zeros_(module.bias)
