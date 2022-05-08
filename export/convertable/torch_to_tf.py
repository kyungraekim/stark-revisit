def get_conv_weights(layer):
    torch_weights = get_weights(layer, ['weight'])
    torch_weights[0] = torch_weights[0].transpose((2, 3, 1, 0))
    if layer.bias is not None:
        torch_weights += get_weights(layer, ['bias'])
    return torch_weights


def get_bn_weights(layer):
    return get_weights(layer, ['weight', 'bias', 'running_mean', 'running_var'])


def get_linear_weights(layer):
    weights = get_weights(layer, ['weight', 'bias'])
    weights[0] = weights[0].transpose()
    return weights


def get_embedding_weights(layer):
    return get_weights(layer, 'weight')


def get_weights(layer, names):
    return [getattr(layer, name).detach().numpy() for name in names]
