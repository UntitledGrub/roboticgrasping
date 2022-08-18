def get_network(network_name):
    network_name = network_name.lower()
    if network_name == 'tct':
        from .model import TCT
        return TCT
    elif network_name == 'fcnn':
        from .model import FCNN
        return FCNN
    else:
        raise NotImplementedError('Network {} is not implemented'.format(network_name))