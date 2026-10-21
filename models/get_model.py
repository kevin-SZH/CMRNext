"""Construction of the CMRNext matching network.
"""

from argparse import Namespace

from models.RAFT.raft import CMRNext


def get_model(_config, img_shape):
    """Get model.

    Args:
        _config: Resolved run options and runtime state.
        img_shape: Requested image dimensions in height-width order.
    """
    if 'fourier_levels' not in _config:
        _config['fourier_levels'] = -1
    raft_args = Namespace()
    raft_args.small = False
    raft_args.iters = 12
    model = CMRNext(raft_args, use_reflectance=_config['use_reflectance'],
                            with_uncertainty=_config['uncertainty'],
                            fourier_levels=_config['fourier_levels'], unc_type=_config['der_type'],
                            unc_freeze=_config["unc_freeze"], context_encoder=_config["context_encoder"])
    return model
