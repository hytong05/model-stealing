"""
Models module - Contains neural network model definitions
"""
from .dnn import create_dnn, create_dnn2, create_dnn2_deeper, create_dnn2_narrower, create_cnn

__all__ = ["create_dnn", "create_dnn2", "create_dnn2_deeper", "create_dnn2_narrower", "create_cnn"]


def get_penetwork(*args, **kwargs):
    """
    Lazy import để tránh phụ thuộc PyTorch khi không cần thiết.
    """
    from .sorel_nets import PENetwork

    return PENetwork(*args, **kwargs)


__all__.append("get_penetwork")

