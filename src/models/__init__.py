"""
Models module - Contains neural network model definitions
"""
from .dnn import create_dnn, create_dnn2

__all__ = ["create_dnn", "create_dnn2"]


def get_penetwork(*args, **kwargs):
    """
    Lazy import để tránh phụ thuộc PyTorch khi không cần thiết.
    """
    from .sorel_nets import PENetwork

    return PENetwork(*args, **kwargs)


__all__.append("get_penetwork")

