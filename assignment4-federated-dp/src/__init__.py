"""
Federated Learning with Differential Privacy
LunarTech Assignment 4
"""

__version__ = "1.0.0"
__author__ = "LunarTech Student"

from .models import FederatedNet
from .federated_client import FlowerClient
from .dp_client import DPFlowerClient
from .data_processing import (
    load_breast_cancer_data,
    create_hospital_splits,
    prepare_centralized_data
)
from .training import (
    train_centralized_models,
    run_federated_learning,
    run_dp_federated_learning
)
from .utils import (
    seed_everything,
    calculate_metrics,
    evaluate_global_model
)

__all__ = [
    'FederatedNet',
    'FlowerClient', 
    'DPFlowerClient',
    'load_breast_cancer_data',
    'create_hospital_splits',
    'prepare_centralized_data',
    'train_centralized_models',
    'run_federated_learning',
    'run_dp_federated_learning',
    'seed_everything',
    'calculate_metrics',
    'evaluate_global_model'
]