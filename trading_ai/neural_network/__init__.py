# ماژول شبکه‌های عصبی - Neural Network Module
from .sparse_network import SparseNeuralNetwork
from .model_trainer import ModelTrainer
from .data_processor import DataProcessor

__all__ = [
    'SparseNeuralNetwork',
    'ModelTrainer', 
    'DataProcessor'
]

__version__ = "1.0.0"
__description__ = "شبکه‌های عصبی اسپارس برای تحلیل بازارهای مالی"
