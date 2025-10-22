# 📁 src/core/spiking_transformer/sparse_config.py

SPARSE_CONFIG = {
    "network_architecture": {
        "d_model": 64,
        "n_heads": 4,
        "num_layers": 2,
        "seq_len": 10,
        "d_ff": 256,
        "sparsity_level": 0.9
    },
    
    "spiking_neurons": {
        "threshold": 1.0,
        "decay_rate": 0.9,
        "reset_potential": 0.0,
        "refractory_period": 1,
        "leakage_factor": 0.95
    },
    
    "attention_mechanism": {
        "spike_integration": "membrane_potential",
        "temporal_encoding": "rate_based",
        "spike_communication": "event_driven",
        "attention_span": "full_sequence"
    },
    
    "optimization": {
        "memory_optimization": True,
        "sparse_operations": True,
        "gradient_checkpointing": False,
        "mixed_precision": True,
        "pruning_method": "magnitude_based",
        "pruning_schedule": "one_shot"
    },
    
    "training": {
        "batch_size": 32,
        "learning_rate": 0.001,
        "weight_decay": 0.01,
        "spike_regularization": 0.1,
        "surrogate_gradient": "sigmoid",
        "bptt_steps": 10
    },
    
    "inference": {
        "batch_size": 1,
        "quantization": "int8",
        "kernel_sparsity": 0.8,
        "activation_sparsity": 0.7,
        "memory_allocation": "dynamic"
    },
    
    "monitoring": {
        "spike_monitoring": True,
        "memory_tracking": True,
        "performance_metrics": True,
        "energy_estimation": False
    }
}
