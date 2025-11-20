"""
W&B Sweep для поиска лучших гиперпараметров CodeT5+
Запуск:
1. python wandb_sweep_config.py  # создает sweep
2. wandb agent YOUR_SWEEP_ID     # запускает поиск
"""

import wandb

# ✅ Конфигурация sweep - что оптимизируем
sweep_config = {
    'method': 'bayes',  # bayesian optimization (умнее чем random/grid)
    'metric': {
        'name': 'final_f1',
        'goal': 'maximize'
    },
    'parameters': {
        # Learning Rate - самый важный параметр
        'learning_rate': {
            'distribution': 'log_uniform_values',
            'min': 1e-5,
            'max': 1e-4,
        },
        
        # Batch size
        'batch_size': {
            'values': [4, 8]
        },
        
        # Weight decay - регуляризация
        'weight_decay': {
            'distribution': 'uniform',
            'min': 0.01,
            'max': 0.1,
        },
        
        # Warmup ratio
        'warmup_ratio': {
            'distribution': 'uniform',
            'min': 0.0,
            'max': 0.1,
        },
        
        # Gradient accumulation
        'grad_acc_steps': {
            'values': [2, 4, 8]
        },
        
        # Learning rate scheduler
        'lr_scheduler': {
            'values': ['linear', 'cosine', 'cosine_with_restarts']
        },
    }
}

# ✅ Advanced конфиг для более агрессивного поиска
sweep_config_aggressive = {
    'method': 'bayes',
    'metric': {
        'name': 'final_f1',
        'goal': 'maximize'
    },
    'early_terminate': {
        'type': 'hyperband',
        'min_iter': 3,
    },
    'parameters': {
        'learning_rate': {
            'distribution': 'log_uniform_values',
            'min': 5e-6,
            'max': 2e-4,
        },
        'batch_size': {
            'values': [2, 4, 8, 16, 32]
        },
        'weight_decay': {
            'distribution': 'log_uniform_values',
            'min': 0.001,
            'max': 0.2,
        },
        'warmup_ratio': {
            'values': [0.0, 0.03, 0.06, 0.1, 0.15]
        },
        'grad_acc_steps': {
            'values': [1, 2, 4, 8, 16]
        },
        'lr_scheduler': {
            'values': ['linear', 'cosine', 'cosine_with_restarts', 'polynomial']
        },
    }
}

if __name__ == "__main__":
    import sys
    
    # Выбор конфигурации
    config = sweep_config_aggressive if '--aggressive' in sys.argv else sweep_config
    
    # Создаем sweep
    sweep_id = wandb.sweep(
        config,
        project='codet5-clone-detection'
    )
    
    print("\n" + "="*60)
    print("🚀 W&B Sweep создан!")
    print("="*60)
    print(f"\n📊 Sweep ID: {sweep_id}")
    print(f"\n🔗 Dashboard: https://wandb.ai/YOUR_USERNAME/codet5-clone-detection/sweeps/{sweep_id}")
    print("\n▶️  Для запуска агента выполните:")
    print(f"\n   wandb agent {sweep_id}")
    print("\n💡 Для запуска нескольких агентов параллельно:")
    print(f"   wandb agent {sweep_id}  # в разных терминалах")
    print("="*60 + "\n")