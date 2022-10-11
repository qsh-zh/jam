def register_rng():
    from jammy.random.rng import global_rng_registry

    try:
        import torch
        # This will also automatically initialize cuda seeds.
        global_rng_registry.register('torch.manual_seed', lambda: torch.manual_seed)
        global_rng_registry.register('torch.cuda.manual_seed', lambda: torch.cuda.manual_seed)
        global_rng_registry.register('torch.cuda.manual_seed_all', lambda: torch.cuda.manual_seed_all)
    except ImportError:
        pass


def init_main():
    register_rng()