def register_rng():
    from jammy.random.rng import global_rng_registry

    try:
        import torch
        # This will also automatically initialize cuda seeds.
        global_rng_registry.register('torch', lambda: torch.manual_seed)
    except ImportError:
        pass


def init_main():
    register_rng()