from jammy.utils.imp import load_class

def quick_opt(model, lr=0.01, name="Adam"):
    opt_class = load_class(("torch.optim", "Adam"))
    return opt_class(model.parameters(), lr=lr)

