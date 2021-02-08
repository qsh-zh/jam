try:
    from apex import amp

    APEX_AVAILABLE = True
except:
    APEX_AVAILABLE = False

__all__ = ["loss_backwards"]


def loss_backwards(fp16, loss, optimizer, **kwargs):
    if fp16:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward(**kwargs)
    else:
        loss.backward(**kwargs)
