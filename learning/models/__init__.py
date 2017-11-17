from learning.models.unet import *


def get_model(name, num_channels, n_classes, batch_norm=False, dropout=0.2):
    model = _get_model_instance(name)

    if name == 'unet':
        model = model(n_classes=n_classes,
                      is_batchnorm=batch_norm,
                      in_channels=num_channels,
                      is_deconv=False,
                      dropout=dropout)
    else:
        raise 'Model {} not available'.format(name)

    return model



def _get_model_instance(name):
    return {
        'unet': unet,
    }[name]
