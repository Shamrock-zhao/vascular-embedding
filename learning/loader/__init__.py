
import json

from learning.loader.vessel_loader import VesselPatchLoader


def get_loader(name):
    """get_loader

    :param name:
    """
    return {
        'vessel': VesselPatchLoader,
    }[name]

