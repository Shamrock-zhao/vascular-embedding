
import json

from learning.loader.vessel_loader import VesselPatchLoader, PatchFromFundusImageLoader, PatchesFromMultipleDatasets


def get_loader(name):
    """get_loader

    :param name:
    """
    return {
        'offline': VesselPatchLoader,
        'online': PatchFromFundusImageLoader,
        'online-multiple': PatchesFromMultipleDatasets,
    }[name]

