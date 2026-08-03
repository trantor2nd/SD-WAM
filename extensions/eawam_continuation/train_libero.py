import runpy

from extensions.eawam import model as legacy_model
from .model import ContinuationErrorAwareFastWAM


if __name__ == "__main__":
    legacy_model.ErrorAwareFastWAM = ContinuationErrorAwareFastWAM
    runpy.run_module("extensions.eawam.train_libero", run_name="__main__")
