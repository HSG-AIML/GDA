from .ucmerced import UCMercedDataModule
from .eurosat import EuroSATDataModule
from .resisc45 import RESISC45DataModule
from .etci2021 import ETCI2021DataModule
from .benge import BENGEDataModule
from .firerisk import FireRiskDataModule
from .treesatai import TreeSatAIDataModule
from .eurosat_sar import EuroSATSARDataModule
from .deepglobe import DeepGlobeLandCoverDataModule
from .oscd import OSCDDataModule
from .levir import LEVIRDataModule
from .caltech256 import Caltech256DataModule

__all__ = [
    "UCMercedDataModule",
    "EuroSATDataModule",
    "RESISC45DataModule",
    "ETCI2021DataModule",
    "BENGEDataModule",
    "FireRiskDataModule",
    "TreeSatAIDataModule",
    "EuroSATSARDataModule",
    "DeepGlobeLandCoverDataModule",
    "OSCDDataModule",
    "LEVIRDataModule",
    "Caltech256DataModule",
]
