from torchgeo.datasets.geo import (
    GeoDataset,
    IntersectionDataset,
    NonGeoClassificationDataset,
    NonGeoDataset,
    RasterDataset,
    UnionDataset,
    VectorDataset,
)
from torchgeo.datasets.splits import (
    random_bbox_assignment,
    random_bbox_splitting,
    random_grid_cell_assignment,
    roi_split,
    time_series_split,
)
from torchgeo.datasets.utils import (
    BoundingBox,
    concat_samples,
    merge_samples,
    stack_samples,
    unbind_samples,
)
from .ucmerced import UCMerced
from .eurosat import EuroSAT
from .resisc45 import RESISC45
from .etci2021 import ETCI2021
from .benge import BENGE
from .firerisk import FireRisk
from .treesatai import TreeSatAI
from .eurosat_sar import EuroSATSAR
from .deepglobe import DeepGlobeLandCover
from .oscd import OSCD
from .caltech256 import Caltech256Dataset as Caltech256

__all__ = [
    "UCMerced",
    "EuroSAT",
    "RESISC45",
    "ETCI2021",
    "BENGE",
    "FireRisk",
    "TreeSatAI",
    "EuroSATSAR",
    "DeepGlobeLandCover",
    "OSCD",
    "Caltech256",
    # Base classes
    "GeoDataset",
    "IntersectionDataset",
    "NonGeoClassificationDataset",
    "NonGeoDataset",
    "RasterDataset",
    "UnionDataset",
    "VectorDataset",
    # Utilities
    "BoundingBox",
    "concat_samples",
    "merge_samples",
    "stack_samples",
    "unbind_samples",
    # Splits
    "random_bbox_assignment",
    "random_bbox_splitting",
    "random_grid_cell_assignment",
    "roi_split",
    "time_series_split",
]
