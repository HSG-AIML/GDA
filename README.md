# Parameter Efficient Self-Supervised Geospatial Domain Adaptation

This repository contains code supporting the CVPR 2024 paper "Parameter Efficient Self-supervised Geospatial Domain Adaptation".



## Versions
This assumes:
* `torchgeo==0.5.0`
* `timm==0.6.12`
* `torch==2.0.1`

## Notes
### ETCI2021
* I removed the water mask, only predict the flood mask
* The two masks sometimes overlap -> should be a multi-label problem


