# GPT: Geometry Processing Toolbox

## Overall Structure
```bash
.
├── README.md
├── nn
│   ├── __init__.py
│   ├── data.py
│   ├── logger.py
│   └── trainer.py
├── ops
│   ├── __init__.py
│   ├── chamfer
│   │   ├── __init__.py
│   │   ├── chamfer_distance.cpp
│   │   ├── chamfer_distance.cu
│   │   └── chamfer_distance.py
│   ├── emd_distance
│   │   ├── __init__.py
│   │   ├── emd.cpp
│   │   ├── emd.cu
│   │   └── emd.py
│   └── sampling
│       ├── __init__.py
│       ├── cuda_utils.h
│       ├── sample.py
│       ├── sampling.cpp
│       ├── sampling_cuda.cu
│       └── utils.h
├── render
│   └── blender.py
├── utils
│   ├── __init__.py
│   ├── binvox.py
│   ├── download.py
│   ├── plyfile.py
│   ├── read.py
│   └── write.py
└── visualize
    ├── __init__.py
    ├── plot.py
    └── vis.py
```
