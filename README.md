# Build Instructions

## Prerequisites
- Python 3

## Building

1. Build Python extensions:
```bash
python3 setup.py build_ext --inplace
```

2. Import in Python:
```bash
import APDNN
```

## Usage

- To use model1:
```bash
python3 cnn_model.py <data_dir>
```

- To use model2:
```bash
python3 cnn_model2.py <data_dir>
```