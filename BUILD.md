# Build Instructions

## Prerequisites
- Make
- Python 3
- Required Python development tools

## Building

1. Build Python extensions:
```bash
python3 setup.py build_ext --inplace
```

2. Import in Python:
```bash
import APDNN
```

## For Using Backend Directly in C++

For compiling the C++ file, use the makefile. Change the target to the c++ file name without extension
```makefile
TARGET = <filename>
```

Then run make:
```bash
make
```