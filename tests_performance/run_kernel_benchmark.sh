#!/bin/bash

export OMP_NUM_THREADS=1
python performance_coherence.py

export OMP_NUM_THREADS=2
python performance_coherence.py

export OMP_NUM_THREADS=4
python performance_coherence.py

export OMP_NUM_THREADS=8
python performance_coherence.py

export OMP_NUM_THREADS=16
python performance_coherence.py

export OMP_NUM_THREADS=32
python performance_coherence.py

