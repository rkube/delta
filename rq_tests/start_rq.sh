#!/bin/zsh
conda activate delta
cd repos/delta
rq worker high default low -u redis://cori02 
