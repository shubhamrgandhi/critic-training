#!/bin/bash
# Live view of your SLURM jobs on Babel.
watch -n 2 'squeue -u $USER -o "%.18i %.10P %.40j %.8u %.2t %.10M %.6D %R"'
