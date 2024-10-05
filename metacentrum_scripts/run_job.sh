#!/bin/bash

# Script for running on the Metacentrum.

# Source: https://github.com/roman-janik/diploma_thesis_program/blob/main/ner/start_training_ner.sh

# Arguments:
# 1. git branch name
# 2. config name (no .yaml extension)
# 3. walltime in format HH:MM:SS

BRANCHNAME=$1
CONFIG=$2
JTIMEOUT=$3
SHOUR=$(echo "$JTIMEOUT" | cut -d: -f1)
STIME=$((SHOUR - 1))

qsub -v branch="$BRANCHNAME",stime="$STIME",config="$CONFIG" -l walltime="$JTIMEOUT" ./prepare_node.sh
