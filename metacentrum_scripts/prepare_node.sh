#!/bin/bash
#PBS -N batch_job_knn
#PBS -l select=1:ncpus=1:mem=10gb:scratch_ssd=10gb
#PBS -l walltime=1:00:00
#PBS -j oe
#PBS -m ae

##PBS -q gpu

# -j oe ... standard error stream of the job will be merged with the standard output stream
# -m ae ...  mail is sent when the job aborts or terminates

# The source code of this file is based on the following source:
#
# Source web: GitHub
# Link to the source: https://github.com/roman-janik/diploma_thesis_program/blob/main/ner/train_ner_model.sh
# Author: Roman Janík (https://github.com/roman-janik)

trap 'clean_scratch' TERM EXIT

HOMEPATH=/storage/praha1/home/$PBS_O_LOGNAME
DATAPATH=$HOMEPATH/llms/datasets/            # folder with datasets
RESPATH=$HOMEPATH/llms/results/      # store results in this folder
HOSTNAME=$(hostname -f)                 # hostname of local machine

printf "\-----------------------------------------------------------\n"
printf "JOB ID:             %s\n" "$PBS_JOBID"
printf "JOB NAME:           %s\n" "$PBS_JOBNAME"
printf "JOB SERVER NODE:    %s\n" "$HOSTNAME"
printf "START TIME:         %s\n" "$(date +%Y-%m-%d-%H-%M)"
printf "GIT BRANCH:         $branch\n"
printf "\-----------------------------------------------------------\n"

start_time=$(date +%s)

cd "$SCRATCHDIR" || exit 2

# clean the SCRATCH directory
clean_scratch

# Clone the repository
printf "Cloning the repository ...\n"
cp "$HOMEPATH"/.ssh/id_ed25519 "$HOMEPATH"/.ssh/known_hosts "$HOME"/.ssh
printf "Print content of .ssh dir\n"
ls -la "$HOME"/.ssh
mkdir llms
cd llms || exit 2
git clone git@github.com:davidchocholaty/hugging-face-transformers.git
if [ $? != 0 ]; then
  printf "Cloning repository failed!\n"
  exit 1
fi
cd hugging-face-transformers || exit 2
git checkout "$branch"
cd ../..

# Prepare directory with results
printf "Prepare directory with results\n"
if [ ! -d "$HOMEPATH"/llms/results/ ]; then # test if dir exists
  mkdir "$HOMEPATH"/llms/ "$HOMEPATH"/llms/results/
fi

# Prepare local directory with results
mkdir llms/results

# Prepare directory with datasets
printf "Prepare directory with datasets\n"
if [ ! -d "$DATAPATH" ]; then # test if dir exists
  mkdir "$DATAPATH"
fi

# Copy converted datasets if they are created
# It is not needed to create a "datasets" directory locally because it is 
# coppied from the storage including the folder.
cp -R "$DATAPATH" llms

# Prepare environment
printf "Prepare environment\n"
source /cvmfs/software.metacentrum.cz/modulefiles/5.1.0/loadmodules
module load python
python -m venv env
source ./env/bin/activate
mkdir tmp
cd llms/hugging-face-transformers || exit 2
pip install --upgrade pip
# Install PyTorch 2.0 with cuda 11.7
TMPDIR=../../tmp pip install torch==2.0.0 --extra-index-url https://download.pytorch.org/whl/cu113 -r requirements.txt

# Prepare list of configurations
if [ "$config" == "all" ]; then
  config_list="configs/*"
else
  if [ "${config:0:1}" == '[' ]; then # list of configs
    config=${config#*[}
    config=${config%]*}
  fi

  config_list=$(for cfg in $config
  do
    echo "configs/$cfg.yaml"
  done)
fi

# Create all experiment results files
curr_date="$(date +%Y-%m-%d-%H-%M)"
all_exp_results="$RESPATH"all_experiment_results_"$curr_date".txt
touch "$all_exp_results"
all_exp_results_csv="$RESPATH"all_experiment_results_"$curr_date".csv

# Set Hugging Face token. The token is set using the .bashrc file.
huggingface-cli login --token $HUGGINGFACE_TOKEN # --add-to-git-credential

# Run and save results for configs in list of configurations
printf "\nPreparation took %s seconds, starting training...\n" $(($(date +%s) - start_time))

config_idx=0
for config_file in $config_list
do
  config_name=${config_file#*/}
  config_name=${config_name%.*}
  printf -- '-%.0s' {1..180}; printf "\n%s. experiment\n" $config_idx
  printf "\nConfig: %s\n" "$config_name"

  # Start training
  printf "Start running\n"

  # Run the demo script.
  python demo.py --config "$config_file" # --results_csv "$all_exp_results_csv"
  printf "Exit code: %s\n" "$?"

  # Save results
  printf "\nSave results\n"
  new_model_dir=$RESPATH/$(date +%Y-%m-%d-%H-%M)-$config_name-${stime}h
  mkdir "$new_model_dir"
  grep -vx '^Loading.*arrow' ../results/experiment_results.txt > ../results/experiment_results_f.txt # Remove logs from dataset load
  printf -- '-%.0s' {1..180} >> "$all_exp_results"; printf "\n%s. experiment\n" $config_idx >> "$all_exp_results"
  ((config_idx++))
  cat ../results/experiment_results_f.txt >> "$all_exp_results"
  mv ../results/* "$new_model_dir"
  cp "$config_file" "$new_model_dir"
done

# clean the SCRATCH directory
clean_scratch
