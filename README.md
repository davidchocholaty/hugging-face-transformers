# Hugging Face Transformers Demo

This repository contains example scripts and configurations for running text generation using Hugging Face Transformers, including a demo pipeline for experimenting with pre-trained models.

## Repository Structure

```
.
├── configs/                 # Configuration files for experiments and models
├── metacentrum_scripts/     # Scripts for submitting jobs to the MetaCentrum HPC cluster
├── demo.py                  # Main demo script for text generation using a specified model
├── results/                 # Directory for storing experiment logs and outputs
└── README.md                # This file
```

## Configuration Files
The ```configs/``` folder contains YAML configuration files defining experiment setups.
Each config typically includes:

- model: path or name of the Hugging Face model
- datasets: optional datasets for training or evaluation (not used for text generation)
- training: parameters such as batch size, number of epochs, optimizer settings (not used for text generation)

Example YAML (```configs/demo_mistral_Mistral-7B-Instruct-v0.3.yaml```):

```
desc: "Baseline experiment. Learning rate scheduler is linear, training 5 epochs."

model:
  name: "Mistral-7B-Instruct-v0.3"
  desc: "Mistral AI model."
  path: "mistralai/Mistral-7B-Instruct-v0.3"

# Not used yet. Kept for demonstration purposes only.
datasets:
  cnec2:
    name: "CNEC 2.0 CoNLL"
    desc: "Czech Named Entity Corpus 2.0 CoNNL dataset. General-language Czech NER dataset."
    url_path: "https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-3493/cnec2.0_extended.zip"
  medival:
    name: "Medieval text"
    desc: "A Human-Annotated Dataset for Language Modeling and Named Entity Recognition in Medieval Documents"
    url_path: "https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-5024/named-entity-recognition-annotations-large.zip?sequence=2&isAllowed=y"
  wikiann:
    name: "Wikiann"
    desc: "WikiANN (sometimes called PAN-X) is a multilingual named entity recognition dataset consisting of Wikipedia articles annotated"
  slavic:
    name: "Slavic"
    desc: "Slavic documents"
    url_train: "https://bsnlp.cs.helsinki.fi/bsnlp-2021/data/bsnlp2021_train_r1.zip"
    url_test: "https://bsnlp.cs.helsinki.fi/bsnlp-2021/data/bsnlp2021_test_v5.zip"

# Not used yet. Kept for demonstration purposes only.
training:
  num_train_epochs: 5
  batch_size: 32

  optimizer:
    learning_rate: 5e-5
    weight_decay: 0.01
    beta1: 0.9
    beta2: 0.999
    eps: 1e-8
  lr_scheduler:
    name: "linear"
    num_warmup_steps: 0
```

## Scripts for Submitting Jobs
Scripts in this folder help submit experiments to the [MetaCentrum](https://metavo.metacentrum.cz/) HPC cluster.

Typical tasks include:
- Defining job resources (CPU, GPU, memory, walltime)
- Setting up environment modules (Python, CUDA, etc.)
- Executing demo.py with the correct config

### Usage
```
./run_job.sh <branch_name> <config_file_name> <timeout>
```

### Example Usage
```
./run_job.sh main demo_mistral_Mistral-7B-Instruct-v0.3 01:00:00
```

## Main Python Script
The ```demo.py``` file is a standalone Python script for running a text-generation demo using Hugging Face Transformers.

Features:
- Loads configuration from a YAML file
- Initializes a Hugging Face pipeline for text generation
- Logs experiment outputs to ```results/experiment_results.txt```
- Prints and logs generated text outputs

### Example Usage
```
python demo.py --config configs/my_experiment.yaml
```

Workflow:
1. Load configuration from YAML file
2. Initialize the text-generation pipeline using the specified model
3. Generate a response for a sample message
4. Log the generated output to results/experiment_results.txt

### Sample Output:
```
Generated Output: [{'generated_text': "I am an AI assistant designed to help with various tasks."}]
```

## License
This project is licensed under the MIT License.
