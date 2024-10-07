import argparse
#import datetime
import logging
import os
#import time

# Use a pipeline as a high-level helper
from transformers import pipeline
from yaml import safe_load


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Running configuration file.')
    args = parser.parse_args()
    return args

# Source: https://github.com/bakajstep/KNN_Project2024/blob/00dbd5969104bd57f36bc48f66446d3e3ef50ac5/cnec2_ner_trainer.py#L37C1-L47C24
"""def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        log_msg(f"Number of GPU available: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            log_msg(f"Available GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        log_msg('No GPU available, using the CPU instead.')
        device = torch.device("cpu")

    return device    """

def log_msg(msg: str):
    print(msg)
    logging.info(msg)

# Source: https://github.com/bakajstep/KNN_Project2024/blob/00dbd5969104bd57f36bc48f66446d3e3ef50ac5/cnec2_ner_trainer.py#L162C1-L184C55
"""def log_summary(exp_name: str, config: dict):
    log_msg(
        f"{'Name:':<24}{exp_name.removeprefix('exp_configs_ner/').removesuffix('.yaml')}\n"
        f"{'Description:':<24}{config['desc']}")
    ct = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_msg(
        f"{'Start time:':<24}{ct}\n{'Model:':<24}{config['model']['name']}\n"
        f"{'Datasets:':<24}{[dts['name'] for dts in config['datasets'].values()]}\n")

    cf_t = config["training"]
    log_msg(
        f"Parameters:\n"
        f"{'Num train epochs:':<24}{cf_t['num_train_epochs']}\n"
        f"{'Batch size:':<24}{cf_t['batch_size']}")
    log_msg(
        f"{'Learning rate:':<24}{cf_t['optimizer']['learning_rate']}\n"
        f"{'Weight decay:':<24}{cf_t['optimizer']['weight_decay']}\n"
        f"{'Lr scheduler:':<24}{cf_t['lr_scheduler']['name']}\n"
        f"{'Warmup steps:':<24}{cf_t['lr_scheduler']['num_warmup_steps']}")
    log_msg(
        f"{'Beta1:':<24}{cf_t['optimizer']['beta1']}\n"
        f"{'Beta2:':<24}{cf_t['optimizer']['beta2']}\n"
        f"{'Epsilon:':<24}{cf_t['optimizer']['eps']}")"""


def main():
    #start_time = time.monotonic()
    output_dir = "../results"
    #model_dir = "../results/model"
    args = parse_arguments()

    # Load config file
    with open(args.config, 'r') as config_file:
        config = safe_load(config_file)

    # Start logging, print experiment configuration
    logging.basicConfig(filename=os.path.join(output_dir, "experiment_results.txt"),
                        level=logging.INFO,
                        encoding='utf-8', format='%(message)s')
    log_msg("Experiment summary:\n")
    #log_summary(args.config, config)
    log_msg("-" * 80 + "\n")

    messages = [
        {"role": "user", "content": "Who are you?"},
    ]
    # model: mistralai/Mistral-7B-Instruct-v0.3
    pipe = pipeline("text-generation", model=config["model"]["path"])
    pipe(messages)

if __name__ == "__main__":
    main()
