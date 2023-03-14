import os
import sys
from transformers import HfArgumentParser
from transformers.trainer_utils import get_last_checkpoint


def parse_args(parser: HfArgumentParser):
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        arg_objs = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        arg_objs = parser.parse_args_into_dataclasses()
    return arg_objs


def last_checkpoint_handling(training_args, model_args):
    """HF logic for getting last checkpoint/overwriting an existing run
    """
    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(
            training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            print(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
    checkpoint = None
    if last_checkpoint is not None:
        checkpoint = last_checkpoint
    elif os.path.isdir(model_args.model_name_or_path):
        checkpoint = model_args.model_name_or_path
    return checkpoint
