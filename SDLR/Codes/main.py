from urllib.parse import urlparse
import numpy as np
import os
import csv
import random
import torch

import sys
sys.path.append(os.getcwd()[:os.getcwd().rfind("\\") + 1])

import allrank.models.losses as losses
from allrank.config import Config
from allrank.data.dataset_loading import load_libsvm_dataset, create_data_loaders
from allrank.models.model import make_model
from allrank.models.model_utils import get_torch_device, CustomDataParallel
from allrank.training.train_utils import fit
from allrank.utils.command_executor import execute_command
from allrank.utils.experiments import dump_experiment_result, assert_expected_metrics
from allrank.utils.file_utils import create_output_dirs, PathsContainer, copy_local_to_gs
from allrank.utils.ltr_logging import init_logger
from allrank.utils.python_utils import dummy_context_mgr
from argparse import ArgumentParser, Namespace
from attr import asdict
from functools import partial
from pprint import pformat
from torch import optim
import shutil
import pandas as pd

def parse_args(i) -> Namespace:
    parser = ArgumentParser("allRank")
    parser.add_argument("--job-dir", help="Base output path for all experiments", required=False, default= ".\\") # /dataset/MQ2007/Fold1
    parser.add_argument("--run-id", help="Name of this run to be recorded (must be unique within output dir)", required=False, default= os.getcwd() + "\\..\\out"+str(i)+"\\")
    parser.add_argument("--config-file-name", required=False, type=str, help="Name of json file with config", default= "..\\in\\lambdarank_atmax"+str(i)+".json")#mohem

    return parser.parse_args()


def run(i):
    i = i
    # reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)

    args = parse_args(i)

    paths = PathsContainer.from_args(args.job_dir, args.run_id, args.config_file_name)

    create_output_dirs(paths.output_dir)

    logger = init_logger(paths.output_dir)
    logger.info(f"created paths container {paths}")

    # read config
    config = Config.from_json(paths.config_path)
    logger.info("Config:\n {}".format(pformat(vars(config), width=1)))

    output_config_path = os.path.join(paths.output_dir, "used_config.json")
    # execute_command("cp {} {}".format(paths.config_path, output_config_path))

    # train_ds, val_ds
    train_ds, val_ds = load_libsvm_dataset(
        input_path=config.data.path,
        slate_length=config.data.slate_length,
        validation_ds_role=config.data.validation_ds_role,
    )

    Documents_Length = []
    for i in range(len(train_ds.X_by_qid[:])):
        Documents_Length += [train_ds.X_by_qid[i].shape[0]]

    Documents_Length = np.array(Documents_Length)
    print("Max Is In Index ", np.argmax(Documents_Length), " With Amount Of ", np.max(Documents_Length))

    """Max_Index = np.argmax(Documents_Length)
    Max_Index = Max_Index
    Max_Documented_Query = np.array(train_ds.X_by_qid[Max_Index])
    Max_Documented_Query = pd.DataFrame(Max_Documented_Query)
    Max_Documented_Query.to_csv("Max_Documented_Query_istella.csv")

    Max_Label_Query = np.array(train_ds.y_by_qid[Max_Index])
    Max_Label_Query = pd.DataFrame(Max_Label_Query)
    Max_Label_Query.to_csv("Max_Label_Query_istella.csv")"""

    n_features = train_ds.shape[-1]

    assert n_features == val_ds.shape[-1], "Last dimensions of train_ds and val_ds do not match!"

    # train_dl, val_dl
    train_dl, val_dl = create_data_loaders(
        train_ds, val_ds, num_workers=config.data.num_workers, batch_size=config.data.batch_size)

    # gpu support
    dev = get_torch_device()
    logger.info("Model training will execute on {}".format(dev.type))

    # instantiate model
    model = make_model(n_features=n_features, **asdict(config.model, recurse=False))
    if torch.cuda.device_count() > 1:
        model = CustomDataParallel(model)
        logger.info("Model training will be distributed to {} GPUs.".format(torch.cuda.device_count()))
    model.to(dev)

    # load optimizer, loss and LR scheduler
    optimizer = getattr(optim, config.optimizer.name)(params=model.parameters(), **config.optimizer.args)
    loss_func = partial(getattr(losses, config.loss.name), **config.loss.args)
    if config.lr_scheduler.name:
        scheduler = getattr(optim.lr_scheduler, config.lr_scheduler.name)(optimizer, **config.lr_scheduler.args)
    else:
        scheduler = None


    with torch.autograd.detect_anomaly() if config.detect_anomaly else dummy_context_mgr():  # type: ignore
        # run training

        result = fit(
            model=model,
            loss_func=loss_func,
            optimizer=optimizer,
            scheduler=scheduler,
            train_dl=train_dl,
            valid_dl=val_dl,
            config=config,
            device=dev,
            output_dir=paths.output_dir,
            tensorboard_output_path=paths.tensorboard_output_path,
            **asdict(config.training)
        )

    """
    if result == None:
        return result
    """
    dump_experiment_result(args, config, paths.output_dir, result)

    if urlparse(args.job_dir).scheme == "gs":
        copy_local_to_gs(paths.local_base_output_path, args.job_dir)

    assert_expected_metrics(result, config.expected_metrics)


if __name__ == "__main__":
    from allrank import config as conf ###
    for i in range(1,11):
        #if i > 1: break
        #if i > 13 and i < 21:
            #continue
        #i = 22
        print("iteration : ",i)
        conf.BandWidth_LR = 1e-4
        run(i)


