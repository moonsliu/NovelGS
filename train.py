import argparse, os, sys
import datetime
import pytz
import glob
import time
import shutil
import subprocess
import numpy as np
from PIL import Image
from packaging import version
from omegaconf import OmegaConf

import torch
import torchvision
from torch.utils.data import DataLoader, Dataset

import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.strategies import DeepSpeedStrategy, DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
from pytorch_lightning.utilities import rank_zero_info, rank_zero_only

from src.utils.train_util import instantiate_from_config


@rank_zero_only
def rank_zero_print(*args):
    print(*args)


def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "--finetune_from",
        type=str,
        nargs="?",
        default="",
        help="path to checkpoint to load model state from"
    )
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        default=None,
        help="resume from checkpoint",
    )
    parser.add_argument(
        "--resume_weights_only",
        action="store_true",
        help="only resume model weights",
    )
    parser.add_argument(
        "-b",
        "--base",
        type=str,
        default="base_config.yaml",
        help="path to base configs",
    )
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        default="",
        help="experiment name",
    )
    parser.add_argument(
        "--num_nodes",
        type=int,
        default=1,
        help="number of nodes to use",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="0",
        help="gpu ids to use",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=42,
        help="seed for seed_everything",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        default="logs",
        help="directory for logging dat shit",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="use deepspeed bf16 training",
    )
    parser.add_argument(
        "--data_root", type=str, default=None,
        help="override the Objaverse root in training and validation configs",
    )
    parser.add_argument(
        "--split_file", type=str, default=None,
        help="override the dataset split JSON path",
    )
    return parser


class SetupCallback(Callback):
    def __init__(self, resume, logdir, ckptdir, cfgdir, config):
        super().__init__()
        self.resume = resume
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config

    def on_fit_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            # Create logdirs and save configs
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)

            rank_zero_print("Project config")
            rank_zero_print(OmegaConf.to_yaml(self.config))
            OmegaConf.save(self.config,
                           os.path.join(self.cfgdir, "project.yaml"))

    def on_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch == self.config.model.adjust_epoch:
            optimizer = trainer.optimizers[0]
            for group_idx, lr in self.lr_groups.items():
                optimizer.param_groups[group_idx]['lr'] = self.config.model.base_learning_rate
            print(f"Adjusted learning rates at epoch {self.config.model.adjust_epoch}: {self.lr_groups}")



class ImageLogger(Callback):
    def __init__(self, batch_frequency, max_images, log_images_kwargs=None):
        super().__init__()
        self.batch_freq = batch_frequency
        self.max_images = max_images
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}

    @rank_zero_only
    def log_local(self, save_dir, split, images, global_step, current_epoch):
        root = os.path.join(save_dir, "images", split)
        for k in images:
            grid = torchvision.utils.make_grid(
                images[k], nrow=images[k].shape[0])
            grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid * 255).astype(np.uint8)
            filename = "{}_gs-{:06}_e-{:06}.png".format(
                k,
                global_step,
                current_epoch)
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(grid).save(path)

    def log_img(self, pl_module, batch, split="train"):
        check_idx = pl_module.global_step
        if split == "val":
            should_log = True
        else:
            should_log = self.check_frequency(check_idx)
        if (should_log and (check_idx % self.batch_freq == 0) and
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0):
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images(
                    batch, split=split, **self.log_images_kwargs)

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    images[k] = torch.clamp(images[k], -1., 1.)

            self.log_local(pl_module.logger.save_dir, split, images,
                           pl_module.global_step, pl_module.current_epoch)

            if is_train:
                pl_module.train()

    def check_frequency(self, check_idx):
        if (check_idx % self.batch_freq) == 0 and check_idx > 0:
            return True
        return False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.log_img(pl_module, batch, split="train")

    @rank_zero_only
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=None):
        if pl_module.global_step > 0:
            self.log_img(pl_module, batch, split="val")


class CUDACallback(Callback):
    # see https://github.com/SeanNaren/minGPT/blob/master/mingpt/callback.py
    def on_train_epoch_start(self, trainer, pl_module):
        # Reset the memory use counter
        torch.cuda.reset_peak_memory_stats(trainer.strategy.root_device.index)
        torch.cuda.synchronize(trainer.strategy.root_device.index)
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        torch.cuda.synchronize(trainer.strategy.root_device.index)
        max_memory = torch.cuda.max_memory_allocated(trainer.strategy.root_device.index) / 2 ** 20
        epoch_time = time.time() - self.start_time

        try:
            max_memory = trainer.strategy.reduce(max_memory)
            epoch_time = trainer.strategy.reduce(epoch_time)

            rank_zero_info(f"Average Epoch time: {epoch_time:.2f} seconds")
            rank_zero_info(f"Average Peak memory {max_memory:.2f} MiB")
        except AttributeError:
            pass


class CodeSnapshot(Callback):
    """
    Modified from https://github.com/threestudio-project/threestudio/blob/main/threestudio/utils/callbacks.py#L60
    """
    def __init__(self, savedir):
        self.savedir = savedir

    def get_file_list(self):
        return [
            b.decode()
            for b in set(
                subprocess.check_output(
                    'git ls-files -- ":!:configs/*"', shell=True
                ).splitlines()
            )
            | set(  # hard code, TODO: use config to exclude folders or files
                subprocess.check_output(
                    "git ls-files --others --exclude-standard", shell=True
                ).splitlines()
            )
        ]

    @rank_zero_only
    def save_code_snapshot(self):
        os.makedirs(self.savedir, exist_ok=True)
        for f in self.get_file_list():
            if not os.path.exists(f) or os.path.isdir(f):
                continue
            os.makedirs(os.path.join(self.savedir, os.path.dirname(f)), exist_ok=True)
            shutil.copyfile(f, os.path.join(self.savedir, f))

    def on_fit_start(self, trainer, pl_module):
        try:
            self.save_code_snapshot()
        except:
            rank_zero_info(
                "Code snapshot is not saved. Please make sure you have git installed and are in a git repository."
            )


if __name__ == "__main__":
    # add cwd for convenience and to make classes in this file available when
    # running as `python main.py`
    sys.path.append(os.getcwd())

    parser = get_parser()
    # parser = Trainer.add_argparse_args(parser)
    opt, unknown = parser.parse_known_args()

    cfg_fname = os.path.split(opt.base)[-1]
    cfg_name = os.path.splitext(cfg_fname)[0]
    exp_name = "-" + opt.name if opt.name != "" else ""
    logdir = os.path.join(opt.logdir, cfg_name+exp_name)

    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    codedir = os.path.join(logdir, "code")
    seed_everything(opt.seed)

    # init configs
    config = OmegaConf.load(opt.base)
    if unknown:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(unknown))
    if opt.data_root is not None:
        config.data.params.train.params.root_dir = opt.data_root
        config.data.params.validation.params.root_dir = opt.data_root
    if opt.split_file is not None:
        config.data.params.train.params.file_name = opt.split_file
        config.data.params.validation.params.file_name = opt.split_file
    lightning_config = config.lightning
    trainer_config = lightning_config.trainer

    trainer_config["accelerator"] = "gpu"
    rank_zero_print(f"Running on GPUs {opt.gpus}")
    ngpu = len(opt.gpus.strip(",").split(','))
    rank_zero_print(f"Length of GPUs {ngpu}")
    trainer_config['devices'] = ngpu

    trainer_opt = argparse.Namespace(**trainer_config)
    lightning_config.trainer = trainer_config

    # model
    model = instantiate_from_config(config.model)
    if opt.resume is not None and opt.resume_weights_only:
        model = model.__class__.load_from_checkpoint(opt.resume, **config.model.params)
        # gs_decoder.transformer_encoder.positional_embedding
        # Load the checkpoint
        # checkpoint = torch.load(opt.resume)
        # def resize_weights(weights, new_shape):
        #     return F.interpolate(weights, size=new_shape, mode='bilinear', align_corners=False)

        # Resize the positional embeddings
        # checkpoint_pos_emb = checkpoint['state_dict']['gs_decoder.transformer_encoder.positional_embedding']
        # resized_pos_emb = F.interpolate(checkpoint_pos_emb.permute(0, 2, 1), size=256, mode='linear', align_corners=False).permute(0, 2, 1)
        # checkpoint['state_dict']['gs_decoder.transformer_encoder.positional_embedding'] = resized_pos_emb

        # # Resize the convolution weights
        # conv_weights = checkpoint['state_dict']['gs_decoder.transformer_encoder.conv.weight']
        # resized_conv_weights = F.interpolate(conv_weights, size=(8, 8), mode='bilinear', align_corners=False)
        # checkpoint['state_dict']['gs_decoder.transformer_encoder.conv.weight'] = resized_conv_weights

        # # Resize the weights for each affected layer
        # checkpoint['state_dict']['gs_decoder.out_layer_depth.weight'] = checkpoint['state_dict']['gs_decoder.out_layer_depth.weight'].mean(dim=0, keepdim=True).repeat(1, 1, 1, 1)
        # checkpoint['state_dict']['gs_decoder.out_layer_feature.weight'] = checkpoint['state_dict']['gs_decoder.out_layer_feature.weight'].mean(dim=0, keepdim=True).repeat(3, 1, 1, 1)
        # checkpoint['state_dict']['gs_decoder.out_layer_opacity.weight'] = checkpoint['state_dict']['gs_decoder.out_layer_opacity.weight'].mean(dim=0, keepdim=True).repeat(1, 1, 1, 1)
        # checkpoint['state_dict']['gs_decoder.out_layer_scaling.weight'] = checkpoint['state_dict']['gs_decoder.out_layer_scaling.weight'].mean(dim=0, keepdim=True).repeat(3, 1, 1, 1)
        # checkpoint['state_dict']['gs_decoder.out_layer_rotation.weight'] = checkpoint['state_dict']['gs_decoder.out_layer_rotation.weight'].mean(dim=0, keepdim=True).repeat(4, 1, 1, 1)

        # # Update the model's state dict with the modified checkpoint
        # model.load_state_dict(checkpoint['state_dict'], strict=False)

    model.logdir = logdir

    # trainer and callbacks
    trainer_kwargs = dict()

    # logger
    default_logger_cfg = {
        "target": "pytorch_lightning.loggers.TensorBoardLogger",
        "params": {
            "name": "tensorboard",
            "save_dir": logdir,
            "version": "0",
        }
    }
    logger_cfg = OmegaConf.merge(default_logger_cfg)
    trainer_kwargs["logger"] = instantiate_from_config(logger_cfg)

    # model checkpoint
    default_modelckpt_cfg = {
        "target": "pytorch_lightning.callbacks.ModelCheckpoint",
        "params": {
            "dirpath": ckptdir,
            "filename": "{step:08}",
            "verbose": True,
            "save_last": True,
            "every_n_train_steps": 5000,
            "save_top_k": -1,   # save all checkpoints
        }
    }

    if "modelcheckpoint" in lightning_config:
        modelckpt_cfg = lightning_config.modelcheckpoint
    else:
        modelckpt_cfg = OmegaConf.create()
    modelckpt_cfg = OmegaConf.merge(default_modelckpt_cfg, modelckpt_cfg)

    # callbacks
    default_callbacks_cfg = {
        "setup_callback": {
            "target": "train.SetupCallback",
            "params": {
                "resume": opt.resume,
                "logdir": logdir,
                "ckptdir": ckptdir,
                "cfgdir": cfgdir,
                "config": config,
            }
        },
        "learning_rate_logger": {
            "target": "train.LearningRateMonitor",
            "params": {
                "logging_interval": "step",
            }
        },
        # "set_static_graph_callback": {
        #     "target": "train.SetStaticGraphCallback",
        #     "params": {}
        # }
        # "cuda_callback": {
        #     "target": "train.CUDACallback"
        # },
        # "code_snapshot": {
        #     "target": "train.CodeSnapshot",
        #     "params": {
        #         "savedir": codedir,
        #     }
        # },
    }
    default_callbacks_cfg["checkpoint_callback"] = modelckpt_cfg

    if "callbacks" in lightning_config:
        callbacks_cfg = lightning_config.callbacks
    else:
        callbacks_cfg = OmegaConf.create()
    callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)

    trainer_kwargs["callbacks"] = [
        instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]

    trainer_kwargs["strategy"] = DDPStrategy(find_unused_parameters=False)

    # trainer
    trainer = Trainer(**trainer_config, **trainer_kwargs, num_nodes=opt.num_nodes)
    trainer.logdir = logdir

    # data
    data = instantiate_from_config(config.data)
    data.prepare_data()
    data.setup("fit")

    # configure learning rate
    base_lr = config.model.base_learning_rate
    if 'accumulate_grad_batches' in lightning_config.trainer:
        accumulate_grad_batches = lightning_config.trainer.accumulate_grad_batches
    else:
        accumulate_grad_batches = 1
    rank_zero_print(f"accumulate_grad_batches = {accumulate_grad_batches}")
    lightning_config.trainer.accumulate_grad_batches = accumulate_grad_batches
    model.learning_rate = base_lr
    rank_zero_print("++++ NOT USING LR SCALING ++++")
    rank_zero_print(f"Setting learning rate to {model.learning_rate:.2e}")

    if opt.resume is not None and not opt.resume_weights_only:
        trainer.fit(model, data, ckpt_path=opt.resume)
    else:
        trainer.fit(model, data)