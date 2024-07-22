import os
import time
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Tuple
from omegaconf import MISSING, OmegaConf
import wandb

from models.policy_network import PolicyNetwork, PolicyNetworkConfig
from .data_provider import DataProvider, DataProviderConfig
from .utils import setup_seed

@dataclass
class ImitationTrainerConfig:
    run_dir: str = MISSING
    save_model_period: int = 10000
    seed: int = MISSING
    num_iters: int = 80000
    lr: float = 1e-3
    weight_decay: float = 0.
    milestones: Tuple[float] = (8000, 16000, 30000, 50000, 70000, 90000)
    gamma: float = 1.
    debug: bool = False
    verbose: bool = False
    wandb: bool = False

    data: DataProviderConfig = field(default_factory=lambda: DataProviderConfig(
        seed="${..seed}",
        obj_pose_pred_frame_num="${..model.obj_pose_pred_frame_num}",
    ))
    model: PolicyNetworkConfig = field(default_factory=lambda: PolicyNetworkConfig(in_channel="${..data.processor.pc.in_channel}"))

class ImitationTrainer:
    def __init__(self, cfg: ImitationTrainerConfig):
        self.cfg = cfg
        # setup run dir
        os.makedirs(cfg.run_dir, exist_ok=False)
        OmegaConf.save(cfg, os.path.join(cfg.run_dir, "config.yaml"))
        # enforce determinism
        setup_seed(cfg.seed)
        torch.backends.cudnn.deterministic = True
        # data provider
        self.data_provider = DataProvider(cfg.data)
        # device
        self.device = torch.device("cuda")
        # model
        self.model = PolicyNetwork(cfg.model).to(self.device)
        # optimizer
        self.encoder_optimizer = torch.optim.Adam(self.model.encoder.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.policy_optimizer = torch.optim.Adam(
            list(self.model.linear1.parameters())+list(self.model.linear2.parameters())+list(self.model.action_head.parameters())+list(self.model.goal_pred_head.parameters())+list(self.model.obj_pose_pred_head.parameters()),
            lr=0.0003,
            eps=1e-5,
            weight_decay=1e-5
        )
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        # lr_scheduler
        self.encoder_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.encoder_optimizer, milestones=cfg.milestones, gamma=cfg.gamma)
        self.policy_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.policy_optimizer, milestones=[20000, 40000, 60000, 80000], gamma=0.5)

        if cfg.wandb:
            self.run = wandb.init(
                project="Mobile_GenH2R",
                config=cfg,
                name=cfg.run_dir,
                group=cfg.run_dir[:cfg.run_dir.rfind("_")],
            )

    def train(self):
        for train_iter in range(1, self.cfg.num_iters+1):
            iter_start_time = time.time()
            batch_data = self.data_provider.get_batch_data()
            for key in batch_data:
                batch_data[key] = batch_data[key].to(self.device)
            self.encoder_optimizer.zero_grad()
            self.policy_optimizer.zero_grad()
            loss, loss_dict = self.model.compute_loss(batch_data)
            loss.backward()
            self.encoder_optimizer.step()
            self.policy_optimizer.step()
            self.encoder_lr_scheduler.step()
            self.policy_lr_scheduler.step()
            if train_iter%self.cfg.save_model_period == 0:
                checkpoint = {"state_dict": self.model.state_dict()}
                model_path = os.path.join(self.cfg.run_dir, f"iter_{train_iter}.pth")
                torch.save(checkpoint, model_path)
            iter_time = time.time()-iter_start_time
            if self.cfg.wandb:
                self.run.log(loss_dict)
            if self.cfg.verbose:
                print(f"train_iter: {train_iter} iter_time: {iter_time}, loss: {loss.item()} ({loss_dict})")
                if train_iter%1000 == 0:
                    print("", end="", flush=True)

def main():
    from omegaconf import OmegaConf
    default_cfg = OmegaConf.structured(ImitationTrainerConfig)
    cli_cfg = OmegaConf.from_cli()
    cfg: ImitationTrainerConfig = OmegaConf.to_object(OmegaConf.merge(default_cfg, cli_cfg))
    trainer = ImitationTrainer(cfg)
    trainer.train()
    # iter_time: 2.499143600463867, loss: 2.788721799850464
    # iter_time: 0.155534029006958, loss: 2.242992401123047
    # iter_time: 0.14410710334777832, loss: 2.0687994956970215

if __name__ == "__main__":
    main()

"""
CUDA_VISIBLE_DEVICES=0 python -m train.train_imitation data.demo_dir=data/demo/s0/train/omg_sequential seed=0 run_dir=data/models/tmp save_model_period=100 verbose=True

CUDA_VISIBLE_DEVICES=4 python -m train.train_imitation data.demo_dir=/share1/junyu/HRI/genh2r/data/demo/s0/train/chomp/landmark_planning_dart seed=0 run_dir=data/models/s0/tmp/5 verbose=True data.processor.pc.flow_frame_num=3 model.obj_pose_pred_frame_num=0 model.pred_coff=0.5
"""