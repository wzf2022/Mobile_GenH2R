import numpy as np
from numpy.typing import NDArray
import torch
from dataclasses import dataclass
from typing import Tuple, List
from scipy import interpolate
from omegaconf import MISSING
import ipdb
import code

def interpolate_waypoints(waypoints: NDArray[np.float64], length: int, mode: str) -> NDArray[np.float64]:
    " waypoints: (way_points_length, dof) "
    " returned traj: (length, dof) "
    x = np.linspace(0, 1, waypoints.shape[0])
    if mode == "linear":
        f = interpolate.interp1d(x, waypoints, axis=0)
    elif mode == "cubic":
        f = interpolate.CubicSpline(x, waypoints, axis=0, bc_type="clamped")
    else:
        raise NotImplementedError
    t = np.linspace(0, 1, length)
    traj = f(t)
    return traj

@dataclass
class TrajectoryConfig:
    # length: int = 25
    length: int = 35
    time_interval: float = 0.13
    dof: int = 12
    start: Tuple[float] = ()
    end: Tuple[float] = ()
    dofs_optimization_mask: Tuple[bool] = (True,True,True,True,True,True,True,True,True,True,False,False)
    init_mode: str = "linear"
    fix_end: bool = False
    lr: float = 0.001
    device: str = "cuda"

class Trajectory:
    def __init__(self, cfg: TrajectoryConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        assert len(cfg.start) == cfg.dof and len(cfg.end) == cfg.dof        # fix_end = false
        # code.interact(local=dict(globals(), **locals()))

        start, end = np.array(cfg.start), np.array(cfg.end)
        if cfg.init_mode == "random":
            if cfg.fix_end:
                interior_traj = np.random.randn(cfg.length, cfg.dof)
            else:
                interior_traj = np.random.randn(cfg.length-1, cfg.dof)
            traj = np.concatenate([start[None], interior_traj, end[None]], axis=0)
        elif cfg.init_mode == "start":
            if cfg.fix_end:
                traj = np.tile(start, (cfg.length, 1))
            else:
                traj = np.tile(start, (cfg.length-1, 1))
        elif cfg.init_mode in ["linear", "cubic"]:
            waypoints = np.stack([start, end], axis=0)
            if cfg.fix_end:
                traj = interpolate_waypoints(waypoints, cfg.length+2, cfg.init_mode)
            else:
                traj = interpolate_waypoints(waypoints, cfg.length+1, cfg.init_mode)
        else:
            raise NotImplementedError

        self.start = torch.tensor(traj[0], device=self.device) # (dof)
        if cfg.fix_end:
            self.end = torch.tensor(traj[-1], device=self.device) # (dof)
            self.traj = torch.tensor(traj[1:-1], requires_grad=True, device=self.device) # (length, dof)
        else:
            self.traj = torch.tensor(traj[1:], requires_grad=True, device=self.device) # (length, dof)
        self.optimization_mask = torch.ones_like(self.traj, dtype=bool) # (length, dof)
        self.optimization_mask[:, ~torch.tensor(cfg.dofs_optimization_mask)] = False # disable columns with false optimization mask

        self.optimizer = torch.optim.Adam([self.traj], lr=cfg.lr)
    
    def update(self, gradient: torch.DoubleTensor) -> None:
        self.traj.grad = gradient*self.optimization_mask
        self.optimizer.step()
        self.optimizer.zero_grad()
        # self.traj.data[self.optimization_mask] -= gradient[self.optimization_mask]

def debug():
    from omegaconf import OmegaConf
    import code

    # import sys
    # sys.path.append("/share1/junyu/HRI/OMG-Planner")
    # from omg.util import interpolate_waypoints as interpolate_waypoints_omg
    x = np.random.randn(2, 5)
    # data_omg = interpolate_waypoints_omg(x, 30, 5, "linear")
    data = interpolate_waypoints(x, 30, "linear")
    # data[1] == data_omg[0]
    # data[-2] == data_omg[-1]
    code.interact(local=dict(globals(), **locals()))


    default_cfg = OmegaConf.structured(TrajectoryConfig)
    cli_cfg = OmegaConf.from_cli()
    cfg: TrajectoryConfig = OmegaConf.to_object(OmegaConf.merge(default_cfg, cli_cfg))

if __name__ == "__main__":
    debug()

"""
python -m policies.chomp.trajectory

python -m bullet.panda_kitchen_scene -f kitchen0
"""