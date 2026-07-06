import hydra
from omegaconf import DictConfig
import wandb
from loguru import logger
from datetime import datetime
from pathlib import Path
import sys
import os
from omegaconf import OmegaConf
from dotenv import load_dotenv

from model.Label.label_runner import Label_Runner
from model.Insight.insight_runner import Insight_Runner
from model.Manage.manage_runner import Manage_Runner
from model.InPredict.inpredict_runner import InPredict_Runner
from utils.core_utils import (
    copy_config_file,
    calculate_md5
)

log_path = Path(f'log/{datetime.now().strftime("%m%d-%H%M%S")}')

class Inferrer():
    def __init__(self, cfg: DictConfig, cid: str, log_dir: Path):
        self.cfg = cfg
        
        match cfg.task:
            case "label":
                self.runner = Label_Runner(
                    cfg=self.cfg,
                    cid=cid,
                    log_dir=log_dir,
                )
            case "insight":
                self.runner = Insight_Runner(
                    cfg=self.cfg,
                    cid=cid,
                    log_dir=log_dir,
                )
            case "manage":
                self.runner = Manage_Runner(
                    cfg=self.cfg,
                )
            case "inpredict":
                self.runner = InPredict_Runner(
                    cfg=self.cfg,
                    cid=cid,
                    log_dir=log_dir,
                )
            case _:
                raise NotImplementedError(f"Task {cfg.task} not implemented")

    def run(self):
        num_processes = OmegaConf.select(self.cfg, "num_process", default=int(os.getenv("NUM_PROCESSES")))
        self.runner.run(num_processes=num_processes)
        self.runner.log_result()

@hydra.main(version_base=None, config_path="config", config_name="label_qwen_FakeTT")
def main(cfg: DictConfig):
    config_str = OmegaConf.to_yaml(cfg)
    config_md5 = calculate_md5(config_str)[:8]

    logger.remove()
    log_dir = Path(f'log/{cfg.task}-{cfg.model_short_name}-{cfg.dataset}/{config_md5}')
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f'{datetime.now().strftime("%m%d-%H%M%S")}.log'

    debug_mode = OmegaConf.select(cfg, "debug", default=False)
    log_level = "DEBUG" if debug_mode else "INFO"
    
    logger.add(log_path, level="DEBUG")
    logger.add(sys.stdout, level=log_level)
    logger.info(OmegaConf.to_yaml(cfg))
    logger.info(f"Log Path: {log_path}")
    logger.info(f"Debug mode: {debug_mode}")
    
    
    tags = []

    inferrer = Inferrer(cfg, cid=config_md5, log_dir=log_dir)
    inferrer.run()

if __name__ == "__main__":
    load_dotenv()
    copy_config_file()
    main()