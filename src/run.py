from omegaconf import DictConfig
from hydra.utils import instantiate

import os

from src.context_matters.logger_cfg import logger
from src.context_matters.pipelines.base_pipeline import BasePipeline
from src.context_matters.agents.agent import Agent

def main(cfg: DictConfig):

    if not os.path.exists(cfg.res_path):
        os.makedirs(cfg.res_path)

    # -------------------- Initialization --------------------

    agent: Agent = instantiate(cfg.agent)

    pipeline: BasePipeline = instantiate(
        cfg.pipeline,
        base_dir=cfg.base_path,
        data_dir=cfg.data_path,
        results_dir=cfg.res_path,
        splits=cfg.splits,
        agent=agent,
        _convert_="all," # Convert all fields to their respective types
    )

    logger.info(
        f"---------------Running the {pipeline.name} Pipeline -------------------\n"
    )
    pipeline.run()