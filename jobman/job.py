import os
import json
import time
import logging
import subprocess
from pathlib import Path
from omegaconf import OmegaConf

from jobman.tpu import TPU
from jobman.ssh import SSH
from jobman.gcsfuse import GCSFUSE 
from jobman.envs.docker import DOCKER
from jobman.envs.conda import CONDA 
from jobman.envs.venv import VENV
from jobman.command import COMMAND

from jobman.utils import setup_logger

class Job:
    
    def __init__(self, cfg):
        
        self.cfg = cfg
       
        self.id = cfg.job.id
        self.name = cfg.job.name
        self.dir = cfg.job.dir
        self.loop = cfg.job.get('loop', False)
       
        self.logger = setup_logger(log_file=Path(self.dir) / 'logs' / 'job.log')
        self.tpu = TPU(cfg)
        self.ssh = SSH(cfg, self.logger) if getattr(cfg, "ssh", None) is not None else None
        self.gcsfuse = GCSFUSE(cfg, self.logger) if getattr(cfg, "gcsfuse", None) is not None else None
        self.command = COMMAND(cfg, self.logger) if getattr(cfg, "command", None) is not None else None
        
        self.env_type = getattr(cfg.job, "env_type", None)
        if self.env_type is None:
            self.env = None
        elif self.env_type == 'docker':
            self.env = DOCKER(cfg, self.logger) if getattr(cfg, "docker", None) is not None else None
        elif self.env_type == 'conda':
            self.env = CONDA(cfg, self.logger) if getattr(cfg, "conda", None) is not None else None
        elif self.env_type == 'venv':
            self.env = VENV(cfg, self.logger) if getattr(cfg, "venv", None) is not None else None
        else:
            raise Exception(f"Unrecognized env_type {self.env_type}")

    def request(self):
        self.logger.info("Checking TPU status...")
        ready = self.tpu.check_and_maybe_delete()
        if ready:
            self.cfg.tpu.ips = self.tpu.get_ips()
            OmegaConf.save(self.cfg, Path(self.dir) / "config.yaml")
            return True
         
        self.logger.info("Requesting TPU...")
        success = self.tpu.request()
        if not success:
            self.logger.error("TPU allocation failed.")
            return False
        
        self.cfg.tpu.ips = self.tpu.get_ips()
        OmegaConf.save(self.cfg, Path(self.dir) / "config.yaml")
        return True

    def setup(self):
        for module in [self.ssh, self.gcsfuse, self.env]:
            if module and not module.setup():
                return False
        
        return True
    
    def execute(self):
        self.command.full_cmd = self.env.patch_command(self.command.base_cmd) if self.env else self.command.base_cmd
        return self.command.setup()
    
    def run(self):
        while True:
            try:
                if not self.request():
                    if not self.loop:
                        return False
                    continue

                if not self.setup():
                    self.logger.error(f"Job {self.id} setup failed.")
                    if not self.loop:
                        return False
                    continue  # try again

                if not self.execute():
                    self.logger.error(f"Job {self.id} execution failed.")
                    if not self.loop:
                        return False
                    continue  # try again

                self.logger.info(f"Job {self.id} finished successfully.")

            except KeyboardInterrupt:
                self.logger.warning("Job interrupted by user")
                return False

            except Exception as e:
                self.logger.exception(f"Job failed with error: {e}")
            
            if not self.loop:
                return False
            self.logger.info("Retrying job due to error...")
            
    def delete(self):
        self.logger = setup_logger(stdout=True)
        self.logger.info(f"Deleting job {self.id}...")

        try:
            self.tpu.delete()
            self.logger.info(f"Deleted TPU for job {self.id}")
        except Exception as e:
            self.logger.warning(f"Failed to delete TPU for job {self.id}: {e}")


        
        