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
       
        self.tpu = TPU(cfg)
        self.ssh = SSH(cfg)
        self.gcsfuse = GCSFUSE(cfg)
        self.command = COMMAND(cfg)
        
        self.env_type = cfg.job.env_type
        if self.env_type == 'docker':
            self.env = DOCKER(cfg)
        elif self.env_type == 'conda':
            self.env = CONDA(cfg)
        elif self.env_type == 'venv':
            self.env = VENV(cfg)
        else:
            raise ValueError(f"Invalid env type {env_type}")
        
        self.log_file = Path(self.dir) / 'logs' / 'job.log'
        self.logger = setup_logger(log_file=self.log_file)

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
        if not self.ssh.setup():
            return False
        if not self.gcsfuse.setup():
            return False
        if not self.env.setup():
            return False
        
        return True
    
    def execute(self):
        self.command.full_cmd = self.env.patch_command(self.command.base_cmd)
        return self.command.run()
    
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
            
        # if self.log_file.exists():
        #     try:
        #         self.log_file.unlink()
        #         self.logger.info(f"Deleted log file: {self.log_file}")
        #     except Exception as e:
        #         self.logger.warning(f"Failed to delete log file: {e}")


        
        