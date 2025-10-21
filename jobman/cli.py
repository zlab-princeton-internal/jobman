# jobman/cli.py
import os
import click
import subprocess
from pathlib import Path
from omegaconf import OmegaConf

from jobman.jobman import JobMan

from jobman.utils import setup_logger
from jobman.profilers.billing_report import main as run_billing_report
from jobman.profilers.quota_report import main as run_quota_report
from jobman.profilers.storage_report import main as run_storage_report

def get_cfg(job_id):
    jm = JobMan()  
    user = os.environ.get("USER")
    
    with jm.with_meta_lock() as meta:
        if job_id not in meta:
            raise KeyError(f"Job ID not found: {job_id}")
        
    job_meta = meta[job_id]
    owner = job_meta['user']
    if owner != user:
        raise PermissionError(f"Meta owner mismatch for {job_id}: owner={owner}, current_user={user}")
    
    config_path = job_meta['config_path']
    cfg = OmegaConf.load(config_path)
    
    return cfg
        
@click.group()
def cli():
    """JobMan CLI: manage TPU jobs."""
    pass

@cli.command()
@click.argument('config_path', type=click.Path(exists=True))
def create(config_path):
    jm = JobMan()  
    job_id = jm.create_job(config_path)
    jm.start_job(job_id)
    
@cli.command(name="resume")
@click.argument("job_id", type=str)
def resume(job_id):
    """Cancel a running job."""
    jm = JobMan()
    jm.start_job(job_id)
    
@cli.command(name="stop")
@click.argument("job_id", type=str)
def stop(job_id):
    """Cancel a running job."""
    jm = JobMan()
    jm.stop_job(job_id)
    
@cli.command(name="reboot")
@click.argument("job_id", type=str)
def reboot(job_id):
    """Cancel a running job."""
    jm = JobMan()
    jm.stop_job(job_id)
    jm.start_job(job_id)
    
@cli.command(name="delete")
@click.argument("job_id", type=str)
def delete(job_id):
    """Cancel a running job."""
    jm = JobMan()
    jm.delete_job(job_id)
    
@cli.command(name="clean")
@click.argument("job_id", type=str)
def clean(job_id):
    """Cancel a running job."""
    jm = JobMan()
    jm.clean_job(job_id)

@cli.command(name="list")
def list_jobs():
    """List all jobs and their status."""
    jm = JobMan()
    jm.list_jobs()
    
@cli.command()
def billing():
    """Run billing report profiler."""
    run_billing_report()

@cli.command()
def quota():
    """Run quota usage profiler."""
    run_quota_report()

@cli.command()
def storage():
    """Run storage usage profiler."""
    run_storage_report()
    
@cli.command(name="ssh")
@click.argument("job_id")
@click.option("--i", default=0, help="the id of host to ssh")
def ssh(job_id, i=0):
    cfg = get_cfg(job_id)
    remote_user = cfg.job.remote_user
    ips = cfg.tpu.get('ips', [])
    if not ips:
        raise ValueError(f'Host0 IP for Job {job_id} not found!')
    if i >= len(ips):
        raise ValueError(f'Host index {i} out of range for Job {job_id}!')
    host0_ip = next(ip.get("external_ip") for ip in ips if ip.worker == i)
    ssh_cmd = ["ssh", f"{remote_user}@{host0_ip}"]
    subprocess.run(ssh_cmd)
    
@cli.command("run")
@click.argument("job_id")
@click.option("--cmd-only", is_flag=True, help="Run the main command only")
def run(job_id, cmd_only):
    """Run a job by job_id using job.py's argparse main."""
    from jobman.job import Job
    cfg = get_cfg(job_id)
    job = Job(cfg)
    if cmd_only:
        job.execute()
    else:
        job.run()

# @cli.command(name="tpu")
# @click.argument("job_id")
# def tpu(job_id):
#     """Request a TPU for a given job_id."""
#     # 直接调用 tpu.main()，并把参数传进去
#     from jobman.tpu import TPU
#     cfg = get_cfg(job_id)
#     tpu = TPU(cfg)
#     tpu.request()
    
# @cli.command(name="ssh")
# @click.argument("job_id")
# def ssh(job_id):
#     """Request a TPU for a given job_id."""
#     from jobman.ssh import SSH
#     cfg = get_cfg(job_id)
#     ssh = SSH(cfg)
#     ssh.setup()
    
# @cli.command(name="gcsfuse")
# @click.argument("job_id")
# def gcsfuse(job_id):
#     """Request a TPU for a given job_id."""
#     from jobman.gcsfuse import GCSFUSE
#     cfg = get_cfg(job_id)
#     gcsfuse = GCSFUSE(cfg)
#     gcsfuse.setup()
    
# @cli.command(name="docker")
# @click.argument("job_id")
# def docker(job_id):
#     """Request a TPU for a given job_id."""
#     from jobman.envs.docker import DOCKER
#     cfg = get_cfg(job_id)
#     logger = setup_logger(log_file=Path(cfg.job.dir)/'logs'/'job.log')
#     docker = DOCKER(cfg, logger)
#     docker.setup()
  
# @cli.command(name="conda")
# @click.argument("job_id")
# def conda(job_id):
#     """Request a TPU for a given job_id."""
#     from jobman.envs.conda import CONDA
#     cfg = get_cfg(job_id)
#     logger = setup_logger(log_file=Path(cfg.job.dir)/'logs'/'job.log')
#     conda = CONDA(cfg, logger)
#     conda.setup()  

# @cli.command(name="venv")
# @click.argument("job_id")
# def venv(job_id):
#     """Request a TPU for a given job_id."""
#     from jobman.envs.venv import VENV
#     cfg = get_cfg(job_id)
#     logger = setup_logger(log_file=Path(cfg.job.dir)/'logs'/'job.log')
#     venv = VENV(cfg, logger)
#     venv.setup()  