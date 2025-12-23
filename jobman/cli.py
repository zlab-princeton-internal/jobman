import os
import pdb
import json
import click
import subprocess
from pathlib import Path
from tabulate import tabulate
from omegaconf import OmegaConf
from jobman.jobman import JobMan
from jobman.tpu import TPU

        
@click.group()
def cli():
    """JobMan CLI: manage TPU jobs."""
    pass

@cli.command(name="create_tpu")
@click.argument("config_path", required=True, type=str)
def create_tpu(config_path):
    jm = JobMan()
    tpu_id = jm.create_tpu(config_path)

@cli.command(name="delete_tpu")
@click.argument("tpu_id", required=True, type=str)
def delete_tpu(tpu_id):
    jm = JobMan()
    tpu_id = jm.delete_tpu(tpu_id)
    
@cli.command(name="list_tpus")
def list_tpus():
    jm = JobMan()
    tpus = jm.list_queues()
    if not tpus:
        click.echo("No TPUs found.")
        return
    headers = ["Name", "Zone", "Accelerator", "State"]
    table = [[tpu['name'], tpu['zone'], tpu['acceleratorType'], tpu['state']] for tpu in tpus]
    table = sorted(table, key=lambda x: x[1])
    click.echo(tabulate(table, headers=headers, tablefmt="github"))
    
@cli.command(name="list_tasks")
def list_tasks():
    jm = JobMan()
