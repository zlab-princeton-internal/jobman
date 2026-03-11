"""Entry point for `python -m jobman.worker ...`"""
import click
import sys

from .tpu import DEFAULT_TPU_VERSION
from .worker import Worker


@click.command()
@click.option("--tpu-name", required=True)
@click.option("--accelerator", required=True)
@click.option("--zone", required=True)
@click.option("--tpu-version", default=DEFAULT_TPU_VERSION)
@click.option("--pricing", default="spot")
@click.option("--allocation-mode", default="tpu-vm")
@click.option("--startup-script", default=None)
@click.option("--debug", is_flag=True, default=False,
              help="Run interactively with live output and no worker/task log files")
def main(tpu_name, accelerator, zone, tpu_version, pricing, allocation_mode, startup_script, debug):
    w = Worker(
        tpu_name=tpu_name,
        accelerator=accelerator,
        zone=zone,
        tpu_version=tpu_version,
        pricing=pricing,
        allocation_mode=allocation_mode,
        startup_script=startup_script,
        debug=debug,
    )
    w.run()


if __name__ == "__main__":
    main()
