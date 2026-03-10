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
def main(tpu_name, accelerator, zone, tpu_version, pricing, allocation_mode):
    w = Worker(
        tpu_name=tpu_name,
        accelerator=accelerator,
        zone=zone,
        tpu_version=tpu_version,
        pricing=pricing,
        allocation_mode=allocation_mode,
    )
    w.run()


if __name__ == "__main__":
    main()
