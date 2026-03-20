"""Entry point for `python -m jobman.worker ...`"""
import click
import sys
import time
import traceback

from .tpu import DEFAULT_TPU_VERSION
from .worker import Worker

SUPERVISOR_RESTART_DELAY_SECS = 10


@click.command()
@click.option("--tpu-name", required=True)
@click.option("--accelerator", required=True)
@click.option("--zone", required=True)
@click.option("--tpu-version", default=DEFAULT_TPU_VERSION)
@click.option("--pricing", default="spot")
@click.option("--allocation-mode", default="tpu-vm")
@click.option("--startup-script", default=None)
@click.option("--ssh-user", default=None, help="SSH username for connecting to TPU VMs")
@click.option("--debug", is_flag=True, default=False,
              help="Run interactively with live output and no worker/task log files")
def main(tpu_name, accelerator, zone, tpu_version, pricing, allocation_mode, startup_script, ssh_user, debug):
    while True:
        try:
            w = Worker(
                tpu_name=tpu_name,
                accelerator=accelerator,
                zone=zone,
                tpu_version=tpu_version,
                pricing=pricing,
                allocation_mode=allocation_mode,
                startup_script=startup_script,
                ssh_user=ssh_user,
                debug=debug,
            )
            w.run()
            return
        except KeyboardInterrupt:
            raise
        except BaseException:
            traceback.print_exc()
            print(
                f"[jobman.worker] fatal worker error escaped to top level; "
                f"restarting in {SUPERVISOR_RESTART_DELAY_SECS}s",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(SUPERVISOR_RESTART_DELAY_SECS)


if __name__ == "__main__":
    main()
