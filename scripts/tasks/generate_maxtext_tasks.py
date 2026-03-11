#!/usr/bin/env python3
"""Helpers for generating maxtext task scripts from the base template.

This only rewrites:
1. `#JOBMAN --accelerator=...`
2. `#JOBMAN --zone=...`
3. `#JOBMAN --name=...`
4. `TRAIN_CMD="..."`
"""

from __future__ import annotations

from pathlib import Path


PRETRAIN_PREFIX = {
    "l200": "L200",
    "meta": "Meta",
    "": "",
}


def build_train_cmd(
    model_type: str,
    pretrain_type: str,
    lr: str,
    schedule: str = "s10",
) -> str:
    """Build the MaxText training command for a single variant.

    Example:
        build_train_cmd("llama3.1-4b-depth", "meta", "1e-4", schedule="s30")
        -> 'bash scripts/llama3.1-4b-depth/Meta_S30.sh --lr=1e-4'

        build_train_cmd("llama3.1-4b-depth", "", "3e-4", schedule="s10")
        -> 'bash scripts/llama3.1-4b-depth/S10.sh --lr=3e-4'
    """
    prefix = PRETRAIN_PREFIX[pretrain_type.lower()]
    schedule_token = schedule.upper()
    script_name = f"{prefix}_{schedule_token}.sh" if prefix else f"{schedule_token}.sh"
    return f"bash scripts/{model_type}/{script_name} --lr={lr}"


def build_job_name(
    model_type: str,
    pretrain_type: str,
    lr: str,
    schedule: str = "s10",
) -> str:
    """Build a compact job name from the varied fields."""
    lr_token = lr.replace(".", "p")
    pretrain_label = pretrain_type or "scratch"
    return f"{model_type}-{pretrain_label}-{schedule}-lr-{lr_token}"


def render_task_script(
    template_text: str,
    *,
    accelerator: str,
    zone: str,
    job_name: str,
    train_cmd: str,
) -> str:
    """Return template text with accelerator, zone, job name, and TRAIN_CMD replaced."""
    rendered_lines = []
    for line in template_text.splitlines():
        if line.startswith("#JOBMAN --accelerator="):
            rendered_lines.append(f"#JOBMAN --accelerator={accelerator}")
        elif line.startswith("#JOBMAN --zone="):
            rendered_lines.append(f"#JOBMAN --zone={zone}")
        elif line.startswith("#JOBMAN --name="):
            rendered_lines.append(f"#JOBMAN --name={job_name}")
        elif line.startswith('TRAIN_CMD="'):
            rendered_lines.append(f'TRAIN_CMD="{train_cmd}"')
        else:
            rendered_lines.append(line)
    return "\n".join(rendered_lines) + "\n"


def write_task_script(
    template_path: str | Path,
    output_path: str | Path,
    *,
    accelerator: str,
    zone: str,
    model_type: str,
    pretrain_type: str,
    lr: str,
    schedule: str = "s10",
) -> Path:
    """Generate one task script from the base template."""
    template_path = Path(template_path)
    output_path = Path(output_path)

    job_name = build_job_name(model_type, pretrain_type, lr, schedule=schedule)
    train_cmd = build_train_cmd(model_type, pretrain_type, lr, schedule=schedule)
    rendered = render_task_script(
        template_path.read_text(),
        accelerator=accelerator,
        zone=zone,
        job_name=job_name,
        train_cmd=train_cmd,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered)
    output_path.chmod(0o755)
    return output_path


if __name__ == "__main__":
    template = Path(__file__).with_name("maxtext_train.sh")
    output_dir = Path(__file__).resolve().parents[1] / "generated_scripts"
    
    for model_type in ["llama3.1-4b-depth", "llama3.1-4b-width", "llama3.1-4b-flap"]:
        for length in [10, 30, 50, 250, 500]:
            for pretrain_type in ["", "l200", "meta"]:
                lr = "1e-4" if pretrain_type == "meta" else "3e-4"
                output = output_dir / f"{model_type}_{pretrain_type or 'scratch'}_s{length}_lr_{lr}.sh"
                write_task_script(
                    template,
                    output,
                    accelerator="v4-128",
                    zone="us-central2-b",
                    model_type=model_type,
                    pretrain_type=pretrain_type,
                    lr=lr,
                    schedule=f"s{length}",
                )

                print(f"Wrote example task script to {output}")
