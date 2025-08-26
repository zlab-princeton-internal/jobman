# Contribute to JobMan

## Detailed Breakdown of Jobman

See the file structure of the source code at `jobman/`
```bash
.
├── cli.py
├── command.py
├── envs
│   ├── base.py
│   ├── conda.py
│   ├── docker.py
│   ├── __init__.py
│   └── venv.py
├── gcsfuse.py
├── __init__.py
├── jobman.py
├── job.py
├── profilers
│   ├── billing_report.py
│   ├── __init__.py
│   ├── quota_report.py
│   └── storage_report.py
├── ssh.py
├── tpu.py
└── utils.py
```

### jobman (job management system)

### job

### tpu

### ssh

### gcsfuse

### env

### command

## Development Roadmap
- [x] add profiler for account storage
- [ ] add profiler for billing
- [x] add profiler for usage/quota
- [ ] add an email notifier of job status
- [x] add user separation
- [ ] add unit tests

## Contribution Guidelines