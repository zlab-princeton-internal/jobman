# Contribute to JobMan

## Detailed Breakdown of Jobman

See the file structure of the source code at `jobman/`
```bash
.
├── cli.py
├── command.py
├── envs
│   ├── conda.py
│   ├── docker.py
│   ├── __init__.py
│   └── venv.py
├── gcsfuse.py
├── __init__.py
├── jobman.py
├── job.py
├── runner.py
├── ssh.py
├── tpu.py
└── utils.py
```

After restructuring jobman, the key components here are `job.py`, `jobman.py`, `tpu.py`, and `runner.py`.

### Job
In Jobman, each job is viewed as a data structure of class containing:
- attributes including tpu, gcsfuse, ssh, environment, command. The Job class is responsible for managing these resources and retry if necessary.
- an exclusive logging directory `jobs/<user_id>/<job_id>` that stores all the relevant information of this job.
- an entry in `jobs/.jobman/meta.json` that records the existence of the job.
- runs the job process in the backend with `tmux`, and simply stops the tmux session when the user needs to pause the job.

### Jobman (job management system)
Jobman is a prefix for "Job Management System for TPUs", which is responsible for the following:
- bookkeeping the status of all the jobs. Race condition is prevented by adding a lock to `jobs/.jobman/meta.json`.
- assign job ids for jobs automatically. Race condition is prevented by adding a lock to `jobs/.jobman/next_job_id.txt`.
- responsible for managing the whole lifecycle of jobs, including 'create', 'start', 'stop', 'resume', 'delete', 'clean'.

Note it's almost impossible to make jobman a never-ending process, yet once the process is killed, information will be lost. Therefore, jobman lives as the data stored in `jobs/.jobman`. It's also important that you don't mess up with the files in `jobs/.jobman` unless you know what you're doing.

### TPU
TPU is the class that manages the lifecycle of tpus. Jobman supports 2 allocation mode for TPUs: 
- tpu-vm: in this mode, TPU creation request is submitted repeatedly until TPU is alive.
- queued-resources: in this mode, a queue request is created, and the process goes dormant, and detects TPU VM status repeatedly until it's alive.

Note the class always checks if the tpu-vm is already alive or in creating process. Otherwise the existing TPU will be viewed as "unrecoverable", and simply deletes and re-request TPUs.

### Runner
Lastly, all other modules of Job are abstracted as `MultiWorkerRunner`, which submits the command to all TPU hosts and run them in parallel. There are 2 assumptions behind this generic class:
- each module requires 2 functions: check and setup. `check` is used to check if the module has already been settled, and skips if that's the case. `setup` is used for set up.
- the atomic commands we send to the TPUs are either `scp` command or `ssh` command.

With those assumptions, Jobman uses highly modular implementation of the modules. All parallel execution logics are at `jobman/runner.py`, and the sub-classes only need to provide `_get_check_steps` and `_get_setup_steps`. For example:
```python
# jobman/envs/conda.py
def _get_check_steps(self, i):
    yield self._ssh(i, check_cmd.format(env_name=self.env_name))
    
def _get_setup_steps(self, i):
    yield self._scp(i, self.config_file, self.remote_config_file)
    yield self._ssh(i, install_cmd.format(env_name=self.env_name, remote_config_file=self.remote_config_file))
```
Note `yield` is used because runner execute one step at a time, and exits immediately once a step fails.

## Development Roadmap
- [x] add profiler for account storage
- [x] add profiler for billing
- [x] add profiler for usage/quota
- [ ] add an email notifier of job status
- [x] add user separation
- [x] add unit tests

## Contribution Guidelines
Here is a simple guideline to contribute to Jobman.
- You can try to complete existing features, or propose new features. 
- For bug reports or if you don't understand the implementation of Jobman, open issues and contact [Yufeng](mailto:yx3038@nyu.edu).
- Open PRs if you want to make code contributions.