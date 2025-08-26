<h1 align="center">Jobman-v2</h1>

<p align="center">
  <img src="figs/jobman_logo.png" alt="Jobman Logo" width="512"/>
</p>

Jobman-v2 is a modular and extensible job management system for TPU VMs. 

## Installation

In order to use Jobman, you need to make sure `gcloud` is available on your machine in the first place. You may refer to [the official doc](https://cloud.google.com/sdk/docs/install) to do so.  
Afterwards, also install `alpha` and `beta`.
```bash
gcloud components install alpha beta
```

Login with your gcloud account.
```bash
gcloud auth login
gcloud auth application-default login
```

Also make sure tmux has been installed
```bash
tmux -V
```
If not, follow [tmux wiki](https://github.com/tmux/tmux/wiki/Installing) to install tmux.

Lastly, build the jobman package from source.
```bash
python -m pip install --upgrade pip
pip install -e .
```

## Get Started 
Before you start using Jobman, be sure to go through [GET_STARTED.md](GET_STARTED.md). This is vital for you to proceed to run your own jobs.

## Overall Structure
This section differs from the Get Started section as it explains briefly how Jobman works. Basically, each job is viewed as a data structure or a class by Jobman, with
- life cycle, including queueing, running, idle, and dead managed by a centralized data structure `jobman`. Specifically, `jobman` creates and kills tmux sessions to manage the jobs in the backend.
- corresponding tpus, ssh, gcsfuse, and environment config as attributes.
- all logs saved to `jobs/<user_id>/<job_id>/logs`.

### Caveats
- since jobs live as tmux sessions, it's suggested that you run this tool on some remote host instead of some local machine, since tmux sessions may die after you shut down your machine.
- on the other hand, `jobman` lives as several local data files inside of `jobs/.jobman` and uses a lock to maintain the consistency. Therefore, please do not mess up with the files in `jobs/.jobman` unless you know what you're doing (if you cannot find `jobs/.jobman`, it's normal since it'll be created after you run your first job).

## Other Resources

### Simpler TPU request tool
The design concept of Jobman is somewhat complex, but it aims to provide the easiest user interface s.t. users unfamiliar with TPU can quickly get started.  
For a simpler setup tool, you may refer to [`other_resources/ultra_create_tpu.sh`](other_resources/ultra_create_tpu.sh) by Peter Tong.

### Slack Chatbot
Boyang Zheng has also developed a brilliant Slack Chatbot that 1) automatically deletes dead tpu vms 2) profiles daily usage and sends to their Slack Channel. You may refer to it at [`other_resources/slack_chatbot`](other_resources/slack_chatbot).

## Dashboard
Coming soon

## FAQ
1. **Q:** I ran `jobman create <config_path>` but nothing happens. What should I do?  
**A:** Under the hood, `jobman create` creates the job directory and starts the job process with tmux in the backend. If the job process fails, it fails silently since it's in tmux.  
The first debugging step is to run `jobman run <job_id>` where `<job_id>` is the id of the job you just created. This will run the job in the front end. If this stucks as well, please kindly check if `gcloud` command works on your machine.

## Contributions & Feedback
If you have any issues with this project or want to contribute to it, please first open an issue in the `Issues` section. This will be of great help to the maintenance of this project!  
Also, if you would like to contribute to this project, please refer to [CONTRIBUTING.md](CONTRIBUTING.md).

