from pathlib import Path
from omegaconf import OmegaConf

from jobman.runner import MultiWorkerRunner

install_cmd = """        
if [ ! -d ~/miniconda ]; then
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p ~/miniconda
fi && \
source ~/miniconda/etc/profile.d/conda.sh
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
conda env create -n {env_name} -f {remote_config_file} --yes
"""

check_cmd = """
if [ -x ~/miniconda/bin/conda ]; then
    source ~/miniconda/etc/profile.d/conda.sh
    conda env list | grep -q '^{env_name}\\s' && exit 0
fi
exit 1
"""

class CONDA(MultiWorkerRunner):
    
    def __init__(self, cfg, logger):
        super().__init__(cfg, logger, action='conda')

        self.env_name = cfg.conda.name
        self.config_file = cfg.conda.config_file
        self.remote_config_file = f"~/{Path(self.config_file).name}"
        
    def _get_check_steps(self, i):
        yield self._ssh(i, check_cmd.format(env_name=self.env_name))
        
    def _get_setup_steps(self, i):
        yield self._scp(i, self.config_file, self.remote_config_file)
        yield self._ssh(i, install_cmd.format(env_name=self.env_name, remote_config_file=self.remote_config_file))

    def patch_command(self, cmd):
        return f'source ~/miniconda/etc/profile.d/conda.sh \
            && conda run -n {self.env_name} bash -c \"{cmd}\"'