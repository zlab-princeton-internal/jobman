from pathlib import Path
from jobman.runner import MultiWorkerRunner

install_cmd = """
sudo usermod -aG docker $USER && sudo systemctl restart docker
newgrp docker <<EONG
docker pull {image}
EONG
"""

class DOCKER(MultiWorkerRunner):
    
    def __init__(self, cfg, logger):
        super().__init__(cfg, logger, action='docker')
        
        self.image = cfg.docker.image
        self.flags = cfg.docker.get('flags', None)
    
    def _get_setup_steps(self, i):
        yield self._ssh(i, install_cmd.format(image=self.image))  
        
    def _get_check_steps(self, i):
        yield self._ssh(i, f"docker image inspect {self.image}")      
        
    def patch_command(self, cmd):
        flags_str = " ".join(self.flags or [])

        docker_cmd = f"docker run {flags_str} {self.image} bash -c \"{cmd}\""

        return docker_cmd
        
