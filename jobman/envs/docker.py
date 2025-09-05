from pathlib import Path
from jobman.runner import MultiWorkerRunner

install_cmd = """
sudo usermod -aG docker $USER && sudo systemctl restart docker
docker pull {image}
"""

class DOCKER(MultiWorkerRunner):
    
    def __init__(self, cfg, logger):
        super().__init__(cfg, logger, action='docker')
        
        self.image = cfg.docker.image
        self.env_vars = cfg.docker.get('env_vars', [])
        self.mount_dirs = cfg.docker.get('mount_dirs', [])
        self.workdir = cfg.docker.get('work_dir', None)
        self.flags = cfg.docker.get('flags', None)
    
    def _get_setup_steps(self, i):
        yield self._ssh(i, install_cmd.format(image=self.image))  
        
    def _get_check_steps(self, i):
        yield self._ssh(i, f"sudo docker image inspect {self.image}")      
        
    def patch_command(self, cmd):

        var_flags = []
        volume_flags = []
        for e in self.env_vars:
            assert "=" in e, "expecting format <var_name>=<var_value>"
            var_flags.append(f"-e {e}")
        var_flags_str = " ".join(var_flags)
        
        for d in self.mount_dirs:
            d = str(Path(d).expanduser())
            if ":" in d:
                host_path, container_path = d.split(":", 1)
            else:
                host_path = container_path = d
            volume_flags.append(f"-v {host_path}:{container_path}")
        
        volume_flags_str = " ".join(volume_flags)
        workdir_flag = f"-w {self.workdir}" if self.workdir else ""
        flags_str = " ".join(self.flags or [])

        docker_cmd = f"docker run {flags_str} {var_flags_str} {volume_flags_str} {workdir_flag} {self.image} bash -c \"{cmd}\""

        return docker_cmd
        
