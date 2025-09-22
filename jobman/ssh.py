import os
import shlex
from pathlib import Path
from textwrap import dedent
from jobman.runner import MultiWorkerRunner

# setup_cmd = """
# echo "[INFO] Generating google_compute_engine key..."
# ssh-keygen -t rsa -f ~/.ssh/google_compute_engine -C "$USER" -N ""
# EMAIL=$(gcloud config get-value account)
# gcloud compute os-login ssh-keys add --key-file ~/.ssh/google_compute_engine.pub --ttl=0
# """

class SSH(MultiWorkerRunner):
    
    def __init__(self, cfg, logger):
        super().__init__(cfg, logger, action='ssh')
        
        self.identities = self.cfg.ssh.get("identities", None)
        
    def _get_check_steps(self, i):
        for entry in self.identities:
            pri = Path(entry.private_key).expanduser()
            pub = Path(entry.public_key).expanduser()
            yield self._ssh(i, f"test -f ~/.ssh/{pri.name} && ls -l ~/.ssh/{pri.name} | grep -q 'rw-------'")
            yield self._ssh(i, f"test -f ~/.ssh/{pub.name} && ls -l ~/.ssh/{pub.name} | grep -q 'rw-r--r--'")
        yield self._ssh(i, "test -f ~/.ssh/config && grep -q 'Host' ~/.ssh/config")

    def _get_setup_steps(self, i):
        if self.identities is None:
            yield 0; return
        
        combined_config = ""
        yield self._ssh(i, "mkdir -p ~/.ssh && chmod 700 ~/.ssh")
        for entry in self.identities:
            pri = Path(entry.private_key).expanduser()
            pub = Path(entry.public_key).expanduser()
            config_entry = dedent(entry.config_entry).strip()
            
            if not pri.exists() or not pub.exists():
                yield 1; continue
            
            yield self._scp(i, str(pri), "~/.ssh")
            yield self._scp(i, str(pub), "~/.ssh")
            yield self._ssh(i, f"chmod 600 ~/.ssh/{pri.name} && chmod 644 ~/.ssh/{pub.name}")
            
            combined_config += config_entry + "\n\n"
            
        escaped_config = combined_config.strip().replace('"', '\\"').replace('\n', '\\n')
        yield self._ssh(i, f"printf '{escaped_config}\\n' > ~/.ssh/config")
        yield self._ssh(i, 'chmod 600 ~/.ssh/config')    
        # yield self._ssh(i, setup_cmd)