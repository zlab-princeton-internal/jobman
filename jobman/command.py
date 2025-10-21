import shlex
from collections.abc import Iterable
from jobman.runner import MultiWorkerRunner

class COMMAND(MultiWorkerRunner):
    
    def __init__(self, cfg, logger, name=None):
        action = name if name is not None else 'command'
        super().__init__(cfg, logger, action=action)
        
        self.base_cmd = cfg.command.cmd
        self.full_cmd = None
        self.workers = self.infer_workers() 
        
    def infer_workers(self):
        accelerator = self.cfg.tpu.accelerator
        num_workers = self.cfg.tpu.num_workers
        worker_spec = self.cfg.command.get("workers", "all")

        if worker_spec == "all":
            return list(range(num_workers))

        elif isinstance(worker_spec, int):
            if not (0 <= worker_spec < num_workers):
                print(f"Invalid worker index: {worker_spec}. Only {num_workers} workers available.", "ERROR")
                return []
            return [worker_spec]

        elif isinstance(worker_spec, Iterable):
            workers = []
            seen = set()
            for w in worker_spec:
                if not isinstance(w, int) or not (0 <= w < num_workers):
                    self.logger.error(f"Invalid worker index in list: {w}. Only {num_workers} workers available.")
                    return []
                if w in seen:
                    self.logger.error(f"Duplicate worker index specified: {w}.")
                    return []
                seen.add(w)
                workers.append(w)
            return workers

        else:
            self.logger.error(f"Invalid type for 'worker': {type(worker_spec)}. Must be 'all', int, or list of int")
            return []
        
    def _get_setup_steps(self, i):
        if self.full_cmd is None:
            self.full_cmd = self.base_cmd
        yield self._ssh(i, self.full_cmd)