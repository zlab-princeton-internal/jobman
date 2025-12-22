import shlex
from collections.abc import Iterable
from jobman.runner import MultiWorkerRunner

class COMMAND(MultiWorkerRunner):
    
    def __init__(self, cfg, logger):
        super().__init__(cfg, logger, action='command')
        
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
                    log(f"Invalid worker index in list: {w}. Only {num_workers} workers available.", "ERROR")
                    return []
                if w in seen:
                    log(f"Duplicate worker index specified: {w}.", "ERROR")
                    return []
                seen.add(w)
                workers.append(w)
            return workers

        else:
            log(f"Invalid type for 'worker': {type(worker_spec)}. Must be 'all', int, or list of int.", "ERROR")
            return []
        
    def _get_setup_steps(self, i):
        if self.full_cmd is None:
            self.full_cmd = self.base_cmd
        # Fewer retries for task execution - fail faster on preemption
        from jobman.runner import SSH_TASK_RETRIES
        yield self._ssh(i, self.full_cmd, max_retries=SSH_TASK_RETRIES)