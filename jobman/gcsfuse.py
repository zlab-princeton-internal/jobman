from pathlib import Path
from jobman.runner import MultiWorkerRunner

install_script = """
set -e
echo "[INFO] Adding gcsfuse apt repo ..."
echo "deb [signed-by=/usr/share/keyrings/cloud.google.asc] https://packages.cloud.google.com/apt gcsfuse-$(lsb_release -c -s) main" | sudo tee /etc/apt/sources.list.d/gcsfuse.list

echo "[INFO] Installing repo key ..."
curl -s https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo tee /usr/share/keyrings/cloud.google.asc >/dev/null

# best-effort unlock dpkg to avoid stuck installs
if sudo fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1; then
    LOCK_PID=$(sudo lsof -t /var/lib/dpkg/lock-frontend || true)
    if [ -n "$LOCK_PID" ]; then
        echo "[WARN] Killing process $LOCK_PID holding dpkg lock"
        sudo kill -9 $LOCK_PID || true
        sleep 2
    fi
fi

echo "[INFO] apt update & install gcsfuse ..."
sudo apt-get update -y && sudo apt-get install -y gcsfuse

if ! command -v gcsfuse >/dev/null 2>&1; then
    echo "[ERROR] gcsfuse not found after install"
    exit 1
fi
"""

mount_script = """
set -e
echo "[INFO] Ensuring mount path exists: {mnt}"
{sudo}mkdir -p {mnt}

echo "[INFO] Mounting bucket '{bucket}' to {mnt} ..."
mountpoint -q {mnt} || {sudo}gcsfuse --implicit-dirs --dir-mode=777 --file-mode=777 --kernel-list-cache-ttl-secs=-1 --metadata-cache-ttl-secs=-1 --stat-cache-max-size-mb=-1 --type-cache-max-size-mb=-1 {allow_other_opt} {cache_opts} {bucket} {mnt}

echo "[INFO] Listing mount ..."
{sudo}ls -la {mnt} || true
"""


def _parse_cache_opts(cache_cfg, logger=None):
    """Parse cache configuration into gcsfuse command line options."""
    if not cache_cfg:
        return '', None

    opts = []

    # Cache directory (e.g., /mnt/ramdisk)
    cache_dir = getattr(cache_cfg, 'dir', None)
    if cache_dir:
        opts.append(f'--cache-dir={cache_dir}')

    # Metadata cache TTL (-1 = infinite)
    metadata_ttl = getattr(cache_cfg, 'metadata_ttl_secs', None)
    if metadata_ttl is not None:
        opts.append(f'--metadata-cache-ttl-secs={metadata_ttl}')

    # Stat cache max size (-1 = unlimited)
    stat_cache_size = getattr(cache_cfg, 'stat_cache_max_size_mb', None)
    if stat_cache_size is not None:
        opts.append(f'--stat-cache-max-size-mb={stat_cache_size}')

    # Type cache max size (-1 = unlimited)
    type_cache_size = getattr(cache_cfg, 'type_cache_max_size_mb', None)
    if type_cache_size is not None:
        opts.append(f'--type-cache-max-size-mb={type_cache_size}')

    # File cache max size (in MB)
    file_cache_size = getattr(cache_cfg, 'file_cache_max_size_mb', None)
    if file_cache_size is not None:
        opts.append(f'--file-cache-max-size-mb={file_cache_size}')

    # Cache file for range read
    cache_range_read = getattr(cache_cfg, 'cache_file_for_range_read', None)
    if cache_range_read is not None:
        opts.append(f'--file-cache-cache-file-for-range-read={str(cache_range_read).lower()}')

    # Enable parallel downloads
    parallel_downloads = getattr(cache_cfg, 'enable_parallel_downloads', None)
    if parallel_downloads is not None:
        opts.append(f'--file-cache-enable-parallel-downloads={str(parallel_downloads).lower()}')

    cache_opts = ' '.join(opts)

    if opts and logger:
        logger.info(f"gcsfuse cache enabled: {cache_opts}")

    # Prefix script runs BEFORE mount to set up cache directory (e.g., ramdisk)
    prefix_script = None
    prefix = getattr(cache_cfg, 'prefix', None)
    if prefix:
        prefix_script = str(prefix)
        if logger:
            logger.info("gcsfuse cache prefix script configured")

    return cache_opts, prefix_script


class GCSFUSE(MultiWorkerRunner):
    def __init__(self, cfg, logger):
        super().__init__(cfg, logger, action='gcsfuse')

        self.bucket = cfg.gcsfuse.bucket_name
        self.mount_path = cfg.gcsfuse.mount_path
        # self.sudo = 'sudo ' if cfg.job.env_type == 'docker' else ''
        self.sudo = ''
        self.allow_other_opt = '-o allow_other' if self.sudo else ''

        # Parse cache configuration for primary mount
        cache_cfg = getattr(cfg.gcsfuse, 'cache', None)
        self.cache_opts, self.prefix_script = _parse_cache_opts(cache_cfg, logger)

        # Additional mounts (same or different bucket, different options)
        # Format: list of {bucket_name, mount_path, cache: {...}}
        self.extra_mounts = []
        extra_mounts_cfg = getattr(cfg.gcsfuse, 'extra_mounts', None)
        if extra_mounts_cfg:
            from omegaconf import OmegaConf
            if OmegaConf.is_config(extra_mounts_cfg):
                extra_mounts_cfg = OmegaConf.to_container(extra_mounts_cfg, resolve=True)

            for mount_cfg in extra_mounts_cfg:
                extra_bucket = mount_cfg.get('bucket_name', self.bucket)
                extra_path = mount_cfg.get('mount_path')
                extra_cache_cfg = mount_cfg.get('cache')

                # Convert cache dict back to object-like access for _parse_cache_opts
                if extra_cache_cfg:
                    from types import SimpleNamespace
                    extra_cache_obj = SimpleNamespace(**extra_cache_cfg)
                else:
                    extra_cache_obj = None

                extra_cache_opts, extra_prefix = _parse_cache_opts(extra_cache_obj, logger)

                self.extra_mounts.append({
                    'bucket': extra_bucket,
                    'mount_path': extra_path,
                    'cache_opts': extra_cache_opts,
                    'prefix_script': extra_prefix,
                })
                logger.info(f"gcsfuse extra mount configured: {extra_bucket} -> {extra_path}")

    def _get_check_steps(self, i):
        yield self._ssh(i, "command -v gcsfuse >/dev/null 2>&1")
        yield self._ssh(i, f"mountpoint -q {self.mount_path}")
        # Check extra mounts
        for mount in self.extra_mounts:
            yield self._ssh(i, f"mountpoint -q {mount['mount_path']}")

    def _get_setup_steps(self, i):
        yield self._ssh(i, install_script)

        # Run prefix script before primary mount (e.g., to create ramdisk)
        if self.prefix_script:
            self.logger.info(f"gcsfuse worker {i}: running prefix script...")
            yield self._ssh(i, self.prefix_script)

        # Primary mount
        yield self._ssh(i, mount_script.format(
            sudo=self.sudo,
            mnt=self.mount_path,
            bucket=self.bucket,
            allow_other_opt=self.allow_other_opt,
            cache_opts=self.cache_opts
        ))

        # Extra mounts
        for mount in self.extra_mounts:
            # Run prefix script for this mount if configured
            if mount.get('prefix_script'):
                self.logger.info(f"gcsfuse worker {i}: running prefix script for {mount['mount_path']}...")
                yield self._ssh(i, mount['prefix_script'])

            yield self._ssh(i, mount_script.format(
                sudo=self.sudo,
                mnt=mount['mount_path'],
                bucket=mount['bucket'],
                allow_other_opt=self.allow_other_opt,
                cache_opts=mount.get('cache_opts', '')
            ))

