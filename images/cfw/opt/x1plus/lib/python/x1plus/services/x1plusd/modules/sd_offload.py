"""
Redirects high-write SD card directories to USB storage via bind mounts.

The Bambu X1 SD card accumulates heavy write traffic from three sources:
  - /sdcard/log/      : syslog rotation (~4 MB every 90 min, continuous)
  - /sdcard/timelapse/: timelapse MP4s written during every print
  - /sdcard/ipcam/    : continuous 240 MB camera recording chunks

When a USB drive is mounted at /media/usb0, this module redirects those
directories to the USB drive using bind mounts. All existing firmware code
continues to use the same /sdcard/* paths transparently. When USB is
removed the bind mounts are lazily unmounted and writes fall back to SD.

Settings:
  sd_offload.enabled  - bool, enables this module (default: true)

[X1PLUS_MODULE_INFO]
module:
  name: sd_offload
  default_enabled: true
[END_X1PLUS_MODULE_INFO]
"""

import asyncio
import logging
import os
import subprocess

logger = logging.getLogger(__name__)

USB_MOUNT      = "/media/usb0"
POLL_INTERVAL  = 15

OFFLOAD_DIRS = [
    "/sdcard/log",
    "/sdcard/timelapse",
    "/sdcard/ipcam",
]

_daemon       = None
_mounted      = set()


def _is_bind_mounted(path):
    try:
        with open("/proc/mounts", "r") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2 and parts[1] == path:
                    return True
    except Exception:
        pass
    return False


def _usb_present():
    try:
        with open("/proc/mounts", "r") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2 and parts[1] == USB_MOUNT:
                    return True
    except Exception:
        pass
    return False


def _mount_dir(sdcard_path):
    usb_path = os.path.join(USB_MOUNT, os.path.basename(sdcard_path))
    try:
        os.makedirs(usb_path, exist_ok=True)
        os.makedirs(sdcard_path, exist_ok=True)
        result = subprocess.run(
            ["mount", "--bind", usb_path, sdcard_path],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            logger.info(f"bind-mounted {sdcard_path} -> {usb_path}")
            _mounted.add(sdcard_path)
        else:
            logger.error(f"bind mount failed for {sdcard_path}: {result.stderr.strip()}")
    except Exception as e:
        logger.error(f"error mounting {sdcard_path}: {e}")


def _umount_dir(sdcard_path):
    try:
        result = subprocess.run(
            ["umount", "-l", sdcard_path],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            logger.info(f"unmounted {sdcard_path}")
        else:
            logger.warning(f"unmount failed for {sdcard_path}: {result.stderr.strip()}")
    except Exception as e:
        logger.error(f"error unmounting {sdcard_path}: {e}")
    _mounted.discard(sdcard_path)


def _setup_mounts():
    for path in OFFLOAD_DIRS:
        if not _is_bind_mounted(path):
            _mount_dir(path)


def _teardown_mounts():
    for path in list(_mounted):
        _umount_dir(path)


class SdOffload:
    def __init__(self, daemon):
        self.daemon = daemon
        self._usb_was_present = False

    async def task(self):
        logger.info("SD offload module started")
        while True:
            enabled = self.daemon.settings.get("sd_offload.enabled", True)
            usb_now = _usb_present()

            if enabled and usb_now and not self._usb_was_present:
                logger.info(f"USB drive detected at {USB_MOUNT}, setting up offload mounts")
                _setup_mounts()

            elif (not usb_now or not enabled) and self._usb_was_present:
                logger.info("USB drive removed or offload disabled, tearing down mounts")
                _teardown_mounts()

            elif enabled and usb_now:
                # Ensure mounts are still in place (e.g. after reboot with USB already plugged in)
                _setup_mounts()

            self._usb_was_present = usb_now and enabled
            await asyncio.sleep(POLL_INTERVAL)


def load(daemon):
    global _daemon
    _daemon = daemon
    daemon.sd_offload = SdOffload(daemon=daemon)


def start():
    asyncio.create_task(_daemon.sd_offload.task())
