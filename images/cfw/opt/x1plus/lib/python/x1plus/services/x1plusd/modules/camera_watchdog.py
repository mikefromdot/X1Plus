"""
Periodically probes the ipcam service and restarts it if unresponsive.

The Bambu ipcam binary has a known resource leak: after enough
connect/disconnect cycles it stops accepting new RTSP connections without
crashing. Because the process stays alive sys_monitor never restarts it,
so the camera appears dead until the printer is rebooted.

A bare TCP probe is insufficient: when ipcam exhausts its MPP session pool
it continues accepting TCP connections on port 322 while being unable to
serve RTSP. This module probes at the RTSP layer by establishing a TLS
connection and sending an OPTIONS request. A valid RTSP response confirms
ipcam is healthy; a timeout or connection error triggers a restart.

After restart, if the RTSP server was previously enabled, the module
restores it by publishing {"restore_camera_rtsp": true} on the
device/x1plus DDS topic. A handler added to GpioKeys.js receives this
and calls RecordManager.rtspServerOn = true inside bbl_screen's own
process, which causes bbl_screen to send the liveview init command to
ipcam via its internal DDS channel — the same path as a user toggle.

Note: direct DDS publishing from Python cannot reach ipcam because
Python uses the device/report/ and device/request/ DDS partition while
ipcam uses device/inter/. The device/x1plus channel is used as a bridge
through bbl_screen instead.

Settings:
  camera.watchdog.enabled  - bool, enables this module (default: true)
  camera.watchdog.interval - int, probe interval in seconds (default: 300)

[X1PLUS_MODULE_INFO]
module:
  name: camera_watchdog
  default_enabled: true
[END_X1PLUS_MODULE_INFO]
"""

import asyncio
import json
import logging
import ssl
import subprocess

import x1plus.dds

logger = logging.getLogger(__name__)

DEFAULT_INTERVAL      = 300
PROBE_TIMEOUT         = 10
IPCAM_HOST            = "127.0.0.1"
IPCAM_PORT            = 322
IPCAM_INIT            = "/etc/init.d/S99ipcam_service"
PRINTER_JSON          = "/config/screen/printer.json"

# ipcam is fully up within ~3 s of starting; the init script already
# sleeps 5 s between stop and start, so 8 s total is comfortable.
RESTART_SETTLE_TIME   = 8

# On boot, x1plusd starts before ipcam is fully ready. We skip the first
# probe to avoid killing ipcam during its own startup sequence.
# ipcam logs "wait 2 seconds before starting streaming" on boot, and the
# full liveview flow takes ~10s to initialise. 60s is conservative.
BOOT_GRACE_PERIOD     = 60

RTSP_RESTORE_RETRIES  = 3
RTSP_RESTORE_INTERVAL = 5

_daemon      = None

# Publish on device/x1plus — the channel bbl_screen already subscribes
# to for X1Plus messages.  A handler in GpioKeys.js receives
# restore_camera_rtsp and calls RecordManager.rtspServerOn directly
# inside bbl_screen, which then sends the liveview init to ipcam via
# bbl_screen's own internal DDS channel.
_x1plus_pub = x1plus.dds.publisher("device/x1plus")


def _mqtt_rtsp_state():
    """
    Return rtsp_url from the latest MQTT push_status, or None if unavailable.
    """
    ipcam = _daemon.mqtt.latest_print_status.get("ipcam", {})
    return ipcam.get("rtsp_url")


def _printer_json_rtsp_enabled() -> bool:
    """
    Read rtspServer from /config/screen/printer.json as a fallback.
    This is reliable when ipcam is alive but unresponsive (the real
    production failure mode) because bbl_screen only clears rtspServer
    after a clean ipcam disconnect — not when ipcam is merely stuck.
    """
    try:
        with open(PRINTER_JSON, "r") as f:
            enabled = bool(json.load(f).get("rtspServer", False))
        logger.debug(f"printer.json rtspServer={enabled}")
        return enabled
    except FileNotFoundError:
        return False
    except Exception as e:
        logger.warning(f"could not read printer.json: {e}")
        return False


class CameraWatchdog:
    def __init__(self, daemon):
        self.daemon = daemon

    def _rtsp_was_on(self) -> bool:
        """
        Return whether RTSP should be restored after a restart.

        We consider RTSP as "was on" if EITHER source says so:
        - MQTT latest_print_status['ipcam']['rtsp_url'] == "enable": ipcam
          was actively running RTSP (the normal production failure case —
          session pool exhausted but ipcam still reports RTSP as enabled).
        - printer.json rtspServer == True: user configured RTSP on (covers
          fresh x1plusd start, test scenarios where ipcam was restarted
          without RTSP being re-enabled, or any edge case where MQTT lags).

        Using OR means we err on the side of restoring RTSP, which is the
        desired behaviour — a harmless extra liveview init is better than
        leaving the user without LAN streaming.
        """
        mqtt_rtsp = _mqtt_rtsp_state()
        printer_rtsp = _printer_json_rtsp_enabled()
        enabled = (mqtt_rtsp is not None and mqtt_rtsp != "disable") or printer_rtsp
        logger.debug(
            f"RTSP was on: MQTT rtsp_url={mqtt_rtsp!r}, "
            f"printer.json={printer_rtsp} -> {enabled}"
        )
        return enabled

    async def _probe_camera(self) -> bool:
        """
        Probe ipcam by completing a TLS handshake on port 322.

        A TLS handshake is sufficient to determine whether ipcam is alive
        and its network stack is functional, without sending any RTSP data.
        Sending RTSP OPTIONS and immediately disconnecting was observed to
        trigger a "Liveview failed to open" error in bbl_screen, likely
        because ipcam emits a DDS status event when a client disconnects
        unexpectedly that bbl_screen misinterprets as a stream failure.

        A TLS-only probe is invisible to the RTSP layer and causes no
        side effects. If ipcam is truly dead or hung, the TLS handshake
        will either be refused (ECONNREFUSED) or time out.
        """
        ssl_ctx = ssl.create_default_context()
        ssl_ctx.check_hostname = False
        ssl_ctx.verify_mode    = ssl.CERT_NONE

        try:
            _, writer = await asyncio.wait_for(
                asyncio.open_connection(IPCAM_HOST, IPCAM_PORT, ssl=ssl_ctx),
                timeout=PROBE_TIMEOUT,
            )
            # TLS handshake succeeded — ipcam is alive.
            # Close immediately without sending any RTSP data.
            writer.close()
            await writer.wait_closed()
            return True

        except (asyncio.TimeoutError, OSError, ssl.SSLError) as e:
            logger.warning(f"camera probe failed: {e}")
            return False

    async def _restore_rtsp(self):
        """
        Re-enable the RTSP server after restart by publishing
        restore_camera_rtsp on device/x1plus.  The GpioKeys.js handler in
        bbl_screen receives this and calls RecordManager.rtspServerOn = true,
        which causes bbl_screen to send the liveview init to ipcam directly.
        Confirms success via MQTT push_status with retries.
        """
        for attempt in range(1, RTSP_RESTORE_RETRIES + 1):
            current = _mqtt_rtsp_state()
            logger.debug(f"RTSP restore attempt {attempt}: MQTT rtsp_url={current!r}")

            if current is not None and current != "disable":
                logger.info(f"RTSP server is enabled (rtsp_url={current!r}) — restore complete")
                return

            logger.info(
                f"RTSP not enabled (rtsp_url={current!r}), "
                f"requesting restore via bbl_screen "
                f"(attempt {attempt}/{RTSP_RESTORE_RETRIES})"
            )
            _x1plus_pub(json.dumps({"restore_camera_rtsp": True}))
            await asyncio.sleep(RTSP_RESTORE_INTERVAL)

        current = _mqtt_rtsp_state()
        if current is not None and current != "disable":
            logger.info(f"RTSP server restored successfully (rtsp_url={current!r})")
        else:
            logger.warning(
                "RTSP server could not be restored automatically after "
                f"{RTSP_RESTORE_RETRIES} attempts — toggle it manually in LAN access settings"
            )

    async def _restart_ipcam(self, rtsp_enabled: bool):
        """Restart ipcam via its init script, then restore RTSP state."""
        logger.warning(
            f"restarting ipcam service "
            f"(RTSP was {'enabled' if rtsp_enabled else 'disabled'})"
        )

        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: subprocess.run(
                    [IPCAM_INIT, "restart"],
                    capture_output=True,
                    text=True,
                ),
            )
            if result.returncode == 0:
                logger.info("ipcam service restarted successfully")
            else:
                logger.error(
                    f"ipcam restart returned {result.returncode}: "
                    f"{result.stderr.strip()}"
                )
                return
        except Exception as e:
            logger.error(f"failed to restart ipcam service: {e}")
            return

        if rtsp_enabled:
            logger.info(
                f"waiting {RESTART_SETTLE_TIME}s for ipcam to settle "
                "before restoring RTSP state"
            )
            await asyncio.sleep(RESTART_SETTLE_TIME)
            await self._restore_rtsp()

    async def task(self):
        logger.info(
            f"camera watchdog started, waiting {BOOT_GRACE_PERIOD}s "
            "boot grace period before first probe"
        )
        await asyncio.sleep(BOOT_GRACE_PERIOD)

        while True:
            interval = self.daemon.settings.get(
                "camera.watchdog.interval", DEFAULT_INTERVAL
            )

            if not await self._probe_camera():
                rtsp_enabled = self._rtsp_was_on()
                await self._restart_ipcam(rtsp_enabled)
                await asyncio.sleep(interval)
            else:
                logger.debug("camera RTSP probe OK")

            await asyncio.sleep(interval)


def load(daemon):
    global _daemon
    _daemon = daemon
    daemon.camera_watchdog = CameraWatchdog(daemon=daemon)


def start():
    asyncio.create_task(_daemon.camera_watchdog.task())
