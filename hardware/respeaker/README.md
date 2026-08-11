# reSpeaker XVF3800 — firmware note

The "with XIAO ESP32-S3" bundle ships in **I2S mode**: the XVF3800 talks only to
the on-board ESP32 and **does not enumerate over USB at all**. The XIAO's own
USB-C port shows up as an Espressif serial device, which is not the microphone.

To use it as a USB mic (what this project does):

1. `sudo apt install dfu-util`
2. Hold the **Mute** button while plugging power into the XMOS-side USB-C port —
   red blinking LED = safe mode, and it enumerates as `2886:001a`.
3. `sudo dfu-util -R -e -a 1 -D respeaker_xvf3800_usb_dfu_firmware_v2.0.10.bin`

Firmware binaries are not committed. Download from:
https://github.com/respeaker/reSpeaker_XVF3800_USB_4MIC_ARRAY/tree/master/xmos_firmwares/usb
(v2.0.10 standard build: 2-channel processed output at 16 kHz — exactly what
whisper and Silero want. The 6-channel build is raw per-mic audio and puts the
array processing back on the CPU.)

After flashing it is a normal USB audio device; `pactl set-default-source` to it
and the agent follows (config uses the PipeWire default on purpose).
