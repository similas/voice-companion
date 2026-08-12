# Memory, and why the box kept dropping SSH

Two incidents, 2026-08-08. Both looked like "it overloaded and kicked me out". The
second one was diagnosed from `/var/log/kern.log`, which survives a reset — the
systemd journal on this box is **volatile**, so everything in it was lost.

## What actually happened

```
Aug  8 01:25:17 orin kernel: Call trace:
  ...
  __alloc_pages+0xde0/0xe10
  __kmalloc+0x36c/0x3c0
  elf_core_dump+0x3a0/0xd60
  do_coredump+0xeb0/0x15b0
  get_signal+0x524/0x970
Aug  8 01:25:17 orin kernel: Mem-Info:
  ...
  Swap cache stats: add 10140614, delete 10138629
```

Read bottom-up, the chain is:

1. Memory ran out.
2. The kernel entered a swap-thrash spiral — **10,140,614** swap-cache operations
   in a single boot. This is the part that made the machine stop answering SSH,
   *before* anything was killed. The session was gone while the box was still up.
3. A process took a fatal signal, and the kernel began writing a **core dump** of a
   multi-gigabyte process.
4. That allocation failed, on a machine already thrashing. Game over.

Two things made an ordinary shortfall unsurvivable:

**All swap is zram.** `swapon --show` reports six `/dev/zram*` devices. zram is
*compressed RAM*, not disk. Swapping does not return memory to the system, it just
stores the same bytes more densely while burning CPU. Under pressure that is a
spiral, not a relief valve.

**Core dumps pipe to apport.** `/proc/sys/kernel/core_pattern` is
`|/usr/share/apport/apport …`, so a crashing process has its entire address space
streamed into a Python helper. On a 7.6 GB box with a 3 GB model resident, the
machine ran out of memory *while trying to record why it had run out of memory*.

## Why the first fix was not enough

The earlier fix put `MemoryMax` on `llama-server` alone. The agent stayed uncapped,
so a shortfall still became a **global** OOM — and a global OOM lets the kernel pick
any victim. Measured during the incident:

| process | `oom_score_adj` | `oom_score` | |
|---|---|---|---|
| sshd (listener) | -1000 | 0 | protected |
| **sshd (your session)** | **0** | **666** | **a legitimate target** |
| voice agent | 0 | — | |

Capping one process does not prevent a global event. Only containing *everything
that can grow* does.

## The fix

**One slice for the whole stack** — `~/.config/systemd/user/voice.slice`:

```ini
MemoryHigh=4200M      # throttle: stack gets slow before anything dies
MemoryMax=5000M       # hard wall, enforced by the kernel
MemorySwapMax=0       # no zram, so pressure cannot become a thrash spiral
```

(Originally 3800M/4600M; raised 2026-08-12 when the kokoro-tts sidecar —
~850 MB measured RSS — joined the slice.)

`llama-server.service`, `kokoro-tts.service` and the agent (via
`systemd-run --scope` in `run.sh`) all run inside it. A shortfall is now a
contained cgroup event; the kernel never has to look outside the slice, so
sshd is never a candidate.

**Explicit kill ordering**, as a backstop for anything the cgroup misses. Raising
`oom_score_adj` needs no privileges — only lowering it does — so this works as an
ordinary user:

| process | `oom_score_adj` | `oom_score` | dies |
|---|---|---|---|
| llama-server | 1000 | 1384 | first |
| kokoro-tts | 950 | — | second |
| voice agent | 900 | 1290 | third |
| sshd (session) | 0 | 666 | only if all are already gone |

**No core dumps** from our processes: `LimitCORE=0` on the llama-server unit, and
`resource.setrlimit(RLIMIT_CORE, (0,0))` in the agent. A crash we restart from is
fine; a crash that takes the host with it is not.

## Verified, not assumed

An unbounded allocator was run inside the slice while llama-server served traffic:

```
memory.events:  high 39617    max 934    oom 0    oom_kill 0
```

The throttle engaged 39,617 times and the hard cap 934 times. **Nothing was OOM
killed, and the SSH session never noticed.** The allocator was simply slowed to a
crawl until it was terminated.

Cost of that protection: the cgroup reclaimed llama-server's page cache to stay
under the cap, dropping its RSS from 2450 MB to 252 MB. It kept serving, but would
re-fault its weights from disk on the next request. That is the correct trade — the
model gets briefly slow instead of the machine dying.

## Still requires root

These need `sudo` and are not applied yet. The first is the highest value.

```bash
# 1. Stop core dumps from being able to kill the machine (the amplifier above).
sudo systemctl disable --now apport.service
echo 'kernel.core_pattern=core' | sudo tee /etc/sysctl.d/60-no-apport-core.conf
sudo sysctl -p /etc/sysctl.d/60-no-apport-core.conf

# 2. Make the journal survive a reset, so the next incident is diagnosable
#    without archaeology in /var/log/kern.log.
sudo mkdir -p /var/log/journal
sudo systemd-tmpfiles --create --prefix /var/log/journal
sudo systemctl restart systemd-journald

# 3. Protect the per-session sshd directly, not just by comparison.
sudo mkdir -p /etc/systemd/system/ssh.service.d
printf '[Service]\nOOMScoreAdjust=-1000\n' | \
  sudo tee /etc/systemd/system/ssh.service.d/oom.conf
sudo systemctl daemon-reload && sudo systemctl restart ssh

# 4. Optional: make the kernel far less eager to reach for zram at all.
echo 'vm.swappiness=10' | sudo tee /etc/sysctl.d/60-swappiness.conf
sudo sysctl -p /etc/sysctl.d/60-swappiness.conf
```

## Checking it

```bash
systemctl --user show voice.slice -p MemoryCurrent -p MemoryMax
systemd-cgls --user-unit voice.slice          # both processes should be listed
cat /sys/fs/cgroup/user.slice/user-1000.slice/user@1000.service/voice.slice/memory.events
```

If `oom_kill` is non-zero, the stack hit the wall and something inside it died —
which is the designed outcome, and strictly better than losing the login session.
