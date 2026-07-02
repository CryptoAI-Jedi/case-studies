# Turning 4 Ex-Mining AMD GPUs into a Local LLM Inference Server (RDNA1 / Vulkan)

A build log: getting four Sapphire Pulse RX 5600 XT cards, retired from Ethereum mining, to serve a real LLM over an OpenAI-compatible API, with no CUDA and no ROCm.

## TL;DR

I had four ex-cryptomining RX 5600 XT cards (6 GB, Navi 10 / RDNA1) and wanted a local LLM inference box. Here is the counterintuitive part: the right answer was to run one of them. The four cards sit in x1 Gen1 mining risers, and cross-card decode over those risers is so slow that a single card pinned correctly beats four cards left to their own routing by roughly 180x.

So the keeper insight up front: x1 Gen1 risers make cross-card decode catastrophic, which means pinning to one card is the fix, not a workaround. If your old mining rig has the same risers, expect to land in the same place.

Around that, three traps cost me a day:

1. The 180x throughput collapse on the API server, caused by the cross-card routing above and fixed with a single environment variable.
2. A headless GPU silently serving from system RAM instead of VRAM.
3. An out-of-memory kernel panic that had nothing to do with VRAM.

End result: one 6 GB card runs a 3B coder model at roughly 40 tokens/sec generation, VRAM-resident, behind a reboot-survivable OpenAI-compatible endpoint. Here is the whole thing, traps included.

## The hardware

| Part | What I used |
| ---- | ---- |
| GPUs | 4x Sapphire Pulse RX 5600 XT 6 GB (Navi 10 / RDNA1 / gfx1010) |
| Board / CPU | Old mining motherboard plus a Haswell-era Xeon with integrated graphics |
| Risers | Mining-era PCIe x1 Gen1 USB risers |
| PSU / RAM / disk | Single 1000 W PSU, 8 GB DDR3, 1 TB external SSD over USB 3.x |
| OS | Ubuntu 22.04 LTS Server |
| Stack | `amdgpu` plus Mesa RADV Vulkan, llama.cpp Vulkan backend |

Two things about this hardware set the tone for everything below.

6 GB VRAM per card is the real ceiling, because it decides which models fit.

The x1 Gen1 risers are the defining constraint, and they are why this build ends at one card instead of four. They are fine for single-card inference, but they punish anything that moves data between cards or loads weights repeatedly. This matters more than you would think, and it is the thread that runs through Step 4 and the takeaways.

## Step 1. Driver path: RADV Vulkan, not ROCm

RDNA1 (gfx1010) was never officially supported by ROCm and never will be. Don't fight it. Do not install AMDGPU-PRO or ROCm. Ubuntu's in-box `amdgpu` kernel driver plus Mesa RADV drives these cards over Vulkan out of the box.

```bash
sudo apt install -y mesa-vulkan-drivers vulkan-tools
vulkaninfo --summary 2>/dev/null | grep -c RADV     # should print your card count
lspci | grep -iE 'vga|3d|display'                    # one AMD/ATI line per card
```

A healthy AMD card shows `vendorID 0x1002` and a `RADV NAVI10` device name. If you see `llvmpipe` in your device list, that is the CPU software renderer (a fallback), not your GPU. Ignore it.

## Step 2. Building llama.cpp with Vulkan on Ubuntu 22.04 (dependency traps)

The Vulkan backend has two build dependencies that Ubuntu 22.04 (Jammy) handles awkwardly.

Trap 1: `SPIRV-Headers` not found. The Vulkan shader generator needs it and it isn't pulled in automatically:

```bash
sudo apt install -y spirv-headers
```

Trap 2: `glslc` doesn't exist in Jammy's repos. The shader-compile step fails with `Vulkan_GLSLC_EXECUTABLE-NOTFOUND`. `glslc` (from Google's shaderc) was packaged for Ubuntu after 22.04 shipped, so it's in 24.04 but not 22.04. If you're still running Ubuntu 22.04, get it from LunarG's official Vulkan SDK repo:

```bash
wget -qO- https://packages.lunarg.com/lunarg-signing-key-pub.asc | sudo tee /etc/apt/trusted.gpg.d/lunarg.asc
sudo wget -qO /etc/apt/sources.list.d/lunarg-vulkan-jammy.list http://packages.lunarg.com/vulkan/lunarg-vulkan-jammy.list
sudo apt update && sudo apt install -y vulkan-sdk
glslc --version
```

Then build. Do it in `tmux`. The Vulkan shader file is one huge translation unit that parks the build at a single percentage for several minutes and looks frozen. It isn't.

```bash
sudo apt install -y build-essential cmake git libcurl4-openssl-dev glslang-tools libvulkan-dev
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
cmake -B build -DGGML_VULKAN=ON -DLLAMA_CURL=ON
cmake --build build --config Release -j$(nproc)
```

If a configure step fails, `rm -rf build` and reconfigure. A cached NOTFOUND won't re-detect a tool you just installed. The build is incremental, so if it dies mid-way, just re-run the same `cmake --build` and it resumes.

## Step 3. The RAM ceiling (an OOM panic that isn't about VRAM)

First real gotcha. Loading a roughly 4.7 GB 7B model kernel-panicked the box. `Out of memory: Killed process ...` cascaded into `Kernel panic - not syncing: System is deadlocked on memory`.

The cause isn't VRAM. llama.cpp memory-maps the GGUF by default, and mmap is demand-paged, not an eager full-RAM load. The OOM shows up once the page cache thrashes, which happens as soon as the mapped file plus the working set exceeds available RAM. On an 8 GB box with a 4.7 GB model, you are underwater before the OS overhead is even counted. System RAM is a hard ceiling independent of your VRAM.

Two fixes:

```bash
# 1) a swapfile so memory pressure pages out instead of panicking
sudo fallocate -l 8G /swapfile && sudo chmod 600 /swapfile
sudo mkswap /swapfile && sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

```bash
# 2) --no-mmap: load weights straight to VRAM without mapping the whole file into RAM
./build/bin/llama-cli -m ~/models/<model>.gguf -ngl 99 --no-mmap -p "hello" -n 64
```

`--no-mmap` also raised throughput on the RAM-constrained box, because weights go direct to VRAM instead of faulting through a starved page cache. Small models on adequate RAM don't need it. Large models on tight RAM do.

## Step 4. The big one: a 180x slowdown on the API server

This is the trap that turned a four-card build into a one-card build, so here's the full diagnostic path, which is a useful pattern.

`llama.cpp` ships `llama-server`, which exposes an OpenAI-compatible `/v1/chat/completions` API. I got it running, hit it, and got a correct answer at 0.22 tokens/sec. Not just slow. Broken.

The trap here is that a correct answer at a broken speed is easy to shrug off and move on. Don't. I chased several wrong theories (process contention, model too big, multi-GPU split) before the decisive test.

Run `llama-bench`. It isolates the GPU path from the server config.

```bash
./build/bin/llama-bench -m ~/models/<model>.gguf -ngl 36
```

It came back at 34 t/s on Vulkan. So the card, driver, and llama.cpp were fine. The problem was specific to how the server ran. That single test turned "everything is broken" into "the server does one thing the benchmark doesn't."

That one thing: the server initializes all four GPUs at startup and routes the decode step across the other cards over the x1 Gen1 risers. Cross-card traffic over x1 Gen1 is catastrophically slow, hence the flat 0.22 t/s. This is the same riser constraint from the hardware section, now showing up as a 180x slowdown instead of a bandwidth footnote.

The fix is to make the other three cards invisible to the process at the driver level. llama.cpp's own device flags (`--split-mode none`, `GGML_VK_VISIBLE_DEVICES`) weren't enough. The latter even reordered devices and selected the software renderer by mistake. What worked was the Mesa/RADV device selector, pinning to one card by its PCI vendor:device ID:

```bash
MESA_VK_DEVICE_SELECT=1002:731f MESA_VK_DEVICE_SELECT_FORCE_DEFAULT_DEVICE=1 \
./build/bin/llama-server -m ~/models/<model>.gguf \
  -ngl 36 --split-mode none --parallel 1 --ctx-size 4096 \
  --host 0.0.0.0 --port 8080
```

0.22 t/s up to 40-plus t/s. The startup log now reports 1 Vulkan device instead of 4, and decode has nowhere to wander. (`1002:731f` is the RX 5600 XT's ID. Find yours with `lspci -nn | grep VGA`.)

A couple more server flags that matter on small VRAM. `--parallel 1`, because the server otherwise pre-reserves a KV cache per slot (e.g. 4 slots by 14 K tokens), which won't fit 6 GB and spills to CPU. And `--ctx-size` sized to what's left after the weights. Pin `-ngl` to the model's actual layer count from the load log. Don't rely on `99`, which can trip the auto-fit logic into aborting.

## Step 5. The headless GPU that served from RAM

After a reboot the server was back to 0.24 t/s, and this time the config was provably identical. `radeontop` told the story:

```
12M / 6109M VRAM        <- model is NOT in VRAM
2197M / 5872M GTT       <- model is in GTT (system RAM over the bus)
Memory Clock  11.43%    <- clock floored
Graphics pipe 100.00%   <- thrashing against RAM-resident weights
```

Cause: on a headless GPU, amdgpu's runtime power management autosuspends the idle card and floors the memory clock, then serves inference out of GTT instead of VRAM. It still looks like it's "on the GPU," but it's reading weights across the bus every step.

Fix: disable runtime PM persistently via the kernel command line.

```bash
# add amdgpu.runpm=0 to GRUB_CMDLINE_LINUX_DEFAULT in /etc/default/grub
sudo update-grub && sudo reboot
# verify:
cat /proc/cmdline | grep -o 'amdgpu.runpm=0'
cat /sys/bus/pci/devices/<PCI_ADDR>/power/control    # should read "on"
```

Back to VRAM-resident, and it survives reboots.

## The health check I now use for everything

The acceptance test is a single curl, and the tokens/sec number is the pass/fail signal:

```bash
curl http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"m","messages":[{"role":"user","content":"hello"}]}'
```

Read `timings.predicted_per_second`. A healthy VRAM-resident baseline on this card is roughly 45 to 55 t/s. The roughly-0.2 t/s failure state is unmistakable against it. I re-run this after any change (model swap, driver update, kernel bump), because every failure mode above shows up as the same 0.2 t/s. Cross-check with `radeontop`: VRAM loaded means healthy, GTT loaded means something is spilling.

## Making it stick: systemd

Wrapped in a systemd unit so it survives logout and restarts on boot. The one non-obvious requirement: the `MESA_VK_DEVICE_SELECT` env vars must be baked into the unit. Omit them and it silently regresses to the multi-card 0.2 t/s behavior.

```ini
[Service]
User=<user>
SupplementaryGroups=render video          # /dev/dri access regardless of login session
Environment=MESA_VK_DEVICE_SELECT=1002:731f
Environment=MESA_VK_DEVICE_SELECT_FORCE_DEFAULT_DEVICE=1
ExecStart=/path/to/llama-server -m /path/to/model.gguf \
  -ngl 36 --split-mode none --parallel 1 --ctx-size 4096 --host 0.0.0.0 --port 8080
Restart=on-failure
```

## Results and takeaways

The defining constraint is the x1 Gen1 risers. They make cross-card decode catastrophic, so pinning to a single card is the fix, not a workaround. If you are replicating this on a retired mining rig, expect to land at one working card regardless of how many are in the box.

On that one card, the result is real. A single ex-mining RX 5600 XT (6 GB) serves a 3B coder model at roughly 45 to 55 t/s generation, VRAM-resident, behind an OpenAI-compatible API. No CUDA, no ROCm.

6 GB comfortably runs 3B to 4B models at Q4. A 7B fits but is tight on RAM as much as VRAM. The other three cards are still in the box, still wired up, and still unused for inference. They could serve other models on other ports, but only one at a time per card, and never as a pool.

Three lessons worth stealing:

1. A correct answer at a broken speed is still broken. Check tokens/sec and `radeontop`, don't just eyeball the output.
2. When something's slow, isolate the layer. `llama-bench` proving the GPU path was healthy is what turned a vague slowdown into a one-line fix.
3. On old or headless hardware, the driver's power management will lie to you. `amdgpu.runpm=0` and confirming VRAM-vs-GTT residency saved me from blaming the card.

Old mining GPUs still have a second life as local inference boxes. You just have to go around ROCm, not through it, and you have to respect what the risers will and won't carry.

---

*Build notes from a personal homelab project. Card and driver specifics are for RDNA1 (RX 5600 XT); other architectures will differ.*

