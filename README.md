# Sims2026 — simulation & hardware lab environments

Self-contained test environments. Everything lives inside this folder —
no system installs required except the base Python 3.11 that created the venv.

## Environments

### 1. JupyterLab + NetworkX  (`Launch-JupyterLab.bat`)

- Python venv in `.venv\` (Python 3.11.9)
- `jupyterlab 4.6.6`, `networkx 3.6.1`, `matplotlib`, `numpy`
- Working dir: `notebooks\` (starter: `01_networkx_getting_started.ipynb`)
- The launcher opens your browser automatically; stop with Ctrl+C in the
  console window.

### 2. Logisim Evolution 5.0.0  (`Launch-Logisim.bat`)

- Portable jpackage build in `logisim\` with bundled Java 21.0.4 runtime —
  no system Java needed.
- Digital logic design/simulation (gates → ALU → simple CPUs).

### 3. Ripes 2.3.1  (`Launch-Ripes.bat`)

- Portable build in `ripes\` (Qt6, `Ripes.exe`).
- RISC-V simulator: write/assemble RV32/64 code, watch real instruction
  execution through pipeline stages, registers, and the datapath view —
  the reference for comparing your own designs.

### 4. Amaranth HDL 0.5.10  (inside `.venv\`)

- `amaranth` + `amaranth-yosys` (WASM Yosys 0.71) + `gtkwave` at
  `tools\gtkwave64\bin\gtkwave.exe`.
- Smoke test: `.venv\Scripts\python.exe tools\amaranth_smoke.py`
  (sim → `counter.vcd`, emit `counter.v`, Yosys read/proc/opt).
- Open `.vcd` waveforms with `Launch-GTKWave.bat <file.vcd>`.

## Shell access

`Lab-Terminal.bat` opens a cmd window with `.venv\Scripts` and GTKWave on
`PATH`, plus `YOSYS` set — `python`, `jupyter-lab`, `yosys`, `gtkwave` all
resolve there.

## Gotchas

- **Yosys CRLF bug (worked around):** the `amaranth-yosys` WASM build rejects
  `\r` in piped scripts, and `amaranth`'s stock runner feeds it `\r\n` on
  Windows. `.venv\Lib\site-packages\sitecustomize.py` patches the toolchain
  to use binary pipes — `verilog.convert` etc. work normally inside this
  venv. If you write `.ys` script files, save them with **LF** endings.
- **Minimal Yosys build:** `read_*/write_*`, `proc`, `opt_*`, `memory_*`,
  `select`/`ls`/`dump`/`json` exist — but **no `synth`, `stat`, `abc`,
  `techmap`, `show`**. For real synthesis/mapping install full Yosys
  (e.g. OSS CAD Suite) later.
- `yosys` on PATH is a `.bat` shim → `python -m amaranth_yosys`.
- Ripes ships its own RISC-V toolchain for assembling programs in-app.
