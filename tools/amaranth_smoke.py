"""Amaranth smoke test: simulate a counter, emit Verilog, run Yosys synth.

Run:  .venv\Scripts\python.exe tools\amaranth_smoke.py
"""
import os
import subprocess
import sys
from pathlib import Path

from amaranth import Module, Signal
from amaranth.lib.wiring import Component, In, Out
from amaranth.sim import Simulator

# Use the bundled amaranth-yosys WASM build directly (sys.executable -m
# amaranth_yosys). "system" would find .venv\Scripts\yosys.bat on PATH, but
# stdin scripts get mangled through cmd's .bat dispatch — the builtin proxy
# is the intended path for amaranth internals.
os.environ.setdefault("AMARANTH_USE_YOSYS", "builtin")
os.environ.setdefault("YOSYS", str(Path(sys.executable).parent / "yosys.bat"))


class Counter(Component):
    count: Out(8)
    ovf: Out(1)

    def elaborate(self, platform):
        m = Module()
        m.d.sync += self.count.eq(self.count + 1)
        with m.If(self.count == 0xFF):
            m.d.sync += self.ovf.eq(1)
        return m


dut = Counter()
sim = Simulator(dut)
sim.add_clock(1e-6)


async def bench(ctx):
    for _ in range(10):
        await ctx.tick()
    value = ctx.get(dut.count)
    print(f"count after 10 ticks: {value}")
    assert value == 10, f"expected 10, got {value}"


sim.add_testbench(bench)
with sim.write_vcd("counter.vcd"):
    sim.run()
print("simulation OK -> counter.vcd written")

from amaranth.back import verilog  # noqa: E402

vlog = verilog.convert(dut, ports=[dut.count, dut.ovf])
Path("counter.v").write_text(vlog)
print("verilog.emit OK -> counter.v written")

# This amaranth-yosys build is intentionally minimal: frontends, proc,
# opt_*, write_*, select/ls/dump/json — no abc/synth/stat. Exercise the
# process->optimize->inspect path it does have.
result = subprocess.run(
    [sys.executable, "-m", "amaranth_yosys", "-Q", "-T",
     "-p", "read_verilog counter.v; proc; opt; ls"],
    capture_output=True, text=True,
)
assert result.returncode == 0, result.stderr + result.stdout
ls_out = result.stdout.strip().splitlines()
print("yosys read/proc/opt/ls OK:")
for line in ls_out[-8:]:
    print("  ", line.rstrip())
print("ALL CHECKS PASSED")
