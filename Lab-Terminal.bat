@echo off
rem Open a cmd shell with the lab environment on PATH:
rem   python, jupyter-lab, yosys (wasm shim), gtkwave
start "Sims2026 Lab Shell" cmd /k "set PATH=%~dp0.venv\Scripts;%~dp0tools\gtkwave64\bin;%PATH% && set YOSYS=%~dp0.venv\Scripts\yosys.bat && cd /d %~dp0"
