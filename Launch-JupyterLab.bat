@echo off
rem Start JupyterLab rooted at the notebooks folder.
cd /d "%~dp0notebooks"
"%~dp0.venv\Scripts\jupyter-lab.exe"
