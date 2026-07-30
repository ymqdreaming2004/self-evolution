@echo off
cd /d D:\Agent_projects\Mine\self-evolution
"D:\Agent_projects\Mine\self-evolution\.venv\Scripts\python.exe" -m self_evolution_agent.worker 1>>data\worker.out.log 2>>data\worker.err.log
