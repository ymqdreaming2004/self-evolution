@echo off
cd /d D:\Agent_projects\Mine\self-evolution
"D:\Agent_projects\Mine\self-evolution\tools\cloudflared.exe" tunnel --url http://127.0.0.1:8000 1>>data\tunnel.out.log 2>>data\tunnel.err.log
