@echo off
echo CHECKING ASSETS DIR
if exist assets ( echo Assets dir exists ) else ( mkdir assets && echo Created assets dir )

echo CHECKING SOURCE FILE
dir "C:\Users\singh\.gemini\antigravity\brain\5b66c430-11f0-4731-b755-0f2924d916c0" > debug.txt 2>&1
type debug.txt
