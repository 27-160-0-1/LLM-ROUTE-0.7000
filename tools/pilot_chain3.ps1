# Supervisor: keep llama-server alive; run the --instruct pilot (resumable) until its log says DONE.
$root = "C:\Users\012\SKT LLM\official-router"
$py = "C:\Users\012\SKT LLM\.venv\Scripts\python.exe"
$llamaDir = "C:\Users\012\SKT LLM\local-llm\llama"
$log = "$root\reports\pilot_chain3.log"
Set-Location $root
function Log($m) { "$(Get-Date -Format 'HH:mm:ss') $m" | Out-File $log -Append -Encoding utf8 }
function Healthy() { try { $r = Invoke-WebRequest -Uri http://127.0.0.1:8080/health -UseBasicParsing -TimeoutSec 5; return ($r.StatusCode -eq 200) } catch { return $false } }
function EnsureServer() {
    if (Healthy) { return }
    Get-Process llama-server -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
    Start-Sleep 3
    Log "starting llama-server"
    Start-Process -FilePath "$llamaDir\llama-server.exe" -ArgumentList '-m ..\A.X-3.1-Light.Q4_K_M.gguf -ngl 18 -c 4096 --parallel 2 --port 8080 --host 127.0.0.1 -t 6' -WorkingDirectory $llamaDir -WindowStyle Hidden -RedirectStandardOutput "$root\reports\llama_server.out" -RedirectStandardError "$root\reports\llama_server.err" | Out-Null
    for ($i = 0; $i -lt 60; $i++) { if (Healthy) { Log "server healthy"; return }; Start-Sleep 10 }
    Log "server failed to come up"
}
# background health keeper
$keeper = Start-Job -ScriptBlock {
    param($llamaDir, $root)
    function Healthy() { try { $r = Invoke-WebRequest -Uri http://127.0.0.1:8080/health -UseBasicParsing -TimeoutSec 5; return ($r.StatusCode -eq 200) } catch { return $false } }
    while ($true) {
        Start-Sleep 30
        if (-not (Healthy)) {
            $alive = Get-Process llama-server -ErrorAction SilentlyContinue
            if (-not $alive) {
                "$(Get-Date -Format 'HH:mm:ss') keeper: server dead, restarting" | Out-File "$root\reports\pilot_chain3.log" -Append -Encoding utf8
                Start-Process -FilePath "$llamaDir\llama-server.exe" -ArgumentList '-m ..\A.X-3.1-Light.Q4_K_M.gguf -ngl 18 -c 4096 --parallel 2 --port 8080 --host 127.0.0.1 -t 6' -WorkingDirectory $llamaDir -WindowStyle Hidden -RedirectStandardOutput "$root\reports\llama_server.out" -RedirectStandardError "$root\reports\llama_server.err" | Out-Null
                Start-Sleep 90
            }
        }
    }
} -ArgumentList $llamaDir, $root
EnsureServer
for ($round = 0; $round -lt 6; $round++) {
    Log "pilot round $round"
    & $py tools\pilot_local_light.py --per-family 15 --temps 0.7 --n 2 --max-tokens 1024 --instruct --out reports\pilot_light_instr.json 2>&1 | Out-File "reports\pilot_light_instr_r$round.log" -Encoding utf8
    if (Select-String -Path "reports\pilot_light_instr_r$round.log" -Pattern "\[pilot\] DONE" -Quiet) { Log "pilot DONE"; break }
    EnsureServer
}
Stop-Job $keeper -ErrorAction SilentlyContinue
Log "chain3 finished"
