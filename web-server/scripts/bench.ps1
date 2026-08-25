param(
    [int]$Requests = 200,
    [int]$Parallel = 10
)

$uri = "http://127.0.0.1:8080/api/health"
$perJob = [int][Math]::Ceiling($Requests / $Parallel)

$sw = [System.Diagnostics.Stopwatch]::StartNew()

$jobs = for ($i = 0; $i -lt $Parallel; $i++) {
    Start-Job -ScriptBlock {
        param($u, $n)
        $ok = 0
        for ($j = 0; $j -lt $n; $j++) {
            try {
                $r = Invoke-WebRequest -Uri $u -UseBasicParsing -TimeoutSec 10
                if ($r.StatusCode -eq 200) { $ok++ }
            } catch { }
        }
        $ok
    } -ArgumentList $uri, $perJob
}

$results = $jobs | Wait-Job | Receive-Job
$completed = ($results | Measure-Object -Sum).Sum
$jobs | Remove-Job
$sw.Stop()

Write-Host ("completed: {0}/{1} in {2:N1}s ({3:N0} req/s)" -f $completed, $Requests, $sw.Elapsed.TotalSeconds, ($completed / $sw.Elapsed.TotalSeconds))
