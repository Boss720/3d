param(
    [ValidateRange(1, 65535)]
    [int]$Port = 8765
)

$ErrorActionPreference = 'Stop'

$exePath = Join-Path $PSScriptRoot 'StereoLab.exe'
if (-not (Test-Path -LiteralPath $exePath -PathType Leaf)) {
    throw "StereoLab.exe was not found beside this launcher: $exePath"
}

$client = [System.Net.Sockets.TcpClient]::new()
try {
    $client.Connect('127.0.0.1', $Port)
    throw "Port $Port is already in use. Close the other app or start StereoLab with another port."
}
catch [System.Net.Sockets.SocketException] {
    # No listener is present; this launcher can safely start StereoLab.
}
finally {
    $client.Dispose()
}

$server = Start-Process -FilePath $exePath -ArgumentList @('--port', "$Port") -WorkingDirectory $PSScriptRoot -PassThru
$url = "http://127.0.0.1:$Port/"
$deadline = [DateTime]::UtcNow.AddSeconds(30)
while ([DateTime]::UtcNow -lt $deadline) {
    if ($server.HasExited) {
        throw "StereoLab stopped before its local page became available. Check the StereoLab console window."
    }
    try {
        $response = Invoke-WebRequest -Uri $url -TimeoutSec 2 -UseBasicParsing
        if ($response.StatusCode -eq 200) {
            Start-Process $url
            exit 0
        }
    }
    catch {
        Start-Sleep -Milliseconds 300
    }
}

throw "StereoLab did not become ready within 30 seconds. The server process may still be running; open $url after checking its console window."
