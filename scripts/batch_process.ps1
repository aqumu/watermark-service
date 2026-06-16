param(
    [Parameter(Mandatory)]
    [string]$Folder,
    [string]$Url = "http://localhost:8000/api/v1/process/batch",
    [string]$OutputZip = "results.zip"
)

$files = Get-ChildItem -LiteralPath $Folder -Include "*.jpg", "*.jpeg", "*.png" -File
if (-not $files) {
    Write-Error "No image files found in $Folder"
    exit 1
}

Write-Host "Found $($files.Count) images. Uploading..."

$uri = [System.Uri]::new($Url)
$webReq = [System.Net.WebRequest]::Create($uri)
$webReq.Method = "POST"
$boundary = "----Boundary$(Get-Random)"
$webReq.ContentType = "multipart/form-data; boundary=$boundary"
$webReq.Timeout = [System.Threading.Timeout]::Infinite

$stream = $webReq.GetRequestStream()
$enc = [System.Text.Encoding]::ASCII
$crlf = "`r`n"

foreach ($file in $files) {
    $data = [System.IO.File]::ReadAllBytes($file.FullName)
    $header = "--$boundary${crlf}Content-Disposition: form-data; name=`"images`"; filename=`"$($file.Name)`"${crlf}Content-Type: application/octet-stream${crlf}${crlf}"
    $headerBytes = $enc.GetBytes($header)
    $stream.Write($headerBytes, 0, $headerBytes.Length)
    $stream.Write($data, 0, $data.Length)
    $stream.Write($enc.GetBytes($crlf), 0, $crlf.Length)
}

$footer = $enc.GetBytes("--$boundary--${crlf}")
$stream.Write($footer, 0, $footer.Length)
$stream.Close()

$response = $webReq.GetResponse()
$respStream = $response.GetResponseStream()
$reader = New-Object System.IO.BinaryReader $respStream
[System.IO.File]::WriteAllBytes((Resolve-Path .).Path + "\$OutputZip", $reader.ReadBytes($response.ContentLength))
$reader.Close()
$respStream.Close()
$response.Close()

Write-Host "Done — saved to $OutputZip"
