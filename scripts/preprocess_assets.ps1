param(
    [int]$FrameFps = 3,
    [int]$FrameWidth = 1280,
    [switch]$Overwrite
)

$ErrorActionPreference = "Stop"

$start = Get-Date
$status = "success"
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$logPath = Join-Path $root "RESULTS_LOG.md"

function Require-File($path, $message) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw $message
    }
}

function Require-Command($name) {
    if (-not (Get-Command $name -ErrorAction SilentlyContinue)) {
        throw "Missing command: $name"
    }
}

try {
    Require-Command "ffmpeg"
    Require-Command "python"

    $videoA = Join-Path $root "assets/A/VID_20260527_155909.mp4"
    $imageC = Join-Path $root "assets/C/IMG_20260527_161109.jpg"
    $framesA = Join-Path $root "data/A_real_object/images"
    $dataC = Join-Path $root "data/C_single_image"

    Require-File $videoA "Missing Object A video: $videoA"
    Require-File $imageC "Missing Object C image: $imageC"

    New-Item -ItemType Directory -Force -Path $framesA | Out-Null
    New-Item -ItemType Directory -Force -Path $dataC | Out-Null

    $existingFrames = @(Get-ChildItem -LiteralPath $framesA -Filter "frame_*.jpg" -File -ErrorAction SilentlyContinue)
    if ($existingFrames.Count -eq 0 -or $Overwrite) {
        if ($Overwrite) {
            Get-ChildItem -LiteralPath $framesA -Filter "frame_*.jpg" -File -ErrorAction SilentlyContinue |
                Remove-Item -Force
        }
        ffmpeg -y -i $videoA -vf "fps=$FrameFps,scale=$FrameWidth`:-2" -q:v 2 (Join-Path $framesA "frame_%04d.jpg")
    } else {
        Write-Host "Skipping A frame extraction; $($existingFrames.Count) frames already exist. Use -Overwrite to regenerate."
    }

    $crops = @(
        @{ Name = "c_crop_1024.png"; Filter = "crop=2400:2400:1300:120,scale=1024:1024" },
        @{ Name = "c_zero123_input_512.png"; Filter = "crop=2400:2400:1300:120,scale=512:512" },
        @{ Name = "c_crop_margin_1024.png"; Filter = "crop=3000:3000:900:0,scale=1024:1024" },
        @{ Name = "c_zero123_input_margin_512.png"; Filter = "crop=3000:3000:900:0,scale=512:512" }
    )

    foreach ($crop in $crops) {
        $outPath = Join-Path $dataC $crop.Name
        if ((Test-Path -LiteralPath $outPath -PathType Leaf) -and -not $Overwrite) {
            Write-Host "Skipping existing C crop: $outPath"
            continue
        }
        ffmpeg -y -i $imageC -vf $crop.Filter $outPath
    }

    python (Join-Path $root "scripts/make_c_rgba_mask.py")
}
catch {
    $status = "failed"
    throw
}
finally {
    $end = Get-Date
    $elapsed = [int]($end - $start).TotalSeconds
    $entry = @(
        "",
        "## $($start.ToString('yyyy-MM-ddTHH:mm:ss')) preprocess_assets",
        "",
        "- status: $status",
        "- elapsed_seconds: $elapsed",
        "- command: powershell -File scripts/preprocess_assets.ps1 -FrameFps $FrameFps -FrameWidth $FrameWidth",
        "- outputs: data/A_real_object/images, data/C_single_image",
        ""
    )
    Add-Content -LiteralPath $logPath -Value $entry -Encoding utf8
}
