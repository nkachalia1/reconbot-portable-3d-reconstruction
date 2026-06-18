param(
    [Parameter(Mandatory = $true)]
    [string]$ImagePath,

    [Parameter(Mandatory = $false)]
    [string]$OutputPath = "outputs/colmap"
)

$ErrorActionPreference = "Stop"

New-Item -ItemType Directory -Force -Path $OutputPath | Out-Null
New-Item -ItemType Directory -Force -Path "$OutputPath/sparse" | Out-Null

colmap feature_extractor --database_path "$OutputPath/database.db" --image_path $ImagePath
colmap exhaustive_matcher --database_path "$OutputPath/database.db"
colmap mapper --database_path "$OutputPath/database.db" --image_path $ImagePath --output_path "$OutputPath/sparse"
