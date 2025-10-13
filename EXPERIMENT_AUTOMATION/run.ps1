# List of config files to run (excluding docker)
$configs = @(
    "C:\EXPERIMENT_AUTOMATION\configuration\baseline_idle_no_tools.yml",
    "C:\EXPERIMENT_AUTOMATION\configuration\spring_docker_idle.yml",
    "C:\EXPERIMENT_AUTOMATION\configuration\spring_docker_none.yml",
    "C:\EXPERIMENT_AUTOMATION\configuration\spring_docker_tools.yml",
    "C:\EXPERIMENT_AUTOMATION\configuration\spring_docker_joularjx.yml",
    "C:\EXPERIMENT_AUTOMATION\configuration\spring_docker_otjae.yml",
    "C:\EXPERIMENT_AUTOMATION\configuration\spring_docker_kepler.yml",
    "C:\EXPERIMENT_AUTOMATION\configuration\spring_docker_scaphandre.yml"
)

# Path to your Python script
$scriptPath = "C:\EXPERIMENT_AUTOMATION\main.py"

# Loop through configs and execute
foreach ($config in $configs) {
    Write-Host "`nRunning config: $config`n" -ForegroundColor Cyan
    C:\Python313\python.exe $scriptPath --config "$config"
    
    Write-Host "`nWaiting 5 minutes before next run...`n" -ForegroundColor Yellow
    Start-Sleep -Seconds 300
}