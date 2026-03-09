param (
    [string]$Region = "us-central1",
    [string]$ServiceName = "ml-supervisado-api"
)

Write-Host "=============================================" -ForegroundColor Cyan
Write-Host " Despliegue en Google Cloud Run + PostgreSQL" -ForegroundColor Cyan
Write-Host "============================================="

$InstanceConnectionName = Read-Host "1. Ingresa el 'Nombre de conexion de la instancia' de Cloud SQL (ej. tu-proyecto:us-central1:mi-instancia)"
if ([string]::IsNullOrWhiteSpace($InstanceConnectionName)) {
    Write-Host "Error: El nombre de la instancia no puede estar vacio." -ForegroundColor Red
    exit 1
}

$DbName = Read-Host "2. Ingresa el nombre de la base de datos (por defecto 'postgres')"
if ([string]::IsNullOrWhiteSpace($DbName)) { $DbName = "postgres" }

$DbUser = Read-Host "3. Ingresa el usuario de la DB (por defecto 'postgres')"
if ([string]::IsNullOrWhiteSpace($DbUser)) { $DbUser = "postgres" }

$DbPass = Read-Host "4. Ingresa la contraseña de la DB" -AsSecureString
$DbPassStr = [System.Runtime.InteropServices.Marshal]::PtrToStringAuto([System.Runtime.InteropServices.Marshal]::SecureStringToBSTR($DbPass))

Write-Host "`nIniciando el despliegue a Cloud Run..." -ForegroundColor Yellow

# Ejecutar el comando gcloud usando PowerShell
$envVars = "INSTANCE_CONNECTION_NAME=$InstanceConnectionName,DB_USER=$DbUser,DB_PASS=$DbPassStr,DB_NAME=$DbName"

Write-Host "Ejecutando: gcloud run deploy $ServiceName --source . --region $Region --allow-unauthenticated --add-cloudsql-instances $InstanceConnectionName --set-env-vars ..." -ForegroundColor DarkGray

gcloud run deploy $ServiceName `
    --source . `
    --region $Region `
    --platform managed `
    --allow-unauthenticated `
    --add-cloudsql-instances $InstanceConnectionName `
    --set-env-vars $envVars

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n¡Despliegue finalizado estrepitosamente con exito! :)" -ForegroundColor Green
} else {
    Write-Host "`nHubo un error durante el despliegue. Revisa los logs de gcloud arriba." -ForegroundColor Red
}
