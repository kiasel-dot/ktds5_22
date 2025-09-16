@echo off
SETLOCAL ENABLEDELAYEDEXPANSION

rem Set values for your subscription and resource group
set subscription_id=dc6618c1-53d2-4bc8-ab82-68140c3fbde1
set resource_group=rg-kiasel
set location=SwedenCentral

rem Get random numbers to create unique resource names
set unique_id=!random!!random!

echo 1 Creating storage...
call az storage account create --name kiaselstr!unique_id! --subscription !subscription_id! --resource-group !resource_group! --location !location! --sku Standard_LRS --encryption-services blob --default-action Allow --allow-blob-public-access true --output none

echo 2 Retrieving storage key...
for /f "delims=" %%a in ('az storage account keys list --subscription !subscription_id! --resource-group "!resource_group!" --account-name kiaselstr!unique_id! --query "[0].value" -o tsv') do set "AZURE_STORAGE_KEY=%%a"
if "%AZURE_STORAGE_KEY%"=="" ( echo Failed to get storage key. Exiting. & exit /b 1 )

echo 3 Creating container 'msgstat' and uploading files from 'data' folder...
call az storage container create ^
  --account-name kiaselstr!unique_id! ^
  --name msgstat ^
  --public-access blob ^
  --auth-mode key ^
  --account-key "%AZURE_STORAGE_KEY%" ^
  --output none ^
  || ( echo Failed to create container. Exiting. & exit /b 1 )

call az storage blob upload-batch -d msgstat -s data ^
  --account-name kiaselstr!unique_id! ^
  --auth-mode key ^
  --account-key "%AZURE_STORAGE_KEY%" ^
  --only-show-errors ^
  --no-progress ^
  --output none ^
  || ( echo Upload from 'data' folder failed. Exiting. & exit /b 1 )

rem Generate container-level SAS (read/list), expiry: 2099-01-01
for /f "delims=" %%a in ('az storage container generate-sas --account-name kiaselstr!unique_id! --name msgstat --permissions rl --expiry 2099-01-01T00:00Z --auth-mode key --account-key "%AZURE_STORAGE_KEY%" -o tsv') do set "CONTAINER_SAS=%%a"

if not defined CONTAINER_SAS (
  echo Failed to get SAS. Exiting.
  exit /b 1
)

set "SAFE_SAS=%CONTAINER_SAS:%=%%%"

set BLOB_BASE_URL=https://kiaselstr!unique_id!.blob.core.windows.net/msgstat