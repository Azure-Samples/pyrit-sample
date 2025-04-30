//-----------------------------------------------
// main.bicep  – subscription‑scope entry point
//-----------------------------------------------
targetScope = 'subscription'

/*───────────────────────────
   1. Core deployment inputs
───────────────────────────*/
@minLength(1)
@maxLength(64)
param rgName string

@allowed([
  'eastus', 'eastus2', 'westus', 'westus2', 'westcentralus'
  'northeurope', 'francecentral', 'switzerlandnorth', 'switzerlandwest'
  'uksouth', 'australiaeast', 'eastasia', 'southeastasia'
  'centralindia', 'jioindiawest', 'japanwest', 'koreacentral'
])
param location string

@allowed([ 'dev', 'test', 'prod' ])
param environment string = 'dev'

param keyVaultName string = 'livetalk-kv'
param subscriptionId string = subscription().subscriptionId

/* ACR settings */
@minLength(5)
@maxLength(50)
param acrName string = 'livetalkacr'
var   acrLoginServer = '${acrName}.azurecr.io'

/*--------- OpenAI ---------*/
param openAiAccountName       string = 'livetalk-openai'
param openAiCustomDomain      string = 'openai-ctr'
param deployOpenAiAccount     bool   = true
param openAiRestore           bool   = false
param openAiModelDeployment   string = 'livetalk-4o'
@allowed([ 'gpt-4o', 'gpt-4o-mini' ])
param openAiModelName         string = 'gpt-4o'
param openAiModelVersion      string = '2024-11-20'
param openAiCapacity          int    = 80

/*──────────── Resources ────────────*/
resource livetalkRG 'Microsoft.Resources/resourceGroups@2021-04-01' = {
  name: rgName
  location: location
}

resource keyVault 'Microsoft.KeyVault/vaults@2021-06-01-preview' existing = {
  scope: livetalkRG
  name: keyVaultName
}

/*──────── VNet (returns containerAppsSubnetId) ────────*/
module vnetModule './modules/vnet.bicep' = {
  name: 'deployVnet'
  scope: livetalkRG
  params: {
    location: location
    envType: environment
    keyVaultReference: keyVault.name
  }
}

/*──────── Log Analytics ────────*/
module logAnalyticsModule './modules/loga.bicep' = {
  name: 'deployLogAnalytics'
  scope: livetalkRG
  params: {
    workspaceName: 'livetalk-loganalytics'
    location: location
    retentionInDays: 30
    environment: environment
    keyVaultReference: keyVault.name
    subscriptionId: subscriptionId
  }
}

/*──────── ACR ────────*/
module acrModule './modules/acr.bicep' = {
  name: 'deployContainerRegistry'
  scope: livetalkRG
  params: {
    location: location
    environment: environment
    keyVaultReference: keyVault.name
    subscriptionId: subscriptionId
    acrName: acrName
    sku: 'Standard'
  }
}

/*──────── Cosmos DB (unchanged) ────────*/
module cosmosDbModule './modules/cosmos.bicep' = {
  name: 'deployCosmosDb'
  scope: livetalkRG
  params: {
    cosmosDbName: 'livetalk-cosmosdb'
    location: location
    resourceGroupLocation: livetalkRG.location
    environment: environment
    keyVaultReference: keyVault.name
    subscriptionId: subscriptionId
  }
}

/*──────── Container Apps env + first app ────────*/
module containerAppsEnvModule './modules/aca.bicep' = {
  name: 'deployContainerAppsEnv'
  scope: livetalkRG
  params: {
    location: location
    infrastructureSubnetId: vnetModule.outputs.containerAppsSubnetId
    logAnalyticsWorkspaceName: logAnalyticsModule.outputs.workspaceName
    envType: environment
    containerAppName: 'livetalk-api'
    containerAppImage: '${acrLoginServer}/api:latest'
    acrLoginServer: acrLoginServer
  }
}

/*──────── Azure OpenAI ────────*/
module openAiModule './modules/aoai.bicep' = {
  name: 'deployAzureOpenAI'
  scope: livetalkRG
  params: {
    aiServiceAccountName: openAiAccountName
    deployAccount: deployOpenAiAccount
    restore: openAiRestore
    location: location
    customSubDomainName: openAiCustomDomain
    subnetId: vnetModule.outputs.containerAppsSubnetId
    modelDeploymentName: openAiModelDeployment
    modelName: openAiModelName
    modelVersion: openAiModelVersion
    capacity: openAiCapacity
    tags: { project: 'LiveTalk', environment: environment }
  }
}

// =======================================================================
// Module: Azure Bing Search Resource (applying security best practices)
// =======================================================================
module bingSearch './modules/bing.bicep' = {
  name: 'deployBingSearch'
  scope: livetalkRG
  params: {
    location: location
    environment: environment
    keyVaultReference: keyVault.name
    subscriptionId: subscriptionId
  }
}

// =================================================================
// Module: Azure Static Web App (enforces HTTPS and CORS policies)
// =================================================================
module staticWebApp './modules/staticwapp.bicep' = {
  name: 'deployStaticWebApp'
  scope: livetalkRG
  params: {
    location: location
    repositoryUrl: 'https://github.com/Azure-Samples/livetalk.git'
    environment: environment
    keyVaultReference: keyVault.name
  }
}

// ===================================================================================
// Module: Azure Blob Storage Account (enforces HTTPS-only and uses private endpoints)
// ===================================================================================
module storageAccount './modules/blob.bicep' = {
  name: 'deployStorageAccount'
  scope: livetalkRG
  params: {
    location: location
    environment: environment
    keyVaultReference: keyVault.name
    subscriptionId: subscriptionId
  }
}
