import boto3

# Initialize Bedrock client
bedrock = boto3.client(service_name='bedrock', region_name='us-east-1')

# List available foundation models
response = bedrock.list_foundation_models()

# Display the model IDs and providers
for model in response['modelSummaries']:
    print(f"Model ID: {model['modelId']} | Provider: {model['providerName']} | Model Name: {model['modelName']}")
