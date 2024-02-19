import boto3

sagemaker = boto3.client('sagemaker')

model_name = 'xgbmclassification'
model_artifacts = 's3://classgermandataset22/model-dir/model.tar.gz'
execution_role_arn = 'arn:aws:iam::905418343565:role/SagemakerRole'
image_uri = '492215442770.dkr.ecr.eu-central-1.amazonaws.com/sagemaker-xgboost:1.5-1'  # Retrieved using the SDK


try:
    # Your Boto3 call
    response = sagemaker.create_model(
    ModelName=model_name,
    PrimaryContainer={
        'Image': image_uri,
        'ModelDataUrl': model_artifacts,
    },
    ExecutionRoleArn=execution_role_arn,
)
    print(response)
except Exception as e:
    print(e)

print("Model ARN:", response['ModelArn'])
