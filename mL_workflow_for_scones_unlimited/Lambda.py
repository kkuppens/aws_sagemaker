#  Function 1 - serializeImageData

import json
import boto3
import base64

s3 = boto3.client('s3')


def lambda_handler(event, context):
    """A function to serialize target data from S3"""

    # Get the s3 address from the Step Function event input
    key = event['s3_key']  ## TODO: fill in
    bucket = event['s3_bucket']  ## TODO: fill in

    # Download the data from s3 to /tmp/image.png
    ## TODO: fill in
    s3_input_uri = "/".join([bucket, key])
    input_bucket = s3_input_uri.split('/')[0]
    input_object = '/'.join(s3_input_uri.split('/')[1:])
    file_name = '/tmp/' + 'image.png'  # os.path.basename(key)
    s3.download_file(input_bucket, input_object, file_name)

    # We read the data from a file
    with open("/tmp/image.png", "rb") as f:
        image_data = base64.b64encode(f.read())

    # Pass the data back to the Step Function
    print("Event:", event.keys())
    return {
        'statusCode': 200,
        'body': {
            "image_data": image_data,
            "s3_bucket": bucket,
            "s3_key": key,
            "inferences": []
        }
    }

#  Function 2 - classifyImage

import json
import sagemaker
import base64
from sagemaker.serializers import IdentitySerializer

# Fill this in with the name of your deployed model
ENDPOINT = 'image-classification-2024-03-28-10-47-03-715'  ## TODO: fill in


def lambda_handler(event, context):

    # Decode the image data
    image = base64.b64decode(event["image_data"])   ## TODO: fill in

    # Instantiate a Predictor
    predictor = sagemaker.predictor.Predictor(ENDPOINT)  ## TODO: fill in

    # For this model the IdentitySerializer needs to be "image/png"
    predictor.serializer = IdentitySerializer("image/png")

    # Make a prediction:
    inferences = predictor.predict(image)  ## TODO: fill in

    # We return the data back to the Step Function
    event["inferences"] = inferences.decode('utf-8')
    return {
        'statusCode': 200,
        'body': json.dumps(event)
    }

#  Function 3 - lowConfidenceFilter

import json
import ast

THRESHOLD = .85

def lambda_handler(event, context):

    # Grab the inferences from the event
    inferences = event["inferences"]  ## TODO: fill in
    inferences = ast.literal_eval(inferences)
    # Check if any values in our inferences are above THRESHOLD
    meets_threshold = any(pred > THRESHOLD for pred in inferences) ## TODO: fill in

    # If our threshold is met, pass our data back out of the
    # Step Function, else, end the Step Function with an error
    if meets_threshold:
        pass
    else:
        raise("THRESHOLD_CONFIDENCE_NOT_MET")

    return {
        'statusCode': 200,
        'body': json.dumps(event)
    }
