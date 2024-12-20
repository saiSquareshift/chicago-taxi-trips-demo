import functions_framework
from google.cloud import dataproc_v1 as dataproc


@functions_framework.http
def hello_http(request):
    """HTTP Cloud Function.
    Args:
        request (flask.Request): The request object.
        <https://flask.palletsprojects.com/en/1.1.x/api/#incoming-request-data>
    Returns:
        The response text, or any set of values that can be turned into a
        Response object using `make_response`
        <https://flask.palletsprojects.com/en/1.1.x/api/#flask.make_response>.
    """

    # Set CORS headers for the preflight request
    if request.method == "OPTIONS":
        # Allows GET requests from any origin with the Content-Type
        # header and caches preflight response for an 3600s
        headers = {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST",
            "Access-Control-Allow-Headers": "Content-Type, Authorization",
            "Access-Control-Max-Age": "3600",
        }

        return ("", 204, headers)

    # Set CORS headers for the main request
    headers = {"Access-Control-Allow-Origin": "*"}
    
    request_json = request.get_json(silent=True)
    request_args = request.args

    if request_json and 'project_id' in request_json:
        project_id = request_json['project_id']
    elif request_args and 'project_id' in request_args:
        project_id = request_args['project_id']
    else:
        project_id = "gke-elastic-394012"

    if request_json and 'region' in request_json:
        region = request_json['region']
    elif request_args and 'region' in request_args:
        region = request_args['region']
    else:
        region = "us-central1"

    if request_json and 'cluster_name' in request_json:
        cluster_name = request_json['cluster_name']
    elif request_args and 'cluster_name' in request_args:
        cluster_name = request_args['cluster_name']
    else:
        cluster_name = "cluster-411a"

    if request_json and 'bucket_name' in request_json:
        bucket_name = request_json['bucket_name']
    elif request_args and 'bucket_name' in request_args:
        bucket_name = request_args['bucket_name']
    else:
        bucket_name = "chicago-taxi-trip-dataset-01"

    if request_json and 'script_path' in request_json:
        script_path = request_json['script_path']
    elif request_args and 'script_path' in request_args:
        script_path = request_args['script_path']
    else:
        script_path = "ETL.py"

    if request_json and 'dataset_id' in request_json:
        dataset_id = request_json['dataset_id']
    elif request_args and 'dataset_id' in request_args:
        dataset_id = request_args['dataset_id']
    else:
        dataset_id = "chicago_taxi_trips"

    
    if request_json and 'output_table_name' in request_json:
        output_table_name = request_json['output_table_name']
    elif request_args and 'output_table_name' in request_args:
        output_table_name = request_args['output_table_name']
    else:
        output_table_name = "sample_taxi_trips"
    
    if request_json and 'sample_pct' in request_json:
        sample_pct = request_json['sample_pct']
    elif request_args and 'sample_pct' in request_args:
        sample_pct = request_args['sample_pct']
    else:
        sample_pct = "0.001"

    
    # Create the job client.
    job_client = dataproc.JobControllerClient(
        client_options={"api_endpoint": f"{region}-dataproc.googleapis.com:443"}
    )

    # Create the job config. 'main_jar_file_uri' can also be a
    # Google Cloud Storage URL.
    job = {
        "placement": {"cluster_name": cluster_name},
        "pyspark_job": {
            "main_python_file_uri": f"gs://{bucket_name}/{script_path}",
            "args": [project_id, dataset_id, output_table_name, sample_pct],
            "file_uris": [f"gs://{bucket_name}/requirements.txt"]
        }
    }


    try:
        operation = job_client.submit_job_as_operation(
            request={"project_id": project_id, "region": region, "job": job}
        )
        return ('Ok', 200, headers)
        response = operation.result()

    except Exception as e:
        response = f"Failed job: {str(e)}"
        return (response, 500, headers)
