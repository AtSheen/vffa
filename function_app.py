import azure.functions as func
import logging
from azure.storage.blob import BlobServiceClient

@app.route(route="httpget", methods=["GET"])
def http_get(req: func.HttpRequest) -> func.HttpResponse:

    logging.info(f"Processing GET request. Name:")

    return func.HttpResponse(f"Hello,!")
