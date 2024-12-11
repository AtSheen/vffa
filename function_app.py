import azure.functions as func
import logging


@app.route(route="httpget", methods=["GET"])
def http_get(req: func.HttpRequest) -> func.HttpResponse:

    logging.info(f"Processing GET request. Name:")

    return func.HttpResponse(f"Hello,!")

@app.route(route="httpget2", methods=["GET"])
def http_get(req: func.HttpRequest) -> func.HttpResponse:
    from azure.storage.blob import BlobServiceClient
    logging.info(f"Processing GET request. Name:")

    return func.HttpResponse(f"Hello,!")
