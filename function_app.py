import azure.functions as func
import logging
 
 
@app.route(route="httpget", methods=["GET"])
def http_get(req: func.HttpRequest) -> func.HttpResponse:
 
    logging.info(f"Processing GET request. Name:")
 
    return func.HttpResponse(f"Hello,!")
 
@app.route(route="httpget2", methods=["GET"])
def http_get2(req: func.HttpRequest) -> func.HttpResponse:
    from azure.storage.blob import BlobServiceClient
    logging.info(f"Processing GET request. Name:")
 
    return func.HttpResponse(f"Hello,!")

@app.route(route="GetTaxCodeInfoFile", auth_level=func.AuthLevel.FUNCTION)
def GetTaxCodeInfoFile(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function to read Tax Code Info Excel file from Blob Storage.')

    # Blob storage details
    blob_name = tax_code_info_blob_name

    # Initialize BlobServiceClient
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)

    try:
        # Download the blob content as bytes
        blob_data = blob_client.download_blob().readall()

        # Read the Excel file into a pandas DataFrame
        tax_code_info = pd.read_excel(BytesIO(blob_data))

        # Convert DataFrame to CSV
        csv_data = tax_code_info.to_csv(index=False)

        # Return CSV data
        return func.HttpResponse(csv_data, mimetype="text/csv", status_code=200, headers={
            "Content-Disposition": f"attachment; filename={blob_name.replace('.xlsx', '.csv')}"
        })

    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)
        return func.HttpResponse(f"Error reading blob: {e}", status_code=500)

@app.route(route="GetLegalEntitiesFile", auth_level=func.AuthLevel.FUNCTION)
def GetLegalEntitiesFile(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function to read Legal Entities CSV file from Blob Storage.')

    # Blob storage details
    blob_name = legal_entities_blob_name

    # Initialize BlobServiceClient
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)

    try:
        # Download the blob content as bytes
        blob_data = blob_client.download_blob().readall()

        # Read the Excel file into a pandas DataFrame
        same_legal_entity = pd.read_csv(BytesIO(blob_data))

        # Convert DataFrame to CSV
        csv_data = same_legal_entity.to_csv(index=False)

        # Return CSV data
        return func.HttpResponse(csv_data, mimetype="text/csv", status_code=200, headers={
            "Content-Disposition": f"attachment; filename={blob_name}"
        })

    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)
        return func.HttpResponse(f"Error reading blob: {e}", status_code=500)


@app.route(route="GetCategoricalConfigFile", auth_level=func.AuthLevel.FUNCTION)
def GetCategoricalConfigFile(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    blob_name = category_config_file_blob_name

    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service_client.get_container_client(container_name)
    blob_client = container_client.get_blob_client(blob_name)
    
    try:
        # Download the blob content as bytes
        blob_data = blob_client.download_blob().readall()

        # Return the joblib file as a binary response
        return func.HttpResponse(
            body=blob_data,
            mimetype="application/octet-stream",
            status_code=200,
            headers={
                "Content-Disposition": f"attachment; filename={blob_name}"
            }
        )

    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)
        return func.HttpResponse(f"Error reading blob: {e}", status_code=500)
 

