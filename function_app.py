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

@app.route(route="GetTaxCodeInfoFile", methods=["GET"])
def GetTaxCodeInfoFile(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function to read Tax Code Info Excel file from Blob Storage.')

    # Blob storage details
    blob_name = "Tax Code Info.xlsx"
    connection_string = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
    container_name = os.getenv('AZURE_STORAGE_CONTAINER_NAME')

 

    # Initialize BlobServiceClient
    from azure.storage.blob import BlobServiceClient
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




 

