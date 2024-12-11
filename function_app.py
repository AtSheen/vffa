import azure.functions as func
import logging
import pandas as pd
import numpy as np
from io import BytesIO
import json
import joblib
import os
import cachetools
import tempfile
from io import StringIO
from typing import Dict, List, Union, Tuple
import sys



app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)



class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)
    
def load_pickle_from_blob(blob_name):
    
    data = {}
    try:
        if blob_name in cache:
            logging.info(f'Returning cached data for {blob_name}')
            return cache[blob_name]

        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        container_client = blob_service_client.get_container_client(container_name)
        blob_client = container_client.get_blob_client(blob_name)
        logging.info('Loading joblib file from blob storage')
        
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            download_stream = blob_client.download_blob()
            temp_file.write(download_stream.readall())
            temp_file_path = temp_file.name

        data = joblib.load(temp_file_path)
        os.remove(temp_file_path)
        cache[blob_name] = data
    except Exception as e:
        logging.error(f"Error: {str(e)}")
    return data

def read_csv_from_blob(container_name, blob_name):
    # Get a reference to the BlobClient
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
    
    # Download the blob's content as a string
    downloaded_blob = blob_client.download_blob().readall()
    
    # Convert the string content to a DataFrame
    df = pd.read_csv(StringIO(downloaded_blob.decode('utf-8')))
    
    return df

def upload_to_blob(file_name, data):
    # Get a reference to the BlobClient
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=file_name)
    
    # Upload the CSV data to the Blob
    blob_client.upload_blob(data, overwrite=True)

def get_data_frame_for_prediction(query, category_config):
    data = {
        'Company Code': query['company_code'],
        'Vendor Number': query['vendor_number'],
        'AP/AR/FI': query['ap_ar'],
        'VAT Rate': query['vat_rate'],
        'Reverse Charge': query['is_reverse_charge'],
        'EU/NON EU': query['eu_noneu'],
        'Goods/Services': query['goods_services'],
        'Domestic/ Foreign': query['domestic_foreign'],
        'Legal Entity': query['legal_entity']
    }

    data = pd.DataFrame([data])
    data = data[category_config['feature_names']]
    for col in data.columns:
        data[col] = pd.Categorical(data[col], categories=[str(i) for i in category_config[col]])
    return data

def calculate_prediction(model, data_for_model) -> Tuple[str, float]:
    """
    Calculates the ML prediction and its probability.

    Args:
        model: Trained machine learning model with predict and predict_proba methods.
        data_for_model: Input data for prediction.

    Returns:
        Tuple[str, float]: Predicted tax code and its probability.
    """
    model_classes = model.classes_.tolist()
    ml_prediction = model.predict(data_for_model)[0]
    predict_probas = model.predict_proba(data_for_model)[0]
    ml_prediction_proba = round((predict_probas.max() * 100), 3)
    return ml_prediction, ml_prediction_proba

def create_proba_dataframe(model_classes, predict_probas) -> pd.DataFrame:
    """
    Creates a DataFrame of tax codes and their prediction probabilities.

    Args:
        model_classes: List of tax code classes from the model.
        predict_probas: Predicted probabilities for each tax code class.

    Returns:
        pd.DataFrame: DataFrame with 'Tax Code' and 'prob' columns.
    """
    ml_prediction_proba_list = list(zip(model_classes, predict_probas * 100))
    ml_prediction_proba_df = pd.DataFrame(ml_prediction_proba_list, columns=['Tax Code', 'prob'])
    ml_prediction_proba_df = ml_prediction_proba_df.sort_values(by='prob', ascending=False).reset_index(drop=True)
    ml_prediction_proba_df['prob'] = ml_prediction_proba_df['prob'].astype(float).round(3)
    ml_prediction_proba_df = ml_prediction_proba_df[ml_prediction_proba_df['prob'] > 0]
    return ml_prediction_proba_df

def track_predictions(model, model_classes, filtered_tax_codes, data_for_model) -> pd.DataFrame:
    """
    Tracks predictions for filtered tax codes not in model classes.

    Args:
        model: Trained machine learning model with estimators_ attribute.
        model_classes: List of tax code classes from the model.
        filtered_tax_codes: List of filtered tax codes to consider.
        data_for_model: Input data for prediction.

    Returns:
        pd.DataFrame: DataFrame with tracked tax codes and their probabilities.
    """
    prediction_tracker = []
    for tax_code in filtered_tax_codes:
        if tax_code not in model_classes:
            tax_code = 'sparse'
        index = model_classes.index(tax_code)
        prediction_proba = round(model.estimators_[index].predict_proba(data_for_model)[0][1] * 100, 3)
        prediction_tracker.append({'Tax Code': tax_code, 'prob': prediction_proba})
    return pd.DataFrame(prediction_tracker)

def get_ml_prediction(model, data_for_model, filtered_tax_codes: List[str]) -> Dict[str, Union[str, float, List[Dict[str, Union[str, float]]]]]:
    """
    Predicts tax codes using a machine learning model and returns predictions with probabilities.

    Args:
        model: Trained machine learning model with predict and predict_proba methods.
        data_for_model: Input data for prediction.
        filtered_tax_codes: List of filtered tax codes to consider in predictions.

    Returns:
        dict: Dictionary containing 'ml_prediction' (predicted tax code),
              'ml_prediction_proba' (probability of prediction), and
              'ml_prediction_proba_df' (DataFrame of tax codes and their probabilities).
    """
    ml_prediction, ml_prediction_proba = calculate_prediction(model, data_for_model)
    model_classes = model.classes_.tolist()
    ml_prediction_proba_df = create_proba_dataframe(model_classes, model.predict_proba(data_for_model)[0])

    prediction_tracker_df = track_predictions(model, model_classes, filtered_tax_codes, data_for_model)
    if prediction_tracker_df.shape[0] > 1:
        prediction_tracker_df = prediction_tracker_df.sort_values(by='prob', ascending=False).reset_index(drop=True)
        ml_prediction = prediction_tracker_df.loc[0, 'Tax Code']
        ml_prediction_proba = prediction_tracker_df.loc[0, 'prob']
        ml_prediction_proba_df = prediction_tracker_df

    output = {
        'ml_prediction': ml_prediction,
        'ml_prediction_proba': ml_prediction_proba,
        'ml_prediction_proba_df': ml_prediction_proba_df.to_dict(orient='records')
    }
    return output


def log_params_and_output_to_blob(query, filtered_tax_codes, output):
    try:
        df = read_csv_from_blob(container_name, ml_invocations_filename)
    except Exception as e:
        logging.error(f"Error reading CSV file: {e}")
        # If the file doesn't exist, create a new DataFrame
        df = pd.DataFrame(columns=['company_code', 'vendor_number', 'vat_rate', 'ap_ar',
                                   'is_reverse_charge', 'legal_entity', 'goods_services',
                                   'eu_noneu', 'domestic_foreign', 'filtered_tax_codes',
                                   'ml_prediction', 'ml_prediction_proba', 'ml_prediction_proba_df'
                                   ])
    # Append new data to the DataFrame
    new_row = {
        'company_code': query['company_code'],
        'vendor_number': query['vendor_number'],
        'vat_rate': query['vat_rate'],
        'ap_ar': query['ap_ar'],
        'is_reverse_charge': query['is_reverse_charge'],
        'legal_entity': query['legal_entity'],
        'goods_services': query['goods_services'],
        'eu_noneu': query['eu_noneu'],
        'domestic_foreign': query['domestic_foreign'],
        'filtered_tax_codes': filtered_tax_codes,
        'ml_prediction': output['ml_prediction'],
        'ml_prediction_proba': output['ml_prediction_proba'],
        'ml_prediction_proba_df': output['ml_prediction_proba_df']

        }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    # Convert the DataFrame back to CSV format
    csv_data = StringIO()
    df.to_csv(csv_data, index=False)
    
    # Upload the updated CSV file back to Blob Storage
    upload_to_blob(ml_invocations_filename, csv_data.getvalue())
    
    logging.info(f"File {ml_invocations_filename} updated and uploaded to Blob Storage successfully.")


@app.route(route="GetTaxCode")
def GetTaxCode(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processing Tax Code.')

    try:
        # Parse the JSON body of the request
        body = req.get_json()
        query = body['query']
        filtered_tax_codes = body['filtered_tax_codes']
        logging.info(f"Request body: {body}")

        # Load the necessary configuration and model from Azure Blob storage
        category_config = load_pickle_from_blob(blob_name=category_config_file_blob_name)
        model = load_pickle_from_blob(blob_name=model_pickle_file)

        # Prepare the data for the model prediction
        data_for_model = get_data_frame_for_prediction(query=query, category_config=category_config)
        
        # Get the model prediction
        output = get_ml_prediction(model, data_for_model, filtered_tax_codes)

        log_params_and_output_to_blob(query, filtered_tax_codes, output)

        logging.info(f'Response: {json.dumps(output, cls=NumpyEncoder)}')
        # Return the prediction result as a JSON response
        return func.HttpResponse(
            json.dumps(output, cls=NumpyEncoder),
            status_code=200,
            mimetype="application/json"
        )
    except Exception as e:
        # Log the error and return an internal server error response
        logging.error(f"Error: {str(e)}", exc_info=True)
        return func.HttpResponse(
            f"Internal Server Error: {str(e)}",
            status_code=500
        )

@app.route(route="GetAttentionListFile", auth_level=func.AuthLevel.FUNCTION)
def GetAttentionListFile(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function to read Attention List Excel file from Blob Storage.')

    # Blob storage details
    blob_name = attention_file_blob_name

    # Initialize BlobServiceClient
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)

    try:
        # Download the blob content as bytes
        blob_data = blob_client.download_blob().readall()

        # Read the Excel file into a pandas DataFrame
        attention_list_df = pd.read_excel(BytesIO(blob_data), sheet_name='Attention list')
        attention_list_df = attention_list_df[['Company code', 'Name ', 'Tax Code', 'Vendor', 'Comment']].dropna(subset=['Vendor'])

        # Convert DataFrame to CSV
        csv_data = attention_list_df.to_csv(index=False)

        # Return CSV data
        return func.HttpResponse(csv_data, mimetype="text/csv", status_code=200, headers={
            "Content-Disposition": f"attachment; filename={blob_name.replace('.xlsx', '.csv')}"
        })

    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)
        return func.HttpResponse(f"Error reading blob: {e}", status_code=500)

@app.route(route="GetIPVatFile", auth_level=func.AuthLevel.FUNCTION)
def GetIPVatFile(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function to read IP Vat Issues List CSV file from Blob Storage.')

    # Blob storage details
    blob_name = ip_vat_blob_name

    # Initialize BlobServiceClient
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)

    try:
        # Download the blob content as bytes
        blob_data = blob_client.download_blob().readall()

        # Read the Excel file into a pandas DataFrame
        ip_vat_df = pd.read_csv(BytesIO(blob_data))

        # Convert DataFrame to CSV
        csv_data = ip_vat_df.to_csv(index=False)

        # Return CSV data
        return func.HttpResponse(csv_data, mimetype="text/csv", status_code=200, headers={
            "Content-Disposition": f"attachment; filename={blob_name}"
        })

    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)
        return func.HttpResponse(f"Error reading blob: {e}", status_code=500)

@app.route(route="GetHistoricalMetaFile", auth_level=func.AuthLevel.FUNCTION)
def GetHistoricalMetaFile(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function to read Historical Metadata CSV file from Blob Storage.')

    # Blob storage details
    blob_name = historical_metadata_blob_name

    # Initialize BlobServiceClient
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)

    try:
        # Download the blob content as bytes
        blob_data = blob_client.download_blob().readall()

        # Read the Excel file into a pandas DataFrame
        historical_meta = pd.read_csv(BytesIO(blob_data))

        # Convert DataFrame to CSV
        csv_data = historical_meta.to_csv(index=False)

        # Return CSV data
        return func.HttpResponse(csv_data, mimetype="text/csv", status_code=200, headers={
            "Content-Disposition": f"attachment; filename={blob_name}"
        })

    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)
        return func.HttpResponse(f"Error reading blob: {e}", status_code=500)


@app.route(route="GetVendorDetailsFile", auth_level=func.AuthLevel.FUNCTION)
def GetVendorDetailsFile(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function to read Vendor Info List Excel file from Blob Storage.')

    # Blob storage details
    blob_name = vendor_details_blob_name

    # Initialize BlobServiceClient
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)

    try:
        # Download the blob content as bytes
        blob_data = blob_client.download_blob().readall()

        # Read the Excel file into a pandas DataFrame
        vendor_country_df = pd.read_excel(BytesIO(blob_data))
        vendor_country_df = vendor_country_df[['Vendor', 'Vendor Name', 'Vendor Country']]
        vendor_country_df.columns = ['Vendor', 'Vendor Name', 'Country Key']
        vendor_country_df['Country Key'] = vendor_country_df['Country Key'].replace('GB', 'UK')

        # Convert DataFrame to CSV
        csv_data = vendor_country_df.to_csv(index=False)

        # Return CSV data
        return func.HttpResponse(csv_data, mimetype="text/csv", status_code=200, headers={
            "Content-Disposition": f"attachment; filename={blob_name.replace('.xlsx', '.csv')}"
        })

    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)
        return func.HttpResponse(f"Error reading blob: {e}", status_code=500)


@app.route(route="GetCompanyCodeDetailsFile", auth_level=func.AuthLevel.FUNCTION)
def GetCompanyCodeDetailsFile(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function to read Company Code Info List Excel file from Blob Storage.')

    # Blob storage details
    blob_name = company_code_details_blob_name

    # Initialize BlobServiceClient
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)

    try:
        # Download the blob content as bytes
        blob_data = blob_client.download_blob().readall()

        # Read the Excel file into a pandas DataFrame
        company_code_df = pd.read_excel(BytesIO(blob_data))
        company_code_df = company_code_df[['Company\ncode', 'Company Name', 'Country']].dropna(subset=['Company\ncode', 'Country']) # type: ignore
        company_code_df.columns = ['Company Code', 'Company Name', 'Reporting Country']
        country_mapper = {'Sweden': 'SE', 'Denmark': 'DK', 'Norway': 'NO', 'Finland': 'FI', 'United Kingdom(UK)': 'UK', 'Netherland': 'NE', 'Great Britain': 'UK'}
        company_code_df['Reporting Country'] = company_code_df['Reporting Country'].apply(lambda x: country_mapper[x])
        company_code_df['Company Code'] = company_code_df['Company Code'].astype(str)

        # Convert DataFrame to CSV
        csv_data = company_code_df.to_csv(index=False)

        # Return CSV data
        return func.HttpResponse(csv_data, mimetype="text/csv", status_code=200, headers={
            "Content-Disposition": f"attachment; filename={blob_name.replace('.xlsx', '.csv')}"
        })

    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)
        return func.HttpResponse(f"Error reading blob: {e}", status_code=500)

@app.route(route="GetTaxCodeDescriptionFile", auth_level=func.AuthLevel.FUNCTION)
def GetTaxCodeDescriptionFile(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function to read Tax Code Description Excel file from Blob Storage.')

    # Blob storage details
    blob_name = tax_code_description_blob_name

    # Initialize BlobServiceClient
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)

    try:
        # Download the blob content as bytes
        blob_data = blob_client.download_blob().readall()

        # Read the Excel file into a pandas DataFrame
        tax_code_mapping = pd.ExcelFile(BytesIO(blob_data))
        df_list = []
        for sheet_name in tax_code_mapping.sheet_names[1:]:
            dff = pd.read_excel(BytesIO(blob_data), sheet_name=sheet_name, skiprows=3)
            df_list.append(dff[['Tax code', 'Description']].dropna())

        concatenated_tax_description = pd.concat(df_list).drop_duplicates()

        # Convert DataFrame to CSV
        csv_data = concatenated_tax_description.to_csv(index=False)

        # Return CSV data
        return func.HttpResponse(csv_data, mimetype="text/csv", status_code=200, headers={
            "Content-Disposition": f"attachment; filename={blob_name.replace('.xlsx', '.csv')}"
        })

    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)
        return func.HttpResponse(f"Error reading blob: {e}", status_code=500)

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
 





@app.route(route="httpget", methods=["GET"])
def http_get(req: func.HttpRequest) -> func.HttpResponse:

    logging.info(f"Processing GET request. Name:")

    return func.HttpResponse(f"Hello,!")
