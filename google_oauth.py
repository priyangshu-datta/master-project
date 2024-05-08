from google.oauth2 import service_account
import os
import google.ai.generativelanguage as glm

credentials = service_account.Credentials.from_service_account_file(
    'service_account_key.json'
    
)

scoped_credentials = credentials.with_scopes(
    [
        "https://www.googleapis.com/auth/cloud-platform",
        "https://www.googleapis.com/auth/generative-language.retriever",
    ]
)

generative_service_client = glm.GenerativeServiceClient(credentials=scoped_credentials)