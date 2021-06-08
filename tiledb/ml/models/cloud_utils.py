import os
import tiledb
import tiledb.cloud
from typing import List

CLOUD_MODELS = "ml_models"
FILETYPE_ML_NODEL = "ml_model"


def get_s3_prefix(namespace: str) -> str:
    """
    Get S3 path from the user profile or organization profile
    :return: s3 path or error
    :namespace: str
    """
    try:
        profile = tiledb.cloud.client.user_profile()

        if namespace == profile.username:
            if profile.default_s3_path is not None:
                return os.path.join(profile.default_s3_path, CLOUD_MODELS)
        else:
            organization = tiledb.cloud.client.organization(namespace)
            if organization.default_s3_path is not None:
                return os.path.join(organization.default_s3_path, CLOUD_MODELS)
    except tiledb.cloud.tiledb_cloud_error.TileDBCloudError as e:
        raise Exception(e)

    return None


def get_s3_credentials(namespace: str) -> str:
    """
    Get credentials for default S3 path from the user profile or organization profile
    :return: s3 credentials or error
    :namespace: str
    """
    try:
        profile = tiledb.cloud.client.user_profile()

        if namespace == profile.username:
            if profile.default_s3_path_credentials_name is not None:
                return profile.default_s3_path_credentials_name
        else:
            organization = tiledb.cloud.client.organization(namespace)
            if organization.default_s3_path_credentials_name is not None:
                return organization.default_s3_path_credentials_name
    except tiledb.cloud.tiledb_cloud_error.TileDBCloudError as e:
        raise Exception(e)

    return None


def get_model_list(
    category: str, namespace: str
) -> List[tiledb.cloud.rest_api.models.array_info.ArrayInfo]:
    if category == "owned":
        model_listing_future = tiledb.cloud.client.list_arrays(
            file_type=[FILETYPE_ML_NODEL],
            namespace=namespace,
            async_req=True,
        )
    elif category == "shared":
        model_listing_future = tiledb.cloud.client.list_shared_arrays(
            file_type=[FILETYPE_ML_NODEL],
            namespace=namespace,
            async_req=True,
        )
    elif category == "public":
        model_listing_future = tiledb.cloud.client.list_public_arrays(
            file_type=[FILETYPE_ML_NODEL],
            namespace=namespace,
            async_req=True,
        )

    return model_listing_future.get()
