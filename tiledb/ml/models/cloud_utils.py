import os
import tiledb
import tiledb.cloud

from typing import List, Union

from . import FILETYPE_ML_MODEL, CLOUD_MODELS


def get_s3_prefix(namespace: str) -> Union[str, None]:
    """
    Get S3 path from the user profile or organization profile.
    :param namespace: str. User's namespace
    :return: str or None. A S3 prefix
    """
    profile = tiledb.cloud.client.user_profile()

    if namespace == profile.username:
        if profile.default_s3_path is not None:
            return os.path.join(profile.default_s3_path, CLOUD_MODELS)
        else:
            return None
    else:
        organization = tiledb.cloud.client.organization(namespace)
        if organization.default_s3_path is not None:
            return os.path.join(organization.default_s3_path, CLOUD_MODELS)
        else:
            return None


def get_model_list(
    category: str, namespace: str
) -> Union[List[tiledb.cloud.rest_api.models.array_info.ArrayInfo], None]:
    """
    Returns a list of model TileDB arrays depending on the specified category.
    :param category: str. Array category. Could be "owned", "shared" or "public"
    :param namespace: str. User's namespace
    :return: List or None. A List of ArrayInfo objects that contain information on
    all our model arrays on TileDB-Cloud.
    """
    if category == "owned":
        return tiledb.cloud.client.list_arrays(
            file_type=[FILETYPE_ML_MODEL],
            namespace=namespace,
            async_req=True,
        ).get()
    elif category == "shared":
        return tiledb.cloud.client.list_shared_arrays(
            file_type=[FILETYPE_ML_MODEL],
            namespace=namespace,
            async_req=True,
        ).get()
    elif category == "public":
        return tiledb.cloud.client.list_public_arrays(
            file_type=[FILETYPE_ML_MODEL],
            namespace=namespace,
            async_req=True,
        ).get()
    else:
        return None
