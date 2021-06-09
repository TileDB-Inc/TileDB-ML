import os
import tiledb
import tiledb.cloud
from typing import List, Optional

CLOUD_MODELS = "ml_models"
FILETYPE_ML_MODEL = "ml_model"


def get_s3_prefix(namespace: str) -> Optional[str, None]:
    """
    Get S3 path from the user profile or organization profile
    :return: s3 path or error
    :namespace: str
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
) -> Optional[List[tiledb.cloud.rest_api.models.array_info.ArrayInfo], None]:
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
