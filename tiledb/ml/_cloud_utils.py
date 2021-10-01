import os
from typing import Mapping, Optional

import tiledb.cloud

import tiledb

from . import CLOUD_MODELS, FILETYPE_ML_MODEL


def get_s3_prefix(namespace: str) -> Optional[str]:
    """
    Get S3 path from the user profile or organization profile.
    :param namespace: User's namespace
    :return: S3 prefix
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


def update_file_properties(uri: str, file_properties: Mapping[str, str]) -> None:
    tiledb.cloud.array.update_file_properties(
        uri=uri,
        file_type=FILETYPE_ML_MODEL,
        file_properties=file_properties,
    )
