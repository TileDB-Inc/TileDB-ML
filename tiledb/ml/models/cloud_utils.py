import os
import tiledb
import tiledb.cloud

from typing import Union

from . import CLOUD_MODELS


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
