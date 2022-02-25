import os
from typing import Dict, Optional

try:
    import tiledb.cloud
except ImportError:
    pass

CLOUD_MODELS = "ml_models"
FILETYPE_ML_MODEL = "ml_model"


def get_s3_prefix(namespace: Optional[str]) -> Optional[str]:
    """
    Get S3 path from the user profile or organization profile.
    :param namespace: User's namespace
    :return: S3 prefix
    """
    if namespace is None:
        return None
    profile = tiledb.cloud.client.user_profile()
    if namespace == profile.username:
        s3_path = profile.default_s3_path
    else:
        s3_path = tiledb.cloud.client.organization(namespace).default_s3_path
    return os.path.join(s3_path, CLOUD_MODELS) if s3_path is not None else None


def update_file_properties(uri: str, file_properties: Dict[str, str]) -> None:
    tiledb.cloud.array.update_file_properties(
        uri=uri,
        file_type=FILETYPE_ML_MODEL,
        file_properties=file_properties,
    )


def get_cloud_uri(uri: str, namespace: Optional[str]) -> str:
    s3_prefix = get_s3_prefix(namespace)
    if s3_prefix is None:
        raise ValueError(
            f"You must set the default s3 prefix path for ML models in "
            f"{namespace} profile settings on TileDB-Cloud"
        )

    return f"tiledb://{namespace}/{os.path.join(s3_prefix, uri)}"
