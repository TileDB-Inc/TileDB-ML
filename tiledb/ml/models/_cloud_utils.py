import os
from typing import Mapping, Optional

try:
    import tiledb.client
except ImportError:
    pass

CLOUD_MODELS = "ml_models"
FILETYPE_ML_MODEL = "ml_model"


def get_s3_prefix(teamspace: Optional[str]) -> Optional[str]:
    """
    Get S3 path from the user profile or organization profile.
    :param namespace: User's namespace
    :return: S3 prefix
    """
    if teamspace is None:
        # TODO: adjust the default location as needed for 3.0.
        # Another option would be the default charged teamspace.
        # teamspace = tiledb.client.client.default_charged_namespace()
        # s3_path = tiledb.client.client.organization(teamspace).default_s3_path
        profile = tiledb.client.client.user_profile()
        s3_path = profile.default_s3_path
        teamspace = profile.username
    else:
        # TODO: adjust this logic as needed for 3.0.
        try:
            org = tiledb.client.client.organization(teamspace)
            s3_path = org.default_s3_path
        except Exception:
            # Handle the case where the teamspace is the user's username.
            profile = tiledb.client.client.user_profile()
            if teamspace == profile.username:
                s3_path = profile.default_s3_path
            else:
                raise
    
    return os.path.join(s3_path, CLOUD_MODELS) if s3_path is not None else None

    # if teamspace is None:
    #     return None
    # profile = tiledb.client.client.user_profile()
    # if namespace == profile.username:
    #     s3_path = profile.default_s3_path
    # else:
    #     s3_path = tiledb.client.client.organization(namespace).default_s3_path
    # return os.path.join(s3_path, CLOUD_MODELS) if s3_path is not None else None


def update_file_properties(uri: str, file_properties: Mapping[str, str]) -> None:
    tiledb.client.array.update_file_properties(
        uri=uri,
        file_type=FILETYPE_ML_MODEL,
        file_properties=file_properties,
    )


def get_cloud_uri(uri: str, teamspace: Optional[str]) -> str:
    s3_prefix = get_s3_prefix(teamspace)
    if s3_prefix is None:
        raise ValueError(
            f"You must set the default s3 prefix path for ML models in "
            f"{teamspace} profile settings on TileDB-Cloud"
        )

    return f"tiledb://{teamspace}/{os.path.join(s3_prefix, uri)}"
