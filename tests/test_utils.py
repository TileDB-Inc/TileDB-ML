import os
from tiledb.ml._cloud_utils import get_s3_prefix

CLOUD_MODELS = "ml_models"
FILETYPE_ML_MODEL = "ml_model"


class TestCloudUtils:
    def test_get_s3_prefix(self, mocker):

        profile = mocker.patch(
            "tiledb.cloud.client.user_profile",
            return_value=mocker.Mock(username="foo", default_s3_path="bar"),
        )
        profile.default_s3_path = "bar"
        actual = get_s3_prefix("foo")
        expected = os.path.join(profile.default_s3_path, CLOUD_MODELS)
        assert actual == expected

        profile = mocker.patch(
            "tiledb.cloud.client.user_profile",
            return_value=mocker.Mock(username="foo", default_s3_path=None),
        )
        profile.default_s3_path = "bar"
        actual = get_s3_prefix("foo")
        expected = None
        assert actual == expected

        org = mocker.patch(
            "tiledb.cloud.client.organization",
            return_value=mocker.Mock(default_s3_path="orgbar"),
        )
        org.default_s3_path = "orgbar"
        actual = get_s3_prefix("nofoo")
        expected = os.path.join(org.default_s3_path, CLOUD_MODELS)
        assert actual == expected

        org = mocker.patch(
            "tiledb.cloud.client.organization",
            return_value=mocker.Mock(
                default_s3_path=None,
            ),
        )
        org.default_s3_path = "orgbar"
        actual = get_s3_prefix("nofoo")
        expected = None
        assert actual == expected
