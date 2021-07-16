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
        assert get_s3_prefix("foo") == "bar/ml_models"

        profile = mocker.patch(
            "tiledb.cloud.client.user_profile",
            return_value=mocker.Mock(username="foo", default_s3_path=None),
        )
        profile.default_s3_path = "bar"
        assert get_s3_prefix("foo") is None

        org = mocker.patch(
            "tiledb.cloud.client.organization",
            return_value=mocker.Mock(default_s3_path="orgbar"),
        )
        org.default_s3_path = "orgbar"
        assert get_s3_prefix("nofoo") == "orgbar/ml_models"

        org = mocker.patch(
            "tiledb.cloud.client.organization",
            return_value=mocker.Mock(
                default_s3_path=None,
            ),
        )
        org.default_s3_path = "orgbar"
        assert get_s3_prefix("nofoo") is None
