from tiledb.ml.models._cloud_utils import (
    get_cloud_uri,
    get_s3_prefix,
    update_file_properties,
)


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

    def test_get_cloud_uri(self, mocker):
        mocker.patch(
            "tiledb.ml.models._cloud_utils.get_s3_prefix", return_value="s3://"
        )

        assert "tiledb://test_namespace/s3://tiledb_array" == get_cloud_uri(
            uri="tiledb_array", namespace="test_namespace"
        )

    def test_update_file_properties(self, mocker):
        mock_tiledb_cloud_update_file_properties = mocker.patch(
            "tiledb.cloud.array.update_file_properties"
        )

        uri = "tiledb_array"
        file_properties = {}

        update_file_properties(uri=uri, file_properties=file_properties)

        mock_tiledb_cloud_update_file_properties.assert_called_once_with(
            uri=uri, file_type="ml_model", file_properties=file_properties
        )
