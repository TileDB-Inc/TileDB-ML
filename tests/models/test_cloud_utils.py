from tiledb.ml.models._cloud_utils import update_file_properties


class TestCloudUtils:
    def test_update_file_properties(self, mocker):
        mock_tiledb_cloud_update_file_properties = mocker.patch(
            "tiledb.client.array.update_file_properties"
        )

        uri = "tiledb_array"
        file_properties = {}

        update_file_properties(uri=uri, file_properties=file_properties)

        mock_tiledb_cloud_update_file_properties.assert_called_once_with(
            uri=uri, file_type="ml_model", file_properties=file_properties
        )
