import tiledb.cloud

FILETYPE_ML_NODEL = "ml_model"


class ModelList:
    """
    Use this to control caching
    """

    def __init__(self, category, namespace=None):
        """
        Create an ArrayListing
        :param category: category to list
        :param namespace: namespace to filter to
        """
        self.category = category
        self.namespace = namespace
        self.model_listing_future = None

    def fetch(self):
        if self.category == "owned":
            self.model_listing_future = tiledb.cloud.client.list_arrays(
                file_type=[FILETYPE_ML_NODEL],
                namespace=self.namespace,
                async_req=True,
            )
        elif self.category == "shared":
            self.model_listing_future = tiledb.cloud.client.list_shared_arrays(
                file_type=[FILETYPE_ML_NODEL],
                namespace=self.namespace,
                async_req=True,
            )
        elif self.category == "public":
            self.model_listing_future = tiledb.cloud.client.list_public_arrays(
                file_type=[FILETYPE_ML_NODEL],
                namespace=self.namespace,
                async_req=True,
            )

        return self

    def get(self):
        if self.model_listing_future is None:
            self.fetch()

        return self.model_listing_future.get()

    def models(self):
        ret = self.get()
        if ret is not None:
            return ret.arrays
        return None