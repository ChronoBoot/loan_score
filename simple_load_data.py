import os
from load_data_abc import LoadData

class SimpleLoadData(LoadData):
    def load(self, blob_service_client, container_name, file_names, download_path):
        container_client = blob_service_client.get_container_client(container=container_name)

        if not os.path.exists(download_path):
            os.makedirs(download_path)

        for blob_name in file_names:
            download_file_path = f"{download_path}{blob_name}"
            if not os.path.exists(download_file_path):
                blob_client = container_client.get_blob_client(blob_name)
                with open(download_file_path, "wb") as download_file:
                    download_file.write(blob_client.download_blob().readall())

    def save(self, file_path, data):
        raise NotImplementedError("The save method is not implemented")