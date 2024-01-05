import unittest
from unittest.mock import Mock, patch
from azure.storage.blob import BlobServiceClient, ContainerClient, BlobClient
from backend.src.data_processing.simple_load_data import SimpleLoadData

class TestSimpleLoadData(unittest.TestCase):
    @patch.dict('os.environ', {
        'AZURE_STORAGE_CONNECTION_STRING': 'DefaultEndpointsProtocol=https;AccountName=testaccount;AccountKey=testkey;BlobEndpoint=testendpoint',
        'AZURE_STORAGE_CONTAINER_NAME': 'test_container_name'
    })
    @patch('backend.src.data_processing.simple_load_data.load_dotenv')
    @patch('os.path.exists')
    @patch('os.makedirs')
    @patch('builtins.open', new_callable=unittest.mock.mock_open)
    def test_load(self, mock_open, mock_makedirs, mock_exists, mock_load_dotenv):
        # Arrange
        mock_blob_service_client = Mock(spec=BlobServiceClient)
        mock_container_client = Mock(spec=ContainerClient)
        mock_blob_client = Mock(spec=BlobClient)
        mock_blob_service_client.get_container_client.return_value = mock_container_client
        mock_container_client.get_blob_client.return_value = mock_blob_client
        mock_blob_client.download_blob().readall.return_value = b'test data'
        mock_exists.return_value = False

        simple_load_data = SimpleLoadData()
        container_name = 'test_container_name'
        file_names = ['test1.txt', 'test2.txt']
        download_path = '/path/to/download/'
        simple_load_data.blob_service_client = mock_blob_service_client
        simple_load_data.container_name = container_name

        # Act
        simple_load_data.load(file_names, download_path)

        # Assert
        mock_blob_service_client.get_container_client.assert_called_once_with(container=container_name)
        mock_exists.assert_called()
        mock_makedirs.assert_called_once_with(download_path)
        mock_open.assert_called()
        mock_blob_client.download_blob().readall.assert_called()

    def test_save(self):
        # Arrange
        simple_load_data = SimpleLoadData()

        # Act and Assert
        with self.assertRaises(NotImplementedError):
            simple_load_data.save('test.txt', 'test data')

if __name__ == '__main__':
    unittest.main()