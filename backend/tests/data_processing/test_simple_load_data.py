import unittest
from unittest.mock import patch
from backend.src.data_processing.simple_load_data import SimpleLoadData

class TestSimpleLoadData(unittest.TestCase):

    @patch('builtins.open')
    @patch('requests.get')
    def test_download_file(self, mock_get, mock_open):
        loader = SimpleLoadData()

        # Arrange
        url = 'https://www.something.com/test.txt'
        filepath = 'test_download_path/test.txt'

        mock_get.return_value.__enter__.return_value = mock_get
        mock_get.return_value.raise_for_status.return_value = None
        mock_get.iter_content.return_value = [b'test data']

        mock_open.return_value.__enter__.return_value = mock_open
        mock_open.return_value.write.return_value = None

        # Act
        loader.download_file(url, filepath)

        # Assert
        mock_get.assert_called_once_with(url, stream=True)
        mock_get.raise_for_status.assert_called_once_with()
        mock_get.iter_content.assert_called_once_with(chunk_size=8192)
        mock_open.assert_called_once_with(filepath, 'wb')
        mock_open.write.assert_called_once_with(b'test data')
        

    @patch('backend.src.data_processing.simple_load_data.SimpleLoadData.download_file')
    def test_load(self, mock_download_file):
        loader = SimpleLoadData()

        # Arrange
        file_urls = ['https://www.something.com/test.txt']
        download_path = 'test_download_path'

        # Act
        loader.load(file_urls, download_path)

        # Assert
        mock_download_file.assert_called_once_with('https://www.something.com/test.txt', 'test_download_path/test.txt')

    def test_save(self):
        # Arrange
        simple_load_data = SimpleLoadData()

        # Act and Assert
        with self.assertRaises(NotImplementedError):
            simple_load_data.save('test.txt', 'test data')

if __name__ == '__main__':
    unittest.main()