import os
import unittest
import pandas as pd
from backend.src.data_processing.simple_read_data import SimpleReadData

class TestSimpleReadData(unittest.TestCase):
    def setUp(self):
        self.reader = SimpleReadData()

    def test_read_data(self):
        # Arrange


        # Act
        #data = self.reader.read_data('data', False, 1000)
        #data_concat = self.reader.read_data('data', True, 10000)

        # Assert


        

    

if __name__ == '__main__':
    unittest.main()