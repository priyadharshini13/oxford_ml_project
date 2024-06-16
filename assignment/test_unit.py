import unittest
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAXResultsWrapper
import os
import kellys  # Assuming kellys.py is in the same directory as this script

class TestKellysFunctions(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Ensure the current working directory is set to the project root
        os.chdir(os.path.dirname(__file__))
        print(f"Current working directory: {os.getcwd()}")

    def test_load_data(self):
        # Test loading data function
        data = kellys.load_data()
        self.assertIsInstance(data, pd.DataFrame)
        self.assertFalse(data.empty, "Loaded DataFrame should not be empty")

    def test_preprocess(self):
        # Test preprocessing function
        sample_data = pd.DataFrame({
            'Date of Sale': pd.date_range('2023-01-01', periods=10),
            'Country': ['USA'] * 10,
            'Store ID': [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
            'Product ID': [101, 102, 103, 104, 105, 101, 102, 103, 104, 105],
            'Product Category': ['A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C', 'A'],
            'Units Sold': [10, 20, 30, 40, 50, 10, 20, 30, 40, 50]
        })
        
        train_data, test_data, train_exog, test_exog = kellys.preprocess(sample_data)

        # Check if training and testing data are of correct types and sizes
        self.assertIsInstance(train_data, pd.DataFrame)
        self.assertIsInstance(test_data, pd.DataFrame)
        self.assertIsInstance(train_exog, pd.DataFrame)
        self.assertIsInstance(test_exog, pd.DataFrame)
        self.assertGreater(len(train_data), 0)
        self.assertGreater(len(test_data), 0)
        self.assertGreater(len(train_exog), 0)
        self.assertGreater(len(test_exog), 0)

    def test_train(self):
        # Test training function
        sample_data = pd.DataFrame({
            'Date of Sale': pd.date_range('2023-01-01', periods=10),
            'Country': ['USA'] * 10,
            'Store ID': [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
            'Product ID': [101, 102, 103, 104, 105, 101, 102, 103, 104, 105],
            'Product Category': ['A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C', 'A'],
            'Units Sold': [10, 20, 30, 40, 50, 10, 20, 30, 40, 50]
        })

        train_data, test_data, train_exog, test_exog = kellys.preprocess(sample_data)
        print(train_data)
        print(train_exog)
        sarima_results = kellys.train(train_data, train_exog)

        # Check if sarima_results is an instance of SARIMAXResults
        self.assertIsInstance(sarima_results, SARIMAXResultsWrapper)

if __name__ == '__main__':
    unittest.main()
