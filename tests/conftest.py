import json
import pytest
import os

@pytest.fixture
def load_mock_data():
    def _loader(filename):
        path = os.path.join(os.path.dirname(__file__), 'mock_data', filename)
        with open(path, 'r') as file:
            return json.load(file)
    return _loader