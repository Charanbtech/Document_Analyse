import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_status_endpoint(client):
    response = client.get('/api/status')
    assert response.status_code == 200
    data = response.get_json()
    assert 'model_loaded' in data
    assert 'db_stats' in data
