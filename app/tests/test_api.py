from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_predict():
    with open("tests/test_image.jpg", "rb") as image_file:
        response = client.post("/predict", files={"file": ("filename", image_file, "image/jpeg")})
    assert response.status_code == 200
    assert "food" in response.json()
    assert "calories" in response.json()
    assert "nutrients" in response.json()
