from app.models.predict import predict
from PIL import Image

def test_predict_model():
    image = Image.open("tests/test_image.jpg")
    result = predict(image)
    assert "food" in result
    assert "calories" in result
    assert "nutrients" in result
