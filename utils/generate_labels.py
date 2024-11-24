import os
import json

def generate_labels(data_dir="data/processed/food/train", output_file="data/labels.json"):
    """
    Belirtilen veri dizinindeki klasör adlarını tarayarak labels.json dosyasını oluşturur.
    Her klasör bir sınıf olarak kabul edilir.
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"The specified directory '{data_dir}' does not exist.")

    # Veri dizinindeki klasörleri tarar
    classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])

    # Etiketleri JSON formatına dönüştürme
    labels = {"classes": classes}

    # JSON dosyasını kaydetme
    with open(output_file, "w") as f:
        json.dump(labels, f, indent=4)

    print(f"{output_file} dosyası başarıyla oluşturuldu.")

# Betiği çalıştır
if __name__ == "__main__":
    generate_labels()
