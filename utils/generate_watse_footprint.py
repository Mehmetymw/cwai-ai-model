import pandas as pd
import numpy as np
import json

with open('./data/json','r' ) as file:
    data = json.load(file)
    
# Rastgele veri oluşturma
np.random.seed(42)
data = {
    "meal_type": np.random.choice(data["classes"], size=1000),
    "portion_size": np.random.randint(200, 800, size=1000),  # Gram
    "time_spent": np.random.randint(10, 60, size=1000),  # Dakika
    "waste": lambda x: np.maximum(0, x["portion_size"] - np.random.randint(150, 700, size=1000)),  # Gram
    "carbon_footprint": lambda x: np.where(
        x["meal_type"] == "meat", 
        np.random.uniform(20, 30, size=1000), 
        np.random.uniform(1.5, 10, size=1000)
    )  # Karbon ayak izi (kg CO2)
}

# Veri çerçevesi oluşturma
df = pd.DataFrame(data)
df["waste"] = np.maximum(0, df["portion_size"] - np.random.randint(150, 700, size=1000))
df["carbon_footprint"] = np.where(
    df["meal_type"] == "meat",
    np.random.uniform(20, 30, size=1000),
    np.random.uniform(1.5, 10, size=1000)
)

# CSV olarak kaydetme
df.to_csv("waste_data.csv", index=False)
print("Sentetik veri oluşturuldu: waste_data.csv")
