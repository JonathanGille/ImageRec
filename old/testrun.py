import timm
import torch
from PIL import Image
from torchvision import transforms

# EfficientNet-B0 Modell laden
model = timm.create_model('efficientnet_b0', pretrained=True)
model.eval()  # Modell in den Evaluierungsmodus setzen

# Vorverarbeitung des Bildes
transform = transforms.Compose([
    transforms.Resize(224),  # EffizientNet erwartet 224x224 Eingabebilder
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Beispielbild laden
image_path = "test1.png"  # Pfad zu deinem Bild
image = Image.open(image_path).convert("RGB")
input_tensor = transform(image).unsqueeze(0)  # Batch-Dimension hinzufügen

# Embedding extrahieren
with torch.no_grad():
    features = model.forward_features(input_tensor)

print("Embeddings:", features.shape)  # Ausgabe: [1, 1280, 7, 7] für EfficientNet-B0