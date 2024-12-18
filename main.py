import timm
import torch
from PIL import Image
from torchvision import transforms
import torch.nn.functional as functional

# PATHS
front_path = "front_test.png" 
back_path = "back_test.png"
side_path = "side_test.png"


# load EfficientNet-B0 Modell mit timm 
model = timm.create_model('efficientnet_b0', pretrained=True)
model.eval() 

transform = transforms.Compose([
    transforms.Resize(224),  # EffizientNet erwartet 224x224 pixel als eingabe
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def generate_embedding(image_path):
    image = Image.open(image_path).convert("RGB")  
    input_tensor = transform(image).unsqueeze(0)  # Bild transformieren und Batch-Dimension hinzufügen
    with torch.no_grad():
        features = model.forward_features(input_tensor)  # Feature-Map extrahieren
        embedding = torch.mean(features, dim=[2, 3])  # Global Average Pooling 
    return embedding

front_e = generate_embedding(front_path)
back_e = generate_embedding(back_path)
side_e = generate_embedding(side_path)


cs_ff = functional.cosine_similarity(front_e, front_e).item()
cs_fb = functional.cosine_similarity(front_e, back_e).item()
cs_fs = functional.cosine_similarity(front_e, side_e).item()
print(f"Cosine Similarity zwischen (front/front): {cs_ff:.4f}")
print(f"Cosine Similarity zwischen (front/back): {cs_fb:.4f}")
print(f"Cosine Similarity zwischen (front/side): {cs_fs:.4f}")