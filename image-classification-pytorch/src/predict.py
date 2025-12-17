from pathlib import Path
from PIL import Image
import torch
import torch.nn.functional as F

def predict_folder(model, folder_path: Path, class_names, transform, device, topk=3):
    model.eval()
    results = {}

    image_files = [
        p for p in folder_path.iterdir()
        if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ]

    with torch.no_grad():
        for img_path in image_files:
            img = Image.open(img_path).convert("RGB")
            tensor = transform(img).unsqueeze(0).to(device)

            outputs = model(tensor)
            probs = F.softmax(outputs, dim=1)
            top_probs, top_idxs = probs.topk(topk, dim=1)

            results[img_path.name] = [
                (class_names[idx], float(prob))
                for prob, idx in zip(top_probs[0], top_idxs[0])
            ]

    return results
