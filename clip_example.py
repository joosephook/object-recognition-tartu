import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

kissing_students = Image.open("images/img296.jpg")
image = preprocess(kissing_students).unsqueeze(0).to(device)
labels = ["a statue", "a statue of a kissing couple under an umbrella", "a fountain", "a building", "a garbage bin"]
text = clip.tokenize(labels).to(device)


with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)

    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()
    for label, prob in zip(labels, probs.ravel()):
        print(f'{prob:.3f}', label)