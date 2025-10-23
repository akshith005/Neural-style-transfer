# ---------- Neural Style Transfer (with Comparison Display) ----------
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# We'll expose the main NST logic as a function so other modules (like the server) can call it directly.

def run_nst(content_path: str = "content.png", style_path: str = "style.png", output_path: str = "output.jpg",
            image_size: int = 512, num_steps: int = 50, content_weight: float = 1, style_weight: float = 1e6,
            device: str | torch.device = None):
    """Run neural style transfer and save output to output_path.

    Returns the path to the saved output image on success.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------- IMAGE LOADING ----------
    loader = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])

    def load_image(path):
        image = Image.open(path).convert("RGB")
        image = loader(image).unsqueeze(0)
        return image.to(device, torch.float)

    content_image = load_image(content_path)
    style_image = load_image(style_path)

    # No GUI/plotting in server mode. We convert tensors to images only when saving.

    # ---------- LOAD VGG MODEL ----------
    cnn = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features.to(device).eval()

    for param in cnn.parameters():
        param.requires_grad = False

    # ---------- NORMALIZATION ----------
    class Normalization(nn.Module):
        def __init__(self, mean, std):
            super(Normalization, self).__init__()
            self.mean = mean.clone().detach().view(-1, 1, 1)
            self.std = std.clone().detach().view(-1, 1, 1)
        def forward(self, img):
            return (img - self.mean) / self.std

    normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

    # ---------- LOSS CLASSES ----------
    class ContentLoss(nn.Module):
        def __init__(self, target):
            super().__init__()
            self.target = target.detach()
        def forward(self, x):
            self.loss = nn.functional.mse_loss(x, self.target)
            return x

    class StyleLoss(nn.Module):
        def __init__(self, target):
            super().__init__()
            self.target = self.gram_matrix(target).detach()
        def gram_matrix(self, x):
            b, c, h, w = x.size()
            features = x.view(b * c, h * w)
            G = torch.mm(features, features.t())
            return G / (b * c * h * w)
        def forward(self, x):
            G = self.gram_matrix(x)
            self.loss = nn.functional.mse_loss(G, self.target)
            return x

    # ---------- BUILD MODEL ----------
    def get_style_model_and_losses(cnn, style_img, content_img):
        normalization = Normalization(normalization_mean, normalization_std)
        content_losses = []
        style_losses = []

        model = nn.Sequential(normalization)
        i = 0
        for layer in cnn.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = f'conv_{i}'
            elif isinstance(layer, nn.ReLU):
                name = f'relu_{i}'
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = f'pool_{i}'
            else:
                name = f'layer_{i}'
            model.add_module(name, layer)

            if name == 'conv_4':  # Content representation layer
                target = model(content_img).detach()
                content_loss = ContentLoss(target)
                model.add_module(f"content_loss_{i}", content_loss)
                content_losses.append(content_loss)

            if name in ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']:
                target = model(style_img).detach()
                style_loss = StyleLoss(target)
                model.add_module(f"style_loss_{i}", style_loss)
                style_losses.append(style_loss)

        return model, style_losses, content_losses

    # ---------- GENERATED IMAGE ----------
    generated = content_image.clone().requires_grad_(True)

    # ---------- MODEL & OPTIMIZER ----------
    model, style_losses, content_losses = get_style_model_and_losses(cnn, style_image, content_image)
    optimizer = optim.LBFGS([generated])

    # ---------- TRAIN LOOP ----------
    print("Running Style Transfer...")
    run = [0]
    while run[0] <= num_steps:
        def closure():
            with torch.no_grad():
                generated.clamp_(0, 1)
            optimizer.zero_grad()
            model(generated)
            style_score = sum(sl.loss for sl in style_losses)
            content_score = sum(cl.loss for cl in content_losses)
            loss = style_weight * style_score + content_weight * content_score
            loss.backward()
            if run[0] % 50 == 0:
                print(f"Step {run[0]:3d} | Style: {style_score.item():.4f} | Content: {content_score.item():.4f}")
            run[0] += 1
            return loss
        optimizer.step(closure)

    # ---------- OUTPUT ----------
    with torch.no_grad():
        generated.clamp_(0, 1)

    final_img = generated.cpu().squeeze(0)
    final_img = transforms.ToPILImage()(final_img)
    final_img.save(output_path)

    # No plotting here — just save the output image to disk and return its path.
    print(f"✅ Done! Output saved to {output_path}")
    return output_path


if __name__ == '__main__':
    # Allow running as a script using default filenames in project root
    run_nst()
