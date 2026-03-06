import argparse
import yaml
import torch
from torchvision import transforms
from PIL import Image
from models.unet import UNet

def dehaze_image(model, image_path, output_path, image_size=128, device='cpu'):
    image = Image.open(image_path).convert('RGB')
    original_size = image.size
    preprocess = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor()])
    input_tensor = preprocess(image).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        output_tensor = model(input_tensor)
    output_tensor = output_tensor.squeeze(0)
    output_tensor = output_tensor.clamp(0.0, 1.0)
    to_pil = transforms.ToPILImage()
    output_image = to_pil(output_tensor.cpu())
    output_image = output_image.resize(original_size, Image.LANCZOS)
    output_image.save(output_path)
    print(f'✅ Dehazed image save ho gayi: {output_path}')
    return output_image

def main():
    parser = argparse.ArgumentParser(description='Ek image dehaze karo')
    parser.add_argument('--image', type=str, required=True, help='Hazy image ka path')
    parser.add_argument('--output', type=str, default='dehazed_output.jpg', help='Output image kahan save karo')
    parser.add_argument('--checkpoint', type=str, default='experiments/best_model.pth', help='Model checkpoint ka path')
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='Config file ka path')
    args = parser.parse_args()
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(in_channels=config['model']['in_channels'], out_channels=config['model']['out_channels'], features=config['model']['features'], use_batch_norm=config['model']['use_batch_norm'], dropout_rate=config['model']['dropout_rate']).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f'Model loaded: {args.checkpoint}')
    dehaze_image(model, args.image, args.output, image_size=config['data']['image_size'], device=device)
if __name__ == '__main__':
    main()