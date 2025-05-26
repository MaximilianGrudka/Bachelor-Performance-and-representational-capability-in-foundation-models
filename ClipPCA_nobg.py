from models.extractorClip import ViTExtractor
from torchvision import transforms as T
import torch
import numpy as np
from argparse import ArgumentParser
from PIL import Image
from sklearn.decomposition import PCA
from torchvision.transforms.functional import resize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def minmax_norm(x):
    """Min-max normalization."""
    return (x - x.min()) / (x.max() - x.min())
    

def visualize(args):
    # Load the image
    input_img = Image.open(args.image_path).convert("RGB")
    input_img = T.Compose([
        T.Resize((224, 224)),  
        T.ToTensor()
    ])(input_img).unsqueeze(0).to(device)


    # Define the extractor
    dino_preprocess = T.Compose([
        T.Resize(224,), 
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    vit_extractor = ViTExtractor(args.model_name, device)

    # Extract self-similarity keys
    with torch.no_grad():
        keys_self_sim = vit_extractor.get_keys_self_sim_from_input(dino_preprocess(input_img), args.layer)

    # Convert to numpy
    keys_self_sim_cpu = keys_self_sim[0].cpu().numpy()[1:]

    # Step 1: Perform PCA with 1 component to compute foreground/background mask
    pca_1 = PCA(n_components=1)
    pca_1.fit(keys_self_sim_cpu)
    pca_1_values = pca_1.transform(keys_self_sim_cpu)
    pca_1_values = minmax_norm(pca_1_values)

    # Step 2: Create a mask based on the first PCA component
    threshold = np.median(pca_1_values)
    M_fg = pca_1_values.squeeze() <= threshold  # Foreground 

    # Step 3: Apply PCA on only the foreground embeddings
    pca_fg = PCA(n_components=3)
    pca_fg.fit(keys_self_sim_cpu[M_fg])
    reduced_fg = pca_fg.transform(keys_self_sim_cpu[M_fg])

    # Normalize projected foreground embeddings
    reduced_fg = minmax_norm(reduced_fg)

    # Initialize an empty array for the entire patch grid
    patch_size = vit_extractor.get_patch_size()
    patch_h_num = vit_extractor.get_height_patch_num(input_img.shape)
    patch_w_num = vit_extractor.get_width_patch_num(input_img.shape)

    h, w = patch_h_num, patch_w_num
    pca_image = np.zeros((h * w, 3))  # Initialize empty array for visualization

    # Assign only foreground embeddings to the visualization array
    pca_image[M_fg] = reduced_fg

    # Reshape into the correct format
    pca_image = pca_image.reshape(h, w, 3)
    pca_image = Image.fromarray(np.uint8(pca_image * 255))

    # Resize back to image dimensions for visualization
    pca_image = T.Resize((h * patch_size, w * patch_size), interpolation=T.InterpolationMode.BILINEAR)(pca_image)
    pca_image.save(args.save_path)
    

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--image_path", type=str, default='datasets/feature_visualization/limes.jpeg')
    parser.add_argument("--layer", type=int, default=11,
                        help='Transformer layer from which to extract the feature, between 0-11')
    parser.add_argument("--model_name", type=str, default='hf_hub:timm/vit_base_patch8_224.augreg2_in21k_ft_in1k')
    parser.add_argument("--save_path", type=str, required=True)
    args = parser.parse_args()

    visualize(args)