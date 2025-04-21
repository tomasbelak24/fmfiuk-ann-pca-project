import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from GHA import GHA

def sequential_gha(X, num_components=16, learning_rate=0.001, max_epochs=200):
    """
    Sequentially extract num_components using 1-neuron GHA and deflation.
    """
    d = X.shape[1]
    components = []

    residual = X.copy()

    for i in range(num_components):
        print(f"🔹 Training component {i+1}/{num_components}")
        
        # Train 1-neuron GHA
        gha = GHA(
            input_dim=d,
            num_components=1,
            learning_rate=learning_rate,
        )
        gha.train(residual, epochs=max_epochs)
        w = gha.get_components()[0]  # shape (d,)

        # Normalize (optional but recommended)
        w = w / np.linalg.norm(w)
        components.append(w)

        # Project w out of the residual
        projections = residual @ w  # shape (n_samples,)
        residual -= np.outer(projections, w)  # deflation step

    return np.array(components)  # shape (k, d)

def load_image(image_path):
    image = Image.open(image_path)
    return np.array(image)


def blockify(image, h, w):
    H, W = image.shape[:2]
    assert H % h == 0 and W % w == 0, "Image dimensions must be divisible by block size"

    num_blocks_y = H // h
    num_blocks_x = W // w

    # Reshape and swap axes to extract blocks
    blocks = image.reshape(num_blocks_y, h, num_blocks_x, w).swapaxes(1, 2)
    
    return blocks


def mean_center(X):
    mean_vector = np.mean(X, axis=0)  # shape: (64,)
    centered_X = X - mean_vector
    return centered_X, mean_vector


def visualize_components(components, num_rows=2, num_cols=4):
    plt.figure(figsize=(2.5 * num_cols, 2.5 * num_rows))
    for i in range(num_rows * num_cols):
        comp = components[i].reshape(8, 8)
        plt.subplot(num_rows, num_cols, i + 1)
        plt.imshow(comp, cmap='gray')
        plt.title(f'PC {i+1}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()


def encode_blocks(blocks, mean_vector, components):
    centered = blocks - mean_vector  # shape: (1024, 64)
    codes = centered @ components.T  # shape: (1024, k)
    return codes


def show_original_vs_reconstructed(original, reconstructed, k):

    plt.figure(figsize=(10, 5))

    # Original
    plt.subplot(1, 2, 1)
    plt.imshow(original, cmap='gray')
    plt.title("Original Image")
    plt.axis('off')

    # Reconstructed
    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed, cmap='gray')
    plt.title(f"Reconstructed (k={k})")
    plt.axis('off')

    plt.tight_layout()
    plt.show()


def check_component_properties(components, atol=1e-2):
    k, d = components.shape
    print(f"Checking {k} components of dimension {d}")

    # Normalize components to compute dot products cleanly
    norms = np.linalg.norm(components, axis=1)
    print("Vector norms (should be ~1):")
    for i, norm in enumerate(norms):
        print(f"  Component {i+1}: norm = {norm:.4f}")
    
    # Check orthogonality (dot product between pairs)
    print("\nOrthogonality check (dot products between pairs):")
    all_orthogonal = True
    for i in range(k):
        for j in range(i + 1, k):
            dot_product = np.dot(components[i], components[j])
            print(f"  Dot(PC{i+1}, PC{j+1}) = {dot_product:.4f}")
            if not np.isclose(dot_product, 0.0, atol=atol):
                all_orthogonal = False

    if all_orthogonal:
        print("\n✅ All components are approximately orthogonal.")
    else:
        print("\n⚠️ Some components are not perfectly orthogonal.")


def reconstruct_image(encoded_blocks, mean_vector, components):
    # Reconstruct the blocks from the encoded data
    reconstructed_blocks = reconstruct_blocks(encoded_blocks, mean_vector, components)

    # Reshape to (32, 32, 8, 8)
    blocks_4d = reshape_blocks(reconstructed_blocks)

    # Convert to image format (256, 256)
    image = blocks_to_image(blocks_4d)

    return image

def reconstruct_blocks(encoded_blocks, mean_vector, components):
    # Reconstruct centered data
    reconstructed_centered = encoded_blocks @ components  # shape: (1024, 64)

    # Add the mean back
    reconstructed = reconstructed_centered + mean_vector  # broadcast: (1024, 64)

    return reconstructed

def reshape_blocks(flat_blocks):
    # Reshape from (1024, 64) → (32, 32, 8, 8)
    return flat_blocks.reshape(32, 32, 8, 8)


def blocks_to_image(blocks_4d):
    # From (32, 32, 8, 8) to (256, 256)
    return blocks_4d.transpose(0, 2, 1, 3).reshape(256, 256)