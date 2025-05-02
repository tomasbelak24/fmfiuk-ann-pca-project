import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def load_image(image_path):
    image = Image.open(image_path).convert('L')
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


def check_component_properties(components, atol=1e-2, verbose=True):
    k, d = components.shape
    print(f"Checking {k} components of dimension {d}")

    # Normalize components to compute dot products cleanly
    norms = np.linalg.norm(components, axis=1)
    not_unit_lenght = 0
    for i, norm in enumerate(norms):
        if verbose:
            print("Vector norms (should be ~1):")
            print(f"  Component {i+1}: norm = {norm:.4f}")
        if not np.isclose(norm, 1.0, atol=atol):
            not_unit_lenght += 1
    if not_unit_lenght > 0:
        print(f"\n⚠️ {not_unit_lenght} components are not unit length.")
    else:
        print("\n✅ All components are unit length.")
            
    # Check orthogonality (dot product between pairs)
    all_orthogonal = True
    for i in range(k):
        for j in range(i + 1, k):
            dot_product = np.dot(components[i], components[j])
            if verbose:
                print("\nOrthogonality check (dot products between pairs):")
                print(f"  Dot(PC{i+1}, PC{j+1}) = {dot_product:.4f}")
            if not np.isclose(dot_product, 0.0, atol=atol):
                all_orthogonal = False
                break
        
        if not all_orthogonal:
            break

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

    image = np.clip(image, 0, 1)

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


def plot_pca_eigenvalues(centered_data):
    cov_matrix = centered_data.T @ centered_data / centered_data.shape[0]
    eigenvalues, _ = np.linalg.eigh(cov_matrix)
    eigenvalues = np.flip(np.sort(eigenvalues))

    plt.figure(figsize=(10, 4))
    plt.plot(eigenvalues, marker='o')
    plt.title("Eigenvalues of the PCA Correlation Matrix")
    plt.xlabel("Principal Component Index")
    plt.ylabel("Eigenvalue")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def visualize_component_coefficients(encoded_blocks, num_components=8, block_grid_shape=(32, 32)):
    
    plt.figure(figsize=(2 * num_components, 2))

    for i in range(num_components):
        coeffs = encoded_blocks[:, i].reshape(block_grid_shape)
        plt.subplot(1, num_components, i + 1)
        plt.imshow(coeffs, cmap='gray')
        plt.axis('off')
        plt.title(f"C{i+1}")

    plt.tight_layout()
    plt.show()