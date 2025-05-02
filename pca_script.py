# %%
import numpy as np
import matplotlib.pyplot as plt
from GHA import GHA
from utils import *
np.random.seed(24)

# %%
image = load_image('images/elaine.png')
image = image.astype(np.float32) / 255.0

print("Image shape:", image.shape)
print("Image dtype:", image.dtype)
print("Image size:", image.size)
print("Image min:", image.min())
print("Image max:", image.max())
print("Image mean:", image.mean())
print("Image std:", image.std())

plt.imshow(image, cmap='gray')
plt.title(f'Original image')
plt.axis('off')
plt.show()

# %%
blocks = blockify(image, 8, 8)
flattened_blocks = blocks.reshape(-1, 8*8)
print("Flattened blocks shape:", flattened_blocks.shape)
centered_blocks, mean_vector = mean_center(flattened_blocks)
#std = np.std(flattened_blocks, axis=0)
#print("std vector shape:", std.shape)
print("mean vector shape:", mean_vector.shape)
#centered_blocks /= std
#centered_blocks

# %%
ks = (8,)
models = dict()

for k in ks:
    print(f"Number of components: {k}")
    
    gha = GHA(input_dim=64, num_components=k)
    if k > 16:
        print("Training in parallel mode")
        gha.train_parallel(centered_blocks, epochs=8000, lr_s=0.001, lr_f=0.0001)
    else:
        print("Training in sequential mode")
        gha.train_sequential(centered_blocks, epochs_per_component=600, lr_s=0.001, lr_f=0.0001)
    
    models[k] = gha

    components = gha.get_components()
    #if k <= 16:
    #    err = np.abs(components @ components.T - np.eye(k))
    #    print("Max error:", err.max())

    check_component_properties(components, verbose=False)
    print(f"Did the components converge to form a orthonormal basis: {np.allclose(components @ components.T, np.eye(k), atol=1e-4, rtol=1e-4)}")
    
    print("Principal components: ")
    visualize_components(components, 1, 8)
    #print("Components shape:", components.shape)

    encoded_blocks = encode_blocks(flattened_blocks, mean_vector, components)
    print("Component coefficients:")
    visualize_component_coefficients(encoded_blocks, num_components=8)
    #print(f"Encoded shape: {encoded_blocks.shape}") # (1024,k)

    reconstructed_image = reconstruct_image(encoded_blocks, mean_vector, components)

    #print("Original image dtype:", image.dtype)
    #print("Original min/max:", image.min(), image.max())
    #print("Original image shape:", image.shape)

    #print("Reconstructed dtype:", reconstructed_image.dtype)
    #print("Reconstructed min/max:", reconstructed_image.min(), reconstructed_image.max())
    #print("Reconstructed image shape:", reconstructed_image.shape)


    print(f"MSE: {np.mean((image - reconstructed_image) ** 2)}")
    show_original_vs_reconstructed(image, reconstructed_image, k)

# %%
reconstruct_with_first_k_components(flattened_blocks, mean_vector, components, first_k_components=list(range(1, 9)))

# %%
plot_pca_eigenvalues(centered_blocks)


# %%
# @title Generalization on other images
k=8
components = models[k].get_components()

other_images = ['images/1.jpg', 'images/2.jpg', 'images/3.jpg']

for path in other_images:
    img = load_image(path).astype(np.float32) / 255.0
    print(img.shape)
    blocks = blockify(img, 8, 8)
    flat_blocks = blocks.reshape(-1, 64)
    codes = encode_blocks(flat_blocks, mean_vector, components)
    recon = reconstruct_image(codes, mean_vector, components)
    show_original_vs_reconstructed(img, recon, k)
    print(f"{path} MSE: {np.mean((img - recon)**2)}")


