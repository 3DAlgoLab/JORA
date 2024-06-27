import kagglehub

# Download latest version
path = kagglehub.model_download("google/paligemma/jax/paligemma-3b-mix-224")

print("Path to model files:", path)
