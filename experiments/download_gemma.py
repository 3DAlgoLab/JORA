import kagglehub

VARIANT = "2b-it"  # @param ['2b', '2b-it', '7b', '7b-it', '1.1-2b-it', '1.1-7b-it'] {type:"string"}
weights_dir = kagglehub.model_download(f"google/gemma/Flax/{VARIANT}")
