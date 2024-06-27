# Pseudo code to understand JORA trainer
import os
import jax
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

import sentencepiece as spm

from jora.lib.dataloader import LlamaDataLoader
from jora.lib.gemma.gemma_utils import GemmaTokenizer
from gemma import params as params_lib

cpu_device = ...


def train_lora(
    config,
    train_dataset,
):

    vocab_path = os.path.join(config.GEMMA_MODEL_PATH, "tokenizer.model")
    vocab = spm.SentencePieceProcessor()
    vocab.Load(vocab_path)

    tokenizer = GemmaTokenizer(vocab)

    collate_fn = ...

    dataloader = LlamaDataLoader(dataset, collate_fn, ...)

    with jax.default_device(cpu_device):
        ckpt_path = "..."
        params = params_lib.load_and_format(ckpt_path)  # google mind gemma code

    # Sharding
    num_gpus = jax.device_count("gpu")
    devices = mesh_utils.create_device_mesh((num_gpus,))
    default_mesh = Mesh(devices, axis_names=("p",))

    # Applying LORA
    # init. lora A to be sampled from a normal distribution
