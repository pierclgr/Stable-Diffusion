from src.models.clip import CLIP
from src.models.vae import VAEEncoder, VAEDecoder
from src.models.diffusion import Diffusion

from src.utils import model_converter

def preload_models_from_standard_weights(ckpt_path, device):
    state_dict = model_converter.load_from_standard_weights(ckpt_path, device=device)

    encoder = VAEEncoder().to(device)
    encoder.load_state_dict(state_dict["encoder"], strict=True)

    decoder = VAEDecoder().to(device)
    decoder.load_state_dict(state_dict["decoder"], strict=True)

    diffusion = Diffusion().to(device)
    diffusion.load_state_dict(state_dict["diffusion"], strict=True)

    clip = CLIP().to(device)
    clip.load_state_dict(state_dict["clip"], strict=True)

    return {
        "clip": clip,
        "encoder": encoder,
        "decoder": decoder,
        "diffusion": diffusion
    }