import torch
import numpy as np
from tqdm import tqdm
from ddpm import DDPMScheduler

WIDTH = 512
HEIGHT = 512
LATENTS_WIDTH = 512 / 8
LATENTS_HEIGHT = 512 / 8


def generate(prompt: str, negative_prompt: str = "", input_image=None, strength=0.8, do_cfg: bool = True,
             cfg_scale: float = 7.5, sampler_name: str = "ddpm", n_inference_steps: int = 50, models=None, seed=None,
             device=None, idle_device=None, tokenizer=None):
    if models is None:
        models = {}

    with torch.no_grad():
        if not (0 < strength <= 1):
            raise ValueError("Strength must be between 0 and 1!")

        if idle_device:
            to_idle: lambda x: x.to(idle_device)
        else:
            to_idle = None

    generator = torch.Generator(device=device)
    if not seed:
        generator.seed()
    else:
        generator.manual_seed(seed)

    clip = models["clip"]
    clip.to(device)

    # Convert the prompt into tokens using the tokenizer
    prompt_tokens = tokenizer.batch_encode_plus([prompt], padding="max_length", max_length=77).input_ids
    prompt_tokens = torch.tensor(prompt_tokens, dtype=torch.long, device=device)
    prompt_context = clip(prompt_tokens)
    context = prompt_context

    if do_cfg:
        # Convert negative prompt into tokens using the tokenizer
        negative_prompt_tokens = tokenizer.batch_encode_plus([negative_prompt], padding="max_length",
                                                             max_length=77).input_ids
        negative_prompt_tokens = torch.tensor(negative_prompt_tokens, dtype=torch.long, device=device)
        negative_prompt_context = clip(negative_prompt_tokens)

        context = torch.cat([context, negative_prompt_context])
    if to_idle:
        to_idle(clip)

    if sampler_name == "ddpm":
        sampler = DDPMScheduler(generator)
        sampler.set_inference_timesteps(n_inference_steps)
    else:
        raise ValueError(f"Sampler {sampler_name} not supported!")

    latent_shape = (1, 4, LATENTS_HEIGHT, LATENTS_WIDTH)

    noise = torch.randn(latent_shape, generator=generator, device=device)
    latents = noise
    if input_image:
        encoder = models["encoder"]
        encoder.to(device)

        input_image_tensor = input_image.resize((WIDTH, HEIGHT))
        input_image_tensor = np.array(input_image_tensor)
        input_image_tensor = torch.tensor(input_image_tensor, dtype=torch.float32, device=device)
        input_image_tensor = rescale(input_image_tensor, (0, 2255), (-1, 1))
        input_image_tensor = input_image_tensor.unsqueeze(0)
        input_image_tensor = input_image_tensor.permute(0, 3, 1, 2)

        # feed the image to the decoder of the VAE to produce latents
        latents = encoder(input_image_tensor, latents)

        sampler.set_strength(strength=strength)
        latents = sampler.add_noise(latents, sampler.timesteps[0])

        if to_idle:
            to_idle(encoder)

    diffusion = models["diffusion"]
    diffusion.to(device)

    timesteps = tqdm(sampler.timesteps)
    for i, timestep in enumerate(timesteps):
        time_embedding = get_time_embedding(timestep).to(device=device)

        model_input = latents

        if do_cfg:
            model_input = model_input.repeat(2, 1, 1, 1)

        # compute model output, the noise predicted by the UNet
        model_output = diffusion(model_input, context, time_embedding)

        if do_cfg:
            output_cond, output_uncond = model_output.chunk(2)
            model_output = cfg_scale * (output_cond - output_uncond) + output_uncond

        # remove noise predicted by UNet
        latents = sampler.step(timestep, latents, model_output)

    if to_idle:
        to_idle(diffusion)

    decoder = models["decoder"]
    decoder.to(device)

    images = decoder(latents)
    if to_idle:
        to_idle(decoder)

    images = rescale(images, (-1, 1), (0, 255), clamp=True)
    images = images.permute(0, 2, 3, 1)
    images = images.to("cpu", torch.uint8).numpy()
    return imags[0]


def rescale(x, old_range, new_range, clamp=False):
    old_min, old_max = old_range
    new_min, new_max = new_range
    x -= old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min
    if clamp:
        x = x.clamp(new_min, new_max)
    return x


def get_time_embedding(timestep: int):
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160)
    x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)
