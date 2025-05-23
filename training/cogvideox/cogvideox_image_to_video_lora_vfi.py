# Copyright 2024 The HuggingFace Team.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gc
import logging
import math
import os
import random
import shutil
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict

import diffusers
import torch
import transformers
import wandb
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import (
    DistributedDataParallelKwargs,
    InitProcessGroupKwargs,
    ProjectConfiguration,
    set_seed,
)
from diffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXDPMScheduler,
    # CogVideoXImageToVideoPipeline,
    CogVideoXTransformer3DModel,
)
#
from pipeline_cogvideox_image2video_interpolation import CogVideoXInterpolationPipeline
#
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params
from diffusers.utils import convert_unet_state_dict_to_peft, export_to_video, load_image
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from huggingface_hub import create_repo, upload_folder
from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, T5EncoderModel
from einops import repeat
from PIL import Image
import numpy as np
import datetime
import json

from args import get_args  # isort:skip
from dataset_vfi import BucketSampler, VideoDatasetWithResizing, VideoDatasetWithResizeAndRectangleCrop  # isort:skip
from text_encoder import compute_prompt_embeddings  # isort:skip
from utils import (
    get_gradient_norm,
    get_optimizer,
    prepare_rotary_positional_embeddings,
    print_memory,
    reset_memory,
    unwrap_model,
)


logger = get_logger(__name__)


def compress_masks(masks, group_size=4):
    """
    Compress the frame dimension by performing a logical AND across groups of frames.
    
    The input tensor is assumed to have shape [b, f, c, h, w]. The first frame is kept
    unchanged, and the remaining frames are grouped in chunks of `group_size`. For each
    group, the output pixel is 1.0 if all frames in the group are 1.0, and 0.0 otherwise.
    
    Args:
        masks (torch.Tensor): Input tensor of shape [b, f, c, h, w] with values 0.0 or 1.0.
        group_size (int): Number of frames to group for the AND operation (default: 4).
    
    Returns:
        torch.Tensor: Compressed tensor with shape [b, 1 + ceil((f-1)/group_size), c, h, w].
    """
    b, f, c, h, w = masks.shape
    
    # The first frame remains unchanged.
    first_frame = masks[:, 0:1, :, :, :]  # shape: [b, 1, c, h, w]
    
    grouped_frames = []
    # Process frames starting from index 1 in groups of `group_size`
    for i in range(1, f, group_size):
        # Select a group; note that the last group might have fewer than `group_size` frames.
        group = masks[:, i:i+group_size, :, :, :]  # shape: [b, group_size, c, h, w]
        # Perform logical AND over the group.
        # Convert to boolean, compute "all" along the group dimension, then convert back to float.
        group_and = (group == 1.0).all(dim=1, keepdim=True).float()  # shape: [b, 1, c, h, w]
        grouped_frames.append(group_and)
    
    # Concatenate the first frame with the ANDed groups along the frame dimension.
    compressed = torch.cat([first_frame] + grouped_frames, dim=1)
    return compressed


def export_to_gif(frames, output_gif_path, fps):
    """
    Export a list of frames to a GIF.

    Args:
    - frames (list): List of frames (as numpy arrays or PIL Image objects).
    - output_gif_path (str): Path to save the output GIF.
    - fps (int): fps

    """
    # Convert numpy arrays to PIL Images if needed
    pil_frames = [Image.fromarray(frame) if isinstance(
        frame, np.ndarray) else frame for frame in frames]

    pil_frames[0].save(output_gif_path.replace('.mp4', '.gif'),
                       format='GIF',
                       append_images=pil_frames[1:],
                       save_all=True,
                       duration=int(1000/fps),
                       loop=0)


def create_grid_gif(columns, spacing=5, fps=7, loop=0, output_path="grid.gif"):
    """
    Create a grid GIF (1 row, N columns) from an arbitrary number of lists of PIL images.
    
    Each argument in *columns is expected to be a list of PIL.Image objects representing a column.
    All lists must have the same number of images, and all images should have the same dimensions.
    
    Parameters:
        *columns: Variable number of lists, each containing PIL.Image objects.
        spacing (int): Spacing in pixels between columns (default: 10).
        fps (int): Frames per second for the GIF (default: 10).
        loop (int): Number of loops (0 for infinite looping; default: 0).
        output_path (str): File path for the output GIF (default: "grid.gif").
    """
    if not columns:
        return 0
    
    num_columns = len(columns)
    num_frames = len(columns[0])
    
    # Ensure all columns have the same number of frames.
    for col in columns:
        if len(col) != num_frames:
            raise ValueError("All columns must contain the same number of images.")
    
    # Assume all images have the same dimensions. Use the first image from the first column.
    first_image = columns[0][0]
    img_width, img_height = first_image.size
    
    # Calculate total width: sum of column widths + spacing between columns.
    total_width = num_columns * img_width + (num_columns - 1) * spacing
    frames = []
    
    # Build each frame of the grid GIF.
    for i in range(num_frames):
        # Create a new blank image for the current frame.
        grid_img = Image.new('RGB', (total_width, img_height), color=(255, 255, 255))
        x_offset = 0
        for col in columns:
            grid_img.paste(col[i], (x_offset, 0))
            x_offset += img_width + spacing
        frames.append(grid_img)
    
    # Calculate frame duration in milliseconds from fps.
    frame_duration = int(1000 / fps)
    
    # Save the frames as an animated GIF.
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=frame_duration,
        loop=loop
    )
    print(f"grid GIF saved to {output_path}")


def get_time_string() -> str:
    x = datetime.datetime.now()
    return f"{x.month:02d}{x.day:02d}-{x.hour:02d}{x.minute:02d}"


def save_model_card(
    repo_id: str,
    videos=None,
    base_model: str = None,
    validation_prompt=None,
    repo_folder=None,
    fps=8,
):
    widget_dict = []
    if videos is not None:
        for i, video in enumerate(videos):
            export_to_video(video, os.path.join(repo_folder, f"final_video_{i}.mp4", fps=fps))
            widget_dict.append(
                {
                    "text": validation_prompt if validation_prompt else " ",
                    "output": {"url": f"video_{i}.mp4"},
                }
            )

    model_description = f"""
# CogVideoX LoRA Finetune

<Gallery />

## Model description

This is a lora finetune of the CogVideoX model `{base_model}`.

The model was trained using [CogVideoX Factory](https://github.com/a-r-r-o-w/cogvideox-factory) - a repository containing memory-optimized training scripts for the CogVideoX family of models using [TorchAO](https://github.com/pytorch/ao) and [DeepSpeed](https://github.com/microsoft/DeepSpeed). The scripts were adopted from [CogVideoX Diffusers trainer](https://github.com/huggingface/diffusers/blob/main/examples/cogvideo/train_cogvideox_lora.py).

## Download model

[Download LoRA]({repo_id}/tree/main) in the Files & Versions tab.

## Usage

Requires the [🧨 Diffusers library](https://github.com/huggingface/diffusers) installed.

```py
import torch
from diffusers import CogVideoXImageToVideoPipeline
from diffusers.utils import export_to_video, load_image

pipe = CogVideoXImageToVideoPipeline.from_pretrained("THUDM/CogVideoX-5b-I2V", torch_dtype=torch.bfloat16).to("cuda")
pipe.load_lora_weights("{repo_id}", weight_name="pytorch_lora_weights.safetensors", adapter_name="cogvideox-lora")

# The LoRA adapter weights are determined by what was used for training.
# In this case, we assume `--lora_alpha` is 32 and `--rank` is 64.
# It can be made lower or higher from what was used in training to decrease or amplify the effect
# of the LoRA upto a tolerance, beyond which one might notice no effect at all or overflows.
pipe.set_adapters(["cogvideox-lora"], [32 / 64])

image = load_image("/path/to/image.png")
video = pipe(image=image, prompt="{validation_prompt}", guidance_scale=6, use_dynamic_cfg=True).frames[0]
export_to_video(video, "output.mp4", fps=8)
```

For more details, including weighting, merging and fusing LoRAs, check the [documentation](https://huggingface.co/docs/diffusers/main/en/using-diffusers/loading_adapters) on loading LoRAs in diffusers.

## License

Please adhere to the licensing terms as described [here](https://huggingface.co/THUDM/CogVideoX-5b-I2V/blob/main/LICENSE).
"""
    model_card = load_or_create_model_card(
        repo_id_or_path=repo_id,
        from_training=True,
        license="other",
        base_model=base_model,
        prompt=validation_prompt,
        model_description=model_description,
        widget=widget_dict,
    )
    tags = [
        "text-to-video",
        "image-to-video",
        "diffusers-training",
        "diffusers",
        "lora",
        "cogvideox",
        "cogvideox-diffusers",
        "template:sd-lora",
    ]

    model_card = populate_model_card(model_card, tags=tags)
    model_card.save(os.path.join(repo_folder, "README.md"))


def log_validation(
    accelerator: Accelerator,
    pipe: CogVideoXInterpolationPipeline,
    args: Dict[str, Any],
    pipeline_args: Dict[str, Any],
    is_final_validation: bool = False,
    image_id: str = "_",
    global_step = None,
):
    logger.info(
        f"Running validation... \n Generating {args.num_validation_videos} videos with prompt: {image_id}."
    )

    pipe = pipe.to(accelerator.device)

    # run inference
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None

    videos = []
    for _ in range(args.num_validation_videos):
        # video = pipe(**pipeline_args, generator=generator, output_type="np").frames[0]
        video = pipe(**pipeline_args, generator=generator)[0][0]
        videos.append(video)

    for tracker in accelerator.trackers:
        phase_name = "test" if is_final_validation else "validation"
        if tracker.name == "wandb":
            video_filenames = []
            for i, video in enumerate(videos):
                save_dir = os.path.join(args.output_dir, "validation", f"step_{global_step}")
                os.makedirs(save_dir, exist_ok=True)
                filename = os.path.join(save_dir, f"{image_id}.mp4")
                # [mp4]
                export_to_video(video, filename, fps=7)
                # [gif]
                export_to_gif(video, filename, fps=7)
                video_filenames.append(filename)

            tracker.log(
                {
                    phase_name: [
                        wandb.Video(filename, caption=f"{i}: {image_id}")
                        for i, filename in enumerate(video_filenames)
                    ]
                }
            )

    return videos


def run_validation(
    args: Dict[str, Any],
    accelerator: Accelerator,
    transformer,
    scheduler,
    model_config: Dict[str, Any],
    weight_dtype: torch.dtype,
    global_step=None,
) -> None:
    assert global_step is not None
    accelerator.print("===== Memory before validation =====")
    print_memory(accelerator.device)
    torch.cuda.synchronize(accelerator.device)

    pipe = CogVideoXInterpolationPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        transformer=unwrap_model(accelerator, transformer),
        scheduler=scheduler,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
    )

    if args.enable_slicing:
        pipe.vae.enable_slicing()
    if args.enable_tiling:
        pipe.vae.enable_tiling()
    if args.enable_model_cpu_offload:
        pipe.enable_model_cpu_offload()

    # validation_prompt = args.id_token + " " + args.validation_prompt
    validation_dir = os.path.join(args.data_root, args.validation_dir)
    validation_videos_dir = [os.path.join(validation_dir, videoname) for videoname in os.listdir(validation_dir)]

    # columns_for_grid = []
    for validation_videos_dir_i in validation_videos_dir:
        validation_video_frames = [
            os.path.join(validation_videos_dir_i, 'video', filename) for filename in os.listdir(os.path.join(validation_videos_dir_i, 'video'))
        ]
        validation_video_frames = sorted(validation_video_frames)
        validation_image_first, validation_image_last = validation_video_frames[0], validation_video_frames[-1]

        with open(os.path.join(validation_videos_dir_i, 'prompt.txt'), 'r') as f:
            validation_prompt = f.read()
            validation_prompt = validation_prompt.strip()

        pipeline_args = {
            "first_image": load_image(validation_image_first),
            "last_image": load_image(validation_image_last),
            "prompt": validation_prompt,
            "guidance_scale": args.guidance_scale,
            "use_dynamic_cfg": args.use_dynamic_cfg,
            "height": args.height,
            "width": args.width,
            "max_sequence_length": model_config.max_text_seq_length,
        }

        column = log_validation(
            pipe=pipe,
            args=args,
            accelerator=accelerator,
            pipeline_args=pipeline_args,
            image_id=os.path.basename(validation_videos_dir_i),
            global_step=global_step,
        )[0]
        # columns_for_grid.append(column)

    save_dir = os.path.join(args.output_dir, "validation", f"step_{global_step}")
    os.makedirs(save_dir, exist_ok=True)
    # filename = os.path.join(save_dir, "grid.gif")
    # create_grid_gif(columns_for_grid, spacing=5, fps=7, loop=0, output_path=filename)


    accelerator.print("===== Memory after validation =====")
    print_memory(accelerator.device)
    reset_memory(accelerator.device)

    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(accelerator.device)


class CollateFunction:
    def __init__(self, weight_dtype: torch.dtype, load_tensors: bool) -> None:
        self.weight_dtype = weight_dtype
        self.load_tensors = load_tensors

    def __call__(self, data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        prompts = [x["prompt"] for x in data[0]]

        if self.load_tensors:
            prompts = torch.stack(prompts).to(dtype=self.weight_dtype, non_blocking=True)

        first_images = [x["first_image"] for x in data[0]]
        first_images = torch.stack(first_images).to(dtype=self.weight_dtype, non_blocking=True)

        last_images = [x["last_image"] for x in data[0]]
        last_images = torch.stack(last_images).to(dtype=self.weight_dtype, non_blocking=True)

        videos = [x["video"] for x in data[0]]
        videos = torch.stack(videos).to(dtype=self.weight_dtype, non_blocking=True)

        return {
            "first_images": first_images,
            "last_images": last_images,
            "videos": videos,
            "prompts": prompts
        }


def main(args):
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    args.output_dir = os.path.join(args.output_dir, get_time_string())
    os.makedirs(args.output_dir, exist_ok=True)
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    init_process_group_kwargs = InitProcessGroupKwargs(backend="nccl", timeout=timedelta(seconds=args.nccl_timeout))
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs, init_process_group_kwargs],
    )

    ## save config ##
    config_save_path = os.path.join(args.output_dir, "training_config.json")
    args.num_processes = accelerator.num_processes
    with open(config_save_path, "w") as f:
        json.dump(vars(args), f, indent=4)
    print(f"Configuration saved at {config_save_path}")

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub and False:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name,
                exist_ok=True,
            ).repo_id

    # Prepare models and scheduler
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
    )

    text_encoder = T5EncoderModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
    )

    # CogVideoX-2b weights are stored in float16
    # CogVideoX-5b and CogVideoX-5b-I2V weights are stored in bfloat16
    load_dtype = torch.bfloat16 if "5b" in args.pretrained_model_name_or_path.lower() else torch.float16
    transformer = CogVideoXTransformer3DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        torch_dtype=load_dtype,
        revision=args.revision,
        variant=args.variant,
    )

    # These changes will also be required when trying to run inference with the trained lora
    if args.ignore_learned_positional_embeddings:   ## ignore_learned_positional_embeddings = False ##
        del transformer.patch_embed.pos_embedding
        transformer.patch_embed.use_learned_positional_embeddings = False
        transformer.config.use_learned_positional_embeddings = False

    vae = AutoencoderKLCogVideoX.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
    )

    scheduler = CogVideoXDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    if args.enable_slicing:
        vae.enable_slicing()
    if args.enable_tiling:
        vae.enable_tiling()

    # We only train the additional adapter LoRA layers
    text_encoder.requires_grad_(False)
    transformer.requires_grad_(False)
    vae.requires_grad_(False)

    VAE_SCALING_FACTOR = vae.config.scaling_factor
    VAE_SCALE_FACTOR_SPATIAL = 2 ** (len(vae.config.block_out_channels) - 1)
    RoPE_BASE_HEIGHT = transformer.config.sample_height * VAE_SCALE_FACTOR_SPATIAL
    RoPE_BASE_WIDTH = transformer.config.sample_width * VAE_SCALE_FACTOR_SPATIAL

    # For mixed precision training we cast all non-trainable weights (vae, text_encoder and transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.state.deepspeed_plugin:
        # DeepSpeed is handling precision, use what's in the DeepSpeed config
        if (
            "fp16" in accelerator.state.deepspeed_plugin.deepspeed_config
            and accelerator.state.deepspeed_plugin.deepspeed_config["fp16"]["enabled"]
        ):
            weight_dtype = torch.float16
        if (
            "bf16" in accelerator.state.deepspeed_plugin.deepspeed_config
            and accelerator.state.deepspeed_plugin.deepspeed_config["bf16"]["enabled"]
        ):
            weight_dtype = torch.bfloat16
    else:
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

    if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    text_encoder.to(accelerator.device, dtype=weight_dtype)
    transformer.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    # now we will add new LoRA weights to the attention layers
    transformer_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.lora_alpha,
        init_lora_weights=True,
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    transformer.add_adapter(transformer_lora_config)

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            transformer_lora_layers_to_save = None

            for model in models:
                if isinstance(unwrap_model(accelerator, model), type(unwrap_model(accelerator, transformer))):
                    model = unwrap_model(accelerator, model)
                    transformer_lora_layers_to_save = get_peft_model_state_dict(model)
                else:
                    raise ValueError(f"Unexpected save model: {model.__class__}")

                # make sure to pop weight so that corresponding model is not saved again
                if weights:
                    weights.pop()

            CogVideoXInterpolationPipeline.save_lora_weights(
                output_dir,
                transformer_lora_layers=transformer_lora_layers_to_save,
            )

    def load_model_hook(models, input_dir):
        transformer_ = None

        # This is a bit of a hack but I don't know any other solution.
        if not accelerator.distributed_type == DistributedType.DEEPSPEED:
            while len(models) > 0:
                model = models.pop()

                if isinstance(unwrap_model(accelerator, model), type(unwrap_model(accelerator, transformer))):
                    transformer_ = unwrap_model(accelerator, model)
                else:
                    raise ValueError(f"Unexpected save model: {unwrap_model(accelerator, model).__class__}")
        else:
            transformer_ = CogVideoXTransformer3DModel.from_pretrained(
                args.pretrained_model_name_or_path, subfolder="transformer"
            )
            transformer_.add_adapter(transformer_lora_config)

        lora_state_dict = CogVideoXInterpolationPipeline.lora_state_dict(input_dir)

        transformer_state_dict = {
            f'{k.replace("transformer.", "")}': v for k, v in lora_state_dict.items() if k.startswith("transformer.")
        }
        transformer_state_dict = convert_unet_state_dict_to_peft(transformer_state_dict)
        incompatible_keys = set_peft_model_state_dict(transformer_, transformer_state_dict, adapter_name="default")
        if incompatible_keys is not None:
            # check only for unexpected keys
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            if unexpected_keys:
                logger.warning(
                    f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                    f" {unexpected_keys}. "
                )

        # Make sure the trainable params are in float32. This is again needed since the base models
        # are in `weight_dtype`. More details:
        # https://github.com/huggingface/diffusers/pull/6514#discussion_r1449796804
        if args.mixed_precision == "fp16":
            # only upcast trainable parameters (LoRA) into fp32
            cast_training_params([transformer_])

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Make sure the trainable params are in float32.
    if args.mixed_precision == "fp16":
        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params([transformer], dtype=torch.float32)

    transformer_lora_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))

    # Optimization parameters
    transformer_parameters_with_lr = {
        "params": transformer_lora_parameters,
        "lr": args.learning_rate,
    }
    params_to_optimize = [transformer_parameters_with_lr]
    num_trainable_parameters = sum(param.numel() for model in params_to_optimize for param in model["params"])

    use_deepspeed_optimizer = (
        accelerator.state.deepspeed_plugin is not None
        and "optimizer" in accelerator.state.deepspeed_plugin.deepspeed_config
    )
    use_deepspeed_scheduler = (
        accelerator.state.deepspeed_plugin is not None
        and "scheduler" in accelerator.state.deepspeed_plugin.deepspeed_config
    )

    optimizer = get_optimizer(
        params_to_optimize=params_to_optimize,
        optimizer_name=args.optimizer,
        learning_rate=args.learning_rate,
        beta1=args.beta1,
        beta2=args.beta2,
        beta3=args.beta3,
        epsilon=args.epsilon,
        weight_decay=args.weight_decay,
        prodigy_decouple=args.prodigy_decouple,
        prodigy_use_bias_correction=args.prodigy_use_bias_correction,
        prodigy_safeguard_warmup=args.prodigy_safeguard_warmup,
        use_8bit=args.use_8bit,
        use_4bit=args.use_4bit,
        use_torchao=args.use_torchao,
        use_deepspeed=use_deepspeed_optimizer,
        use_cpu_offload_optimizer=args.use_cpu_offload_optimizer,
        offload_gradients=args.offload_gradients,
    )

    # Dataset and DataLoader
    dataset_init_kwargs = {
        "data_root": args.data_root,
        "dataset_file": args.dataset_file,
        "caption_column": args.caption_column,
        "video_column": args.video_column,
        "max_num_frames": args.max_num_frames,
        "id_token": args.id_token,
        "height_buckets": args.height_buckets,
        "width_buckets": args.width_buckets,
        "frame_buckets": args.frame_buckets,
        "load_tensors": args.load_tensors,  ## False
        "random_flip": False, #args.random_flip,
        "image_to_video": True,
    }
    if args.video_reshape_mode is None: ## this branch ##
        train_dataset = VideoDatasetWithResizing(**dataset_init_kwargs)
    else:
        train_dataset = VideoDatasetWithResizeAndRectangleCrop(
            video_reshape_mode=args.video_reshape_mode, **dataset_init_kwargs
        )

    collate_fn = CollateFunction(weight_dtype, args.load_tensors)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=1,
        sampler=BucketSampler(train_dataset, batch_size=args.train_batch_size, shuffle=True),
        collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers,
        pin_memory=args.pin_memory,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    if args.use_cpu_offload_optimizer:
        lr_scheduler = None
        accelerator.print(
            "CPU Offload Optimizer cannot be used with DeepSpeed or builtin PyTorch LR Schedulers. If "
            "you are training with those settings, they will be ignored."
        )
    else:
        if use_deepspeed_scheduler:
            from accelerate.utils import DummyScheduler

            lr_scheduler = DummyScheduler(
                name=args.lr_scheduler,
                optimizer=optimizer,
                total_num_steps=args.max_train_steps * accelerator.num_processes,
                num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
            )
        else:
            lr_scheduler = get_scheduler(
                args.lr_scheduler,
                optimizer=optimizer,
                num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
                num_training_steps=args.max_train_steps * accelerator.num_processes,
                num_cycles=args.lr_num_cycles,
                power=args.lr_power,
            )

    # Prepare everything with our `accelerator`.
    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.distributed_type == DistributedType.DEEPSPEED or accelerator.is_main_process:
        tracker_name = args.tracker_name or "cogvideox-lora"
        accelerator.init_trackers(tracker_name, config=vars(args))

        accelerator.print("===== Memory before training =====")
        reset_memory(accelerator.device)
        print_memory(accelerator.device)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    accelerator.print("***** Running training *****")
    accelerator.print(f"  Num trainable parameters = {num_trainable_parameters}")
    accelerator.print(f"  Num examples = {len(train_dataset)}")
    accelerator.print(f"  Num batches each epoch = {len(train_dataloader)}")
    accelerator.print(f"  Num epochs = {args.num_train_epochs}")
    accelerator.print(f"  Instantaneous batch size per device = {args.train_batch_size}")
    accelerator.print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    accelerator.print(f"  Gradient accumulation steps = {args.gradient_accumulation_steps}")
    accelerator.print(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if not args.resume_from_checkpoint:
        initial_global_step = 0
    else:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    # For DeepSpeed training
    model_config = transformer.module.config if hasattr(transformer, "module") else transformer.config

    if args.load_tensors:
        del vae, text_encoder
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(accelerator.device)

    alphas_cumprod = scheduler.alphas_cumprod.to(accelerator.device, dtype=torch.float32)

    for epoch in range(first_epoch, args.num_train_epochs):
        transformer.train()

        for step, batch in enumerate(train_dataloader):
            models_to_accumulate = [transformer]
            logs = {}

            with accelerator.accumulate(models_to_accumulate):

                first_images = batch["first_images"].to(accelerator.device, non_blocking=True)
                last_images = batch["last_images"].to(accelerator.device, non_blocking=True)
                videos = batch["videos"].to(accelerator.device, non_blocking=True)
                prompts = batch["prompts"]

                # Encode videos
                if not args.load_tensors:   ## this branch ( args.load_tensors = False ) ##
                    
                    first_images = first_images.permute(0, 2, 1, 3, 4)  # [B, C, F, H, W]
                    first_image_noise_sigma = torch.normal(
                        mean=-3.0, std=0.5, size=(first_images.size(0),), device=accelerator.device, dtype=weight_dtype
                    )
                    first_image_noise_sigma = torch.exp(first_image_noise_sigma)
                    noisy_first_images = first_images + torch.randn_like(first_images) * first_image_noise_sigma[:, None, None, None, None]
                    first_image_latent_dist = vae.encode(noisy_first_images).latent_dist

                    last_images = last_images.permute(0, 2, 1, 3, 4)  # [B, C, F, H, W]
                    last_image_noise_sigma = torch.normal(
                        mean=-3.0, std=0.5, size=(last_images.size(0),), device=accelerator.device, dtype=weight_dtype
                    )
                    last_image_noise_sigma = torch.exp(last_image_noise_sigma)
                    noisy_last_images = last_images + torch.randn_like(last_images) * last_image_noise_sigma[:, None, None, None, None]
                    last_image_latent_dist = vae.encode(noisy_last_images).latent_dist

                    videos = videos.permute(0, 2, 1, 3, 4)  # [B, C, F, H, W]
                    latent_dist = vae.encode(videos).latent_dist
                else:
                    raise NotImplementedError
                    image_latent_dist = DiagonalGaussianDistribution(images)
                    latent_dist = DiagonalGaussianDistribution(videos)

                first_image_latents = first_image_latent_dist.sample() * VAE_SCALING_FACTOR
                first_image_latents = first_image_latents.permute(0, 2, 1, 3, 4)  # [B, F, C, H, W]
                first_image_latents = first_image_latents.to(memory_format=torch.contiguous_format, dtype=weight_dtype)

                last_image_latents = last_image_latent_dist.sample() * VAE_SCALING_FACTOR
                last_image_latents = last_image_latents.permute(0, 2, 1, 3, 4)  # [B, F, C, H, W]
                last_image_latents = last_image_latents.to(memory_format=torch.contiguous_format, dtype=weight_dtype)

                video_latents = latent_dist.sample() * VAE_SCALING_FACTOR
                video_latents = video_latents.permute(0, 2, 1, 3, 4)  # [B, F, C, H, W]
                video_latents = video_latents.to(memory_format=torch.contiguous_format, dtype=weight_dtype)

                padding_shape = (video_latents.shape[0], video_latents.shape[1] - 2, *video_latents.shape[2:])
                latent_padding = first_image_latents.new_zeros(padding_shape)
                image_latents = torch.cat([first_image_latents, latent_padding, last_image_latents], dim=1)  

                if random.random() < args.noised_image_dropout:
                    image_latents = torch.zeros_like(image_latents)

                # Encode prompts
                if not args.load_tensors:
                    prompt_embeds = compute_prompt_embeddings(
                        tokenizer,
                        text_encoder,
                        prompts,
                        model_config.max_text_seq_length,
                        accelerator.device,
                        weight_dtype,
                        requires_grad=False,
                    )
                else:
                    prompt_embeds = prompts.to(dtype=weight_dtype)

                # Sample noise that will be added to the latents
                noise = torch.randn_like(video_latents)
                batch_size, num_frames, num_channels, height, width = video_latents.shape

                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0,
                    scheduler.config.num_train_timesteps,
                    (batch_size,),
                    dtype=torch.int64,
                    device=accelerator.device,
                )

                # Prepare rotary embeds
                image_rotary_emb = (
                    prepare_rotary_positional_embeddings(
                        height=height * VAE_SCALE_FACTOR_SPATIAL,
                        width=width * VAE_SCALE_FACTOR_SPATIAL,
                        num_frames=num_frames,
                        vae_scale_factor_spatial=VAE_SCALE_FACTOR_SPATIAL,
                        patch_size=model_config.patch_size,
                        patch_size_t=model_config.patch_size_t if hasattr(model_config, "patch_size_t") else None,
                        attention_head_dim=model_config.attention_head_dim,
                        device=accelerator.device,
                        base_height=RoPE_BASE_HEIGHT,
                        base_width=RoPE_BASE_WIDTH,
                    )
                    if model_config.use_rotary_positional_embeddings
                    else None
                )

                # Add noise to the model input according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_video_latents = scheduler.add_noise(video_latents, noise, timesteps)
                noisy_model_input = torch.cat([noisy_video_latents, image_latents], dim=2)

                ofs_embed_dim = model_config.ofs_embed_dim if hasattr(model_config, "ofs_embed_dim") else None,
                ofs_emb = None if ofs_embed_dim is None else noisy_model_input.new_full((1,), fill_value=2.0)
                # Predict the noise residual
                model_output = transformer(
                    hidden_states=noisy_model_input,
                    encoder_hidden_states=prompt_embeds,
                    timestep=timesteps,
                    ofs=ofs_emb,
                    image_rotary_emb=image_rotary_emb,
                    return_dict=False,
                )[0]

                model_pred = scheduler.get_velocity(model_output, noisy_video_latents, timesteps)

                weights = 1 / (1 - alphas_cumprod[timesteps])
                while len(weights.shape) < len(model_pred.shape):
                    weights = weights.unsqueeze(-1)

                target = video_latents  # shape: [1, 13, 16, 60, 90] ~ [b, f, c, h, w]
                loss = torch.mean(
                    (weights * (model_pred - target) ** 2).reshape(batch_size, -1),
                    dim=1,
                )
                loss = loss.mean()
                accelerator.backward(loss)

                if accelerator.sync_gradients and accelerator.distributed_type != DistributedType.DEEPSPEED:
                    gradient_norm_before_clip = get_gradient_norm(transformer.parameters())
                    accelerator.clip_grad_norm_(transformer.parameters(), args.max_grad_norm)
                    gradient_norm_after_clip = get_gradient_norm(transformer.parameters())
                    logs.update(
                        {
                            "gradient_norm_before_clip": gradient_norm_before_clip,
                            "gradient_norm_after_clip": gradient_norm_after_clip,
                        }
                    )

                if accelerator.state.deepspeed_plugin is None:
                    optimizer.step()
                    optimizer.zero_grad()

                if not args.use_cpu_offload_optimizer:
                    lr_scheduler.step()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                # Checkpointing
                if accelerator.is_main_process or accelerator.distributed_type == DistributedType.DEEPSPEED:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"Removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        torch.save(transformer.patch_embed.proj.state_dict(), os.path.join(save_path, "patch_embed-proj.pth"))
                        logger.info(f"Saved state to {save_path}")

                # Validation after step
                # should_run_validation = (
                #     args.validation_steps is not None and global_step % args.validation_steps == 0
                # )
                should_run_validation = (
                    args.validation_steps is not None and ((global_step % args.validation_steps == 0) or (global_step==1))
                )
                if should_run_validation and accelerator.is_main_process:
                    run_validation(args, accelerator, transformer, scheduler, model_config, weight_dtype, global_step=global_step)

            last_lr = lr_scheduler.get_last_lr()[0] if lr_scheduler is not None else args.learning_rate
            logs.update(
                {
                    "loss": loss.detach().item(),
                    "lr": last_lr,
                }
            )
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break
        
        # Validation after epoch
        # if accelerator.is_main_process:
        #     should_run_validation = args.validation_prompt is not None and (
        #         args.validation_epochs is not None and (epoch + 1) % args.validation_epochs == 0
        #     )
        #     if should_run_validation:
        #         run_validation(args, accelerator, transformer, scheduler, model_config, weight_dtype, global_step="last")

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        transformer = unwrap_model(accelerator, transformer)
        dtype = (
            torch.float16
            if args.mixed_precision == "fp16"
            else torch.bfloat16
            if args.mixed_precision == "bf16"
            else torch.float32
        )
        transformer = transformer.to(dtype)
        transformer_lora_layers = get_peft_model_state_dict(transformer)

        CogVideoXInterpolationPipeline.save_lora_weights(
            save_directory=args.output_dir,
            transformer_lora_layers=transformer_lora_layers,
        )
        torch.save(transformer.patch_embed.proj.state_dict(), os.path.join(args.output_dir, "patch_embed-proj.pth"))

        # Cleanup trained models to save memory
        if args.load_tensors:
            del transformer
        else:
            del transformer, text_encoder, vae

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(accelerator.device)

        accelerator.print("===== Memory before testing =====")
        print_memory(accelerator.device)
        reset_memory(accelerator.device)

        # Final test inference
        pipe = CogVideoXInterpolationPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            revision=args.revision,
            variant=args.variant,
            torch_dtype=weight_dtype,
        )
        pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config)

        if args.enable_slicing:
            pipe.vae.enable_slicing()
        if args.enable_tiling:
            pipe.vae.enable_tiling()
        if args.enable_model_cpu_offload:
            pipe.enable_model_cpu_offload()

        # Load LoRA weights
        lora_scaling = args.lora_alpha / args.rank
        pipe.load_lora_weights(args.output_dir, adapter_name="cogvideox-lora")
        pipe.set_adapters(["cogvideox-lora"], [lora_scaling])

        # Run inference
        validation_outputs = []
        # if args.validation_prompt and args.num_validation_videos > 0:
        if args.num_validation_videos > 0:

            validation_dir = os.path.join(args.data_root, args.validation_dir)
            validation_videos_dir = [os.path.join(validation_dir, videoname) for videoname in os.listdir(validation_dir)]

            for validation_videos_dir_i in validation_videos_dir:
                validation_video_frames = [
                    os.path.join(validation_videos_dir_i, 'video', filename) for filename in os.listdir(os.path.join(validation_videos_dir_i, 'video'))
                ]
                validation_video_frames = sorted(validation_video_frames)
                validation_image_first, validation_image_last = validation_video_frames[0], validation_video_frames[-1]

                with open(os.path.join(validation_videos_dir_i, 'prompt.txt'), 'r') as f:
                    validation_prompt = f.read()
                    validation_prompt = validation_prompt.strip()

                pipeline_args = {
                    "first_image": load_image(validation_image_first),
                    "last_image": load_image(validation_image_last),
                    "prompt": validation_prompt,
                    "guidance_scale": args.guidance_scale,
                    "use_dynamic_cfg": args.use_dynamic_cfg,
                    "height": args.height,
                    "width": args.width,
                }

                video = log_validation(
                    accelerator=accelerator,
                    pipe=pipe,
                    args=args,
                    pipeline_args=pipeline_args,
                    is_final_validation=True,
                    image_id=os.path.basename(validation_videos_dir_i),
                )
                validation_outputs.extend(video)

        accelerator.print("===== Memory after testing =====")
        print_memory(accelerator.device)
        reset_memory(accelerator.device)
        torch.cuda.synchronize(accelerator.device)

        if args.push_to_hub and False:
            save_model_card(
                repo_id,
                videos=validation_outputs,
                base_model=args.pretrained_model_name_or_path,
                validation_prompt=args.validation_prompt,
                repo_folder=args.output_dir,
                fps=args.fps,
            )
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    accelerator.end_training()


if __name__ == "__main__":
    args = get_args()
    main(args)
