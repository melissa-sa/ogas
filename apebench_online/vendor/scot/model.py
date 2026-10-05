from typing import Optional, Tuple, List, Union
from dataclasses import dataclass
import collections.abc

import math
import torch
import torch.nn as nn

from apebench_online.vendor.scot.modules import (
    ConditionalLayerNorm,
    LayerNorm,
    ConvNeXtBlock,
    ResNetBlock,
    ScOTPatchMerging,
    ScOTPatchUnmerging,
    ScOTEncodeStage,
    ScOTDecodeStage,
)


@dataclass
class ScOTEncoderOutput:
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[tuple[torch.FloatTensor, ...]] = None
    reshaped_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None


class ScOTEncoder(nn.Module):
    """
    This is just a Swinv2Encoder with changed dpr.
    We just have to change the drop path rate since we also have a decoder by default.
    """

    def __init__(self, config, grid_size, pretrained_window_sizes=(0, 0, 0, 0)):
        super().__init__()
        self.num_layers = len(config.depths)
        self.config = config
        if self.config.pretrained_window_sizes is not None:
            pretrained_window_sizes = config.pretrained_window_sizes
        drop_rates_encode_decode = torch.linspace(0, config.drop_path_rate, 2 * sum(config.depths))
        dpr = [x.item() for x in drop_rates_encode_decode[: drop_rates_encode_decode.shape[0] // 2]]
        self.layers = nn.ModuleList(
            [
                ScOTEncodeStage(
                    config=config,
                    dim=int(config.embed_dim * 2**i_layer),
                    input_resolution=(
                        grid_size[0] // (2**i_layer),
                        grid_size[1] // (2**i_layer),
                    ),
                    depth=config.depths[i_layer],
                    num_heads=config.num_heads[i_layer],
                    drop_path=dpr[sum(config.depths[:i_layer]) : sum(config.depths[: i_layer + 1])],
                    downsample=(ScOTPatchMerging if (i_layer < self.num_layers - 1) else None),
                    pretrained_window_size=pretrained_window_sizes[i_layer],
                )
                for i_layer in range(self.num_layers)
            ]
        )

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_dimensions: Tuple[int, int],
        time: torch.Tensor,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        output_hidden_states_before_downsampling: Optional[bool] = False,
        always_partition: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple, ScOTEncoderOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_reshaped_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        if output_hidden_states:
            batch_size, _, hidden_size = hidden_states.shape
            # rearrange b (h w) c -> b c h w
            reshaped_hidden_state = hidden_states.view(batch_size, *input_dimensions, hidden_size)
            reshaped_hidden_state = reshaped_hidden_state.permute(0, 3, 1, 2)
            all_hidden_states += (hidden_states,)
            all_reshaped_hidden_states += (reshaped_hidden_state,)

        for i, layer_module in enumerate(self.layers):
            layer_head_mask = head_mask[i] if head_mask is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    input_dimensions,
                    time,
                    layer_head_mask,
                    output_attentions,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    input_dimensions,
                    time,
                    layer_head_mask,
                    output_attentions,
                    always_partition,
                )

            hidden_states = layer_outputs[0]
            hidden_states_before_downsampling = layer_outputs[1]
            output_dimensions = layer_outputs[2]

            input_dimensions = (output_dimensions[-2], output_dimensions[-1])

            if output_hidden_states and output_hidden_states_before_downsampling:
                batch_size, _, hidden_size = hidden_states_before_downsampling.shape
                # rearrange b (h w) c -> b c h w
                # here we use the original (not downsampled) height and width
                reshaped_hidden_state = hidden_states_before_downsampling.view(
                    batch_size,
                    *(output_dimensions[0], output_dimensions[1]),
                    hidden_size,
                )
                reshaped_hidden_state = reshaped_hidden_state.permute(0, 3, 1, 2)
                all_hidden_states += (hidden_states_before_downsampling,)
                all_reshaped_hidden_states += (reshaped_hidden_state,)
            elif output_hidden_states and not output_hidden_states_before_downsampling:
                batch_size, _, hidden_size = hidden_states.shape
                # rearrange b (h w) c -> b c h w
                reshaped_hidden_state = hidden_states.view(
                    batch_size, *input_dimensions, hidden_size
                )
                reshaped_hidden_state = reshaped_hidden_state.permute(0, 3, 1, 2)
                all_hidden_states += (hidden_states,)
                all_reshaped_hidden_states += (reshaped_hidden_state,)

            if output_attentions:
                all_self_attentions += layer_outputs[3:]

        if not return_dict:
            return tuple(
                v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None
            )

        return ScOTEncoderOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            reshaped_hidden_states=all_reshaped_hidden_states,
        )


class ScOTDecoder(nn.Module):
    """Here we do reverse encoder."""

    def __init__(self, config, grid_size, pretrained_window_sizes=(0, 0, 0, 0)):
        super().__init__()
        self.num_layers = len(config.depths)
        self.config = config
        if self.config.pretrained_window_sizes is not None:
            pretrained_window_sizes = config.pretrained_window_sizes
        drop_rates_encode_decode = torch.linspace(0, config.drop_path_rate, 2 * sum(config.depths))
        dpr = [x.item() for x in drop_rates_encode_decode[drop_rates_encode_decode.shape[0] // 2 :]]
        self.layers = nn.ModuleList(
            [
                ScOTDecodeStage(
                    config=config,
                    dim=int(config.embed_dim * 2**i_layer),
                    input_resolution=(
                        grid_size[0] // (2**i_layer),
                        grid_size[1] // (2**i_layer),
                    ),
                    depth=config.depths[i_layer],
                    num_heads=config.num_heads[i_layer],
                    drop_path=dpr[sum(config.depths[i_layer + 1 :]) : sum(config.depths[i_layer:])],
                    upsample=ScOTPatchUnmerging if i_layer > 0 else None,
                    upsampled_size=(
                        grid_size[0] // (2 ** (i_layer - 1)),
                        grid_size[1] // (2 ** (i_layer - 1)),
                    ),
                    pretrained_window_size=pretrained_window_sizes[i_layer],
                )
                for i_layer in reversed(range(self.num_layers))
            ]
        )

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_dimensions: Tuple[int, int],
        skip_states: List[torch.FloatTensor],
        time: torch.Tensor,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        output_hidden_states_before_upsampling: Optional[bool] = False,
        always_partition: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple, ScOTEncoderOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_reshaped_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        if output_hidden_states:
            batch_size, _, hidden_size = hidden_states.shape
            # rearrange b (h w) c -> b c h w
            reshaped_hidden_state = hidden_states.view(batch_size, *input_dimensions, hidden_size)
            reshaped_hidden_state = reshaped_hidden_state.permute(0, 3, 1, 2)
            all_hidden_states += (hidden_states,)
            all_reshaped_hidden_states += (reshaped_hidden_state,)

        for i, layer_module in enumerate(self.layers):
            layer_head_mask = head_mask[i] if head_mask is not None else None

            if i != 0 and skip_states[len(skip_states) - i] is not None:
                # residual connection
                hidden_states = hidden_states + skip_states[len(skip_states) - i]
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    input_dimensions,
                    time,
                    layer_head_mask,
                    output_attentions,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    input_dimensions,
                    time,
                    layer_head_mask,
                    output_attentions,
                    always_partition,
                )

            hidden_states = layer_outputs[0]
            hidden_states_before_upsampling = layer_outputs[1]
            output_dimensions = layer_outputs[2]

            input_dimensions = (output_dimensions[-2], output_dimensions[-1])

            if output_hidden_states and output_hidden_states_before_upsampling:
                batch_size, _, hidden_size = hidden_states_before_upsampling.shape
                # rearrange b (h w) c -> b c h w
                # here we use the original (not downsampled) height and width
                reshaped_hidden_state = hidden_states_before_upsampling.view(
                    batch_size,
                    *(output_dimensions[0], output_dimensions[1]),
                    hidden_size,
                )
                reshaped_hidden_state = reshaped_hidden_state.permute(0, 3, 1, 2)
                all_hidden_states += (hidden_states_before_upsampling,)
                all_reshaped_hidden_states += (reshaped_hidden_state,)
            elif output_hidden_states and not output_hidden_states_before_upsampling:
                batch_size, _, hidden_size = hidden_states.shape
                # rearrange b (h w) c -> b c h w
                reshaped_hidden_state = hidden_states.view(
                    batch_size, *input_dimensions, hidden_size
                )
                reshaped_hidden_state = reshaped_hidden_state.permute(0, 3, 1, 2)
                all_hidden_states += (hidden_states,)
                all_reshaped_hidden_states += (reshaped_hidden_state,)

            if output_attentions:
                all_self_attentions += layer_outputs[3:]

        if not return_dict:
            return tuple(
                v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None
            )

        return ScOTEncoderOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            reshaped_hidden_states=all_reshaped_hidden_states,
        )


@dataclass
class ScOTOutput:
    loss: Optional[torch.FloatTensor] = None
    output: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor, ...]] = None
    reshaped_hidden_states: Optional[Tuple[torch.FloatTensor]] = None


class ScOTConfig:
    def __init__(
        self,
        image_size=224,
        patch_size=4,
        num_channels=3,
        num_out_channels=3,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        skip_connections=[2, 2, 2, 0],
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        drop_path_rate=0.1,
        use_absolute_embeddings=False,
        use_conditioning=False,
        use_mask_token=False,
        use_return_dict=True,
        output_attentions=False,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        p=1,  # for loss: 1 for l1, 2 for l2
        pretrained_window_sizes=None,
        channel_slice_list_normalized_loss=None,  # if None will fall back to absolute loss otherwise normalized loss with split channels
        residual_model="convnext",  # "convnext" or "resnet"
        learn_residual=False,  # learn the residual for time-dependent problems
        output_hidden_states=True,
        cond_dim=1,
    ):
        super().__init__()

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_out_channels = num_out_channels
        self.embed_dim = embed_dim
        self.depths = depths
        self.num_layers = len(depths)
        self.num_heads = num_heads
        self.skip_connections = skip_connections
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.drop_path_rate = drop_path_rate
        self.use_absolute_embeddings = use_absolute_embeddings
        self.use_conditioning = use_conditioning
        self.use_mask_token = use_mask_token
        self.use_return_dict = use_return_dict
        self.output_attentions = output_attentions
        self.learn_residual = learn_residual if self.use_conditioning else False
        self.layer_norm_eps = layer_norm_eps
        self.initializer_range = initializer_range
        self.p = p
        self.pretrained_window_sizes = pretrained_window_sizes
        self.channel_slice_list_normalized_loss = channel_slice_list_normalized_loss
        self.residual_model = residual_model
        self.output_hidden_states = output_hidden_states
        self.cond_dim = cond_dim


class ScOTPatchEmbeddings(nn.Module):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """

    def __init__(self, config):
        super().__init__()
        image_size, patch_size = config.image_size, config.patch_size
        num_channels, hidden_size = config.num_channels, config.embed_dim
        image_size = (
            image_size
            if isinstance(image_size, collections.abc.Iterable)
            else (image_size, image_size)
        )
        patch_size = (
            patch_size
            if isinstance(patch_size, collections.abc.Iterable)
            else (patch_size, patch_size)
        )
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches
        self.grid_size = (
            image_size[0] // patch_size[0],
            image_size[1] // patch_size[1],
        )

        self.projection = nn.Conv2d(
            num_channels, hidden_size, kernel_size=patch_size, stride=patch_size
        )

    def maybe_pad(self, pixel_values, height, width):
        if width % self.patch_size[1] != 0:
            pad_values = (0, self.patch_size[1] - width % self.patch_size[1])
            pixel_values = nn.functional.pad(pixel_values, pad_values)
        if height % self.patch_size[0] != 0:
            pad_values = (0, 0, 0, self.patch_size[0] - height % self.patch_size[0])
            pixel_values = nn.functional.pad(pixel_values, pad_values)
        return pixel_values

    def forward(self, pixel_values: Optional[torch.FloatTensor]) -> Tuple[torch.Tensor, Tuple[int]]:
        _, num_channels, height, width = pixel_values.shape
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        # pad the input to be divisible by self.patch_size, if needed
        pixel_values = self.maybe_pad(pixel_values, height, width)
        embeddings = self.projection(pixel_values)
        _, _, height, width = embeddings.shape
        output_dimensions = (height, width)
        embeddings = embeddings.flatten(2).transpose(1, 2)

        return embeddings, output_dimensions


class ScOTEmbeddings(nn.Module):
    """
    Construct the patch and position embeddings. Optionally, also the mask token.
    """

    def __init__(self, config, use_mask_token=False):
        super().__init__()

        self.patch_embeddings = ScOTPatchEmbeddings(config)
        num_patches = self.patch_embeddings.num_patches
        self.patch_grid = self.patch_embeddings.grid_size
        self.mask_token = (
            nn.Parameter(torch.zeros(1, 1, config.embed_dim)) if use_mask_token else None
        )

        if config.use_absolute_embeddings:
            self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches, config.embed_dim))
        else:
            self.position_embeddings = None

        if config.use_conditioning:
            layer_norm = ConditionalLayerNorm
        else:
            layer_norm = LayerNorm

        cond_dim = getattr(config, "cond_dim", 1)
        self.norm = layer_norm(config.embed_dim, cond_dim=cond_dim)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor],
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        time: Optional[torch.FloatTensor] = None,
    ) -> Tuple[torch.Tensor]:
        embeddings, output_dimensions = self.patch_embeddings(pixel_values)
        embeddings = self.norm(embeddings, time)
        batch_size, seq_len, _ = embeddings.size()

        if bool_masked_pos is not None:
            mask_tokens = self.mask_token.expand(batch_size, seq_len, -1)
            # replace the masked visual tokens by mask_tokens
            mask = bool_masked_pos.unsqueeze(-1).type_as(mask_tokens)
            embeddings = embeddings * (1.0 - mask) + mask_tokens * mask

        if self.position_embeddings is not None:
            embeddings = embeddings + self.position_embeddings

        embeddings = self.dropout(embeddings)

        return embeddings, output_dimensions


class ScOTPatchRecovery(nn.Module):
    """https://github.com/198808xc/Pangu-Weather/blob/main/pseudocode.py"""

    def __init__(self, config):
        super().__init__()
        image_size, patch_size = config.image_size, config.patch_size
        num_out_channels, hidden_size = (
            config.num_out_channels,
            config.embed_dim,  # if not config.skip_connections[0] else 2 * config.embed_dim,
        )
        image_size = (
            image_size
            if isinstance(image_size, collections.abc.Iterable)
            else (image_size, image_size)
        )
        patch_size = (
            patch_size
            if isinstance(patch_size, collections.abc.Iterable)
            else (patch_size, patch_size)
        )
        num_patches = (image_size[0] // patch_size[0]) * (image_size[1] // patch_size[1])
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.image_size = image_size
        self.num_out_channels = num_out_channels
        self.grid_size = (
            image_size[0] // patch_size[0],
            image_size[1] // patch_size[1],
        )

        self.projection = nn.ConvTranspose2d(
            in_channels=hidden_size,
            out_channels=num_out_channels,
            kernel_size=patch_size,
            stride=patch_size,
        )
        # the following is not done in Pangu
        self.mixup = nn.Conv2d(
            num_out_channels,
            num_out_channels,
            kernel_size=5,
            stride=1,
            padding=2,
            bias=False,
        )

    def maybe_crop(self, pixel_values, height, width):
        if pixel_values.shape[2] > height:
            pixel_values = pixel_values[:, :, :height, :]
        if pixel_values.shape[3] > width:
            pixel_values = pixel_values[:, :, :, :width]
        return pixel_values

    def forward(self, hidden_states):
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = hidden_states.reshape(
            hidden_states.shape[0], hidden_states.shape[1], *self.grid_size
        )

        output = self.projection(hidden_states)
        output = self.maybe_crop(output, self.image_size[0], self.image_size[1])
        return self.mixup(output)


class ScOT(nn.Module):
    """Inspired by https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/models/swinv2/modeling_swinv2.py#L1129"""

    def __init__(self, config, use_mask_token=False):
        super().__init__()
        self.config = config
        self.num_layers_encoder = len(config.depths)
        self.num_layers_decoder = len(config.depths)
        self.num_features = int(config.embed_dim * 2 ** (self.num_layers_encoder - 1))

        self.embeddings = ScOTEmbeddings(config, use_mask_token=use_mask_token)
        self.encoder = ScOTEncoder(config, self.embeddings.patch_grid)
        self.decoder = ScOTDecoder(config, self.embeddings.patch_grid)
        self.patch_recovery = ScOTPatchRecovery(config)

        if config.residual_model == "convnext":
            res_model = ConvNeXtBlock
        elif config.residual_model == "resnet":
            res_model = ResNetBlock
        else:
            raise ValueError("residual_model must be 'convnext' or 'resnet'")

        self.residual_blocks = nn.ModuleList(
            [
                (
                    nn.ModuleList(
                        [res_model(config, config.embed_dim * 2**i) for _ in range(depth)]
                    )
                    if depth > 0
                    else nn.ModuleList([nn.Identity()])
                )
                for i, depth in enumerate(config.skip_connections)
            ]
        )

        self.apply(self._initialize_weights)

    def _initialize_weights(self, module):
        """
        Initialize the weights if they are not already initialized.
        """
        if getattr(module, "_is_initialized", False):
            return
        self._init_weights(module)
        module._is_initialized = True

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, LayerNorm) or isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, ConditionalLayerNorm):
            module.weight.bias.data.zero_()
            module.bias.bias.data.zero_()
            module.weight.weight.data.fill_(1.0)
            module.bias.weight.data.fill_(1.0)

    def get_head_mask(
        self,
        head_mask: Optional[torch.Tensor],
        num_hidden_layers: int,
        is_attention_chunked: bool = False,
    ) -> torch.Tensor:
        """
        Prepare the head mask if needed.

        Args:
            head_mask (`torch.Tensor` with shape `[num_heads]` or `[num_hidden_layers x num_heads]`, *optional*):
                The mask indicating if we should keep the heads or not (1.0 for keep, 0.0 for discard).
            num_hidden_layers (`int`):
                The number of hidden layers in the model.
            is_attention_chunked (`bool`, *optional*, defaults to `False`):
                Whether or not the attentions scores are computed by chunks or not.

        Returns:
            `torch.Tensor` with shape `[num_hidden_layers x batch x num_heads x seq_length x seq_length]` or list with
            `[None]` for each layer.
        """
        if head_mask is not None:
            head_mask = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
            if is_attention_chunked is True:
                head_mask = head_mask.unsqueeze(-1)
        else:
            head_mask = [None] * num_hidden_layers

        return head_mask

    def _convert_head_mask_to_5d(self, head_mask, num_hidden_layers):
        """-> [num_hidden_layers x batch x num_heads x seq_length x seq_length]"""
        if head_mask.dim() == 1:
            head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
        elif head_mask.dim() == 2:
            head_mask = (
                head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            )  # We can specify head_mask for each layer
        assert head_mask.dim() == 5, f"head_mask.dim != 5, instead {head_mask.dim()}"
        head_mask = head_mask.to(dtype=self.dtype)  # switch to float if need + fp16 compatibility
        return head_mask

    def get_input_embeddings(self):
        return self.embeddings.patch_embeddings

    def _prune_heads(self, heads_to_prune):
        for layer, heads in heads_to_prune.items():
            self.encoder.layers[layer].attention.prune_heads(heads)
        for layer, heads in reversed(heads_to_prune.items()):
            self.decoder.layers[layer].attention.prune_heads(heads)

    def _downsample(self, image, target_size):
        image_size = image.shape[-2]
        freqs = torch.fft.fftfreq(image_size, d=1 / image_size)
        sel = torch.logical_and(freqs >= -target_size / 2, freqs <= target_size / 2 - 1)
        image_hat = torch.fft.fft2(image, norm="forward")
        image_hat = image_hat[:, :, sel, :][:, :, :, sel]
        image = torch.fft.ifft2(image_hat, norm="forward").real
        return image

    def _upsample(self, image, target_size):
        # https://stackoverflow.com/questions/71143279/upsampling-images-in-frequency-domain-using-pytorch
        image_size = image.shape[-2]
        image_hat = torch.fft.fft2(image, norm="forward")
        image_hat = torch.fft.fftshift(image_hat)
        pad_size = (target_size - image_size) // 2
        real = nn.functional.pad(
            image_hat.real, (pad_size, pad_size, pad_size, pad_size), value=0.0
        )
        imag = nn.functional.pad(
            image_hat.imag, (pad_size, pad_size, pad_size, pad_size), value=0.0
        )
        image_hat = torch.fft.ifftshift(torch.complex(real, imag))
        image = torch.fft.ifft2(image_hat, norm="forward").real
        return image

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        time: Optional[torch.FloatTensor] = None,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        pixel_mask: Optional[torch.BoolTensor] = None,
        labels: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, ScOTOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )

        if pixel_values is None:
            raise ValueError("pixel_values cannot be None")

        head_mask = self.get_head_mask(head_mask, self.num_layers_encoder + self.num_layers_decoder)

        if isinstance(head_mask, list):
            head_mask_encoder = head_mask[: self.num_layers_encoder]
            head_mask_decoder = head_mask[self.num_layers_encoder :]
        else:
            head_mask_encoder, head_mask_decoder = head_mask.split(
                [self.num_layers_encoder, self.num_layers_decoder]
            )

        image_size = pixel_values.shape[2]
        # image must be square
        if image_size != self.config.image_size:
            if image_size < self.config.image_size:
                pixel_values = self._upsample(pixel_values, self.config.image_size)
            else:
                pixel_values = self._downsample(pixel_values, self.config.image_size)

        embedding_output, input_dimensions = self.embeddings(
            pixel_values, bool_masked_pos=bool_masked_pos, time=time
        )

        encoder_outputs = self.encoder(
            embedding_output,
            input_dimensions,
            time,
            head_mask=head_mask_encoder,
            output_attentions=output_attentions,
            output_hidden_states=True,
            output_hidden_states_before_downsampling=True,
            return_dict=return_dict,
        )

        if return_dict:
            skip_states = list(encoder_outputs.hidden_states[1:])
        else:
            skip_states = list(encoder_outputs[1][1:])

        for i in range(len(skip_states)):
            for block in self.residual_blocks[i]:
                if isinstance(block, nn.Identity):
                    skip_states[i] = block(skip_states[i])
                else:
                    skip_states[i] = block(skip_states[i], time)

        #! assumes square images
        input_dim = math.floor(skip_states[-1].shape[1] ** 0.5)
        decoder_output = self.decoder(
            skip_states[-1],
            (input_dim, input_dim),
            time=time,
            skip_states=skip_states[:-1],
            head_mask=head_mask_decoder,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if return_dict:
            sequence_output = decoder_output.last_hidden_state
        else:
            sequence_output = decoder_output[0]
        prediction = self.patch_recovery(sequence_output)
        # The following can be used for learning just the residual for time-dependent problems
        if self.config.learn_residual:
            if self.config.num_channels > self.config.num_out_channels:
                pixel_values = pixel_values[:, 0 : self.config.num_out_channels]
            prediction += pixel_values

        if image_size != self.config.image_size:
            if image_size > self.config.image_size:
                prediction = self._upsample(prediction, image_size)
            else:
                prediction = self._downsample(prediction, image_size)

        if pixel_mask is not None:
            prediction[pixel_mask] = labels[pixel_mask].type_as(prediction)
        loss = None
        if labels is not None:
            if self.config.p == 1:
                loss_fn = nn.functional.l1_loss
            elif self.config.p == 2:
                loss_fn = nn.functional.mse_loss
            else:
                raise ValueError("p must be 1 or 2")
            if self.config.channel_slice_list_normalized_loss is not None:
                loss = torch.mean(
                    torch.stack(
                        [
                            loss_fn(
                                prediction[
                                    :,
                                    self.config.channel_slice_list_normalized_loss[
                                        i
                                    ] : self.config.channel_slice_list_normalized_loss[i + 1],
                                ],
                                labels[
                                    :,
                                    self.config.channel_slice_list_normalized_loss[
                                        i
                                    ] : self.config.channel_slice_list_normalized_loss[i + 1],
                                ],
                            )
                            / (
                                loss_fn(
                                    labels[
                                        :,
                                        self.config.channel_slice_list_normalized_loss[
                                            i
                                        ] : self.config.channel_slice_list_normalized_loss[i + 1],
                                    ],
                                    torch.zeros_like(
                                        labels[
                                            :,
                                            self.config.channel_slice_list_normalized_loss[
                                                i
                                            ] : self.config.channel_slice_list_normalized_loss[
                                                i + 1
                                            ],
                                        ]
                                    ),
                                )
                                + 1e-10
                            )
                            for i in range(len(self.config.channel_slice_list_normalized_loss) - 1)
                        ]
                    )
                )
            else:
                loss = loss_fn(prediction, labels)

        if not return_dict:
            output = (prediction,) + decoder_output[1:] + encoder_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return ScOTOutput(
            loss=loss,
            output=prediction,
            hidden_states=(
                decoder_output.hidden_states + encoder_outputs.hidden_states
                if output_hidden_states is not None and output_hidden_states is True
                else None
            ),
            attentions=(
                decoder_output.attentions + encoder_outputs.attentions
                if output_attentions is not None and output_attentions is True
                else None
            ),
            reshaped_hidden_states=(
                decoder_output.reshaped_hidden_states + encoder_outputs.reshaped_hidden_states
                if output_hidden_states is not None and output_hidden_states is True
                else None
            ),
        )
