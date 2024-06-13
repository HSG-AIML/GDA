"""Model definitions."""

import functools
import re

import timm
import torch

import src.utils
import src.t_few_lora
import src.pos_embed
import src.patch_embed

# mae imports
import src.mae.models_mae
import src.mae.models_vit

# sat-mae imports
import src.sat_mae.models_mae
import src.sat_mae.models_vit
import src.sat_mae.models_mae
import src.sat_mae.models_vit_group_channels
import src.sat_mae.models_vit_temporal
import src.sat_mae.models_vit_group_channels

# scale_mae imports
import src.scale_mae.models_mae
import src.scale_mae.models_vit
import src.scale_mae.util.pos_embed


class ScaledLowRankConvAdapter(torch.nn.Module):
    """SLR adapter for conv layers."""

    def __init__(self, conv2d: torch.nn.Conv2d, hidden_dim: int = 16):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.proj = conv2d
        self.kernel_size = conv2d.kernel_size
        self.scaler = torch.nn.Parameter(torch.ones(self.proj.out_channels))

        assert conv2d.kernel_size == (16, 16)
        kernel_size = (4, 4)
        self.down = torch.nn.Conv2d(
            self.proj.in_channels,
            self.hidden_dim,
            kernel_size=kernel_size,
            stride=kernel_size,
        )
        self.up = torch.nn.Conv2d(
            self.hidden_dim,
            self.proj.out_channels,
            kernel_size=kernel_size,
            stride=kernel_size,
        )

        for p in self.proj.parameters():
            p.requires_grad = False

    def forward(self, x: torch.tensor) -> torch.tensor:
        x_lr = self.up(self.down(x))
        x = self.proj(x)

        x += x_lr

        return torch.einsum("bdhw,d->bdhw", x, self.scaler)


class ScaledLowRankAdapter(torch.nn.Module):
    """SLR adapter for linear layers.

    Adds a low rank adapter and scaling parameters to a linear layer"
    """

    def __init__(self, linear: torch.nn.Linear, hidden_dim: int = 16):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.linear = linear
        self.out_dim, self.in_dim = self.linear.weight.shape

        # freeze original parameters
        for p in self.linear.parameters():
            p.requires_grad = False

        # initialize scaling vectors as ones
        self.in_scaler = torch.nn.Parameter(torch.ones(self.in_dim))
        self.out_scaler = torch.nn.Parameter(torch.ones(self.out_dim))

        self.down = torch.nn.Linear(self.in_dim, self.hidden_dim)
        self.up = torch.nn.Linear(self.hidden_dim, self.out_dim)

        # init low-rank matrices as normal/zeros
        self.up.weight.data.fill_(0)
        self.up.bias.data.fill_(0)
        torch.nn.init.normal_(self.down.weight.data)

    def forward(self, x: torch.tensor) -> torch.tensor:
        # x *= self.in_scaler
        # x_lr = self.up(self.down(x))
        # x = self.linear(x)
        # x += x_lr
        # x *= self.out_scaler

        # without in-place operations (chaned due to torch error, version above used for most experiments)
        x_scaled = x * self.in_scaler
        x_lr = self.up(self.down(x_scaled))
        x = self.linear(x_scaled)
        x_new = x + x_lr
        x = x_new * self.out_scaler

        # return x + x_lr
        return x


class ScalingConvAdapter(torch.nn.Module):
    """Scaling adapter for conv layers."""

    def __init__(self, conv2d: torch.nn.Conv2d, hidden_dim: int = 16):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.proj = conv2d

        self.scaler = torch.nn.Parameter(torch.ones(self.proj.out_channels))
        for p in self.proj.parameters():
            p.requires_grad = False

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.proj(x)
        x *= self.scaler
        return x


class ScalingAdapter(torch.nn.Module):
    """Scaling adapter for linear layers."""

    def __init__(self, linear: torch.nn.Linear, hidden_dim: int = 16):
        """Add a low rank adapter and scaling parameters to a linear layer"""
        super().__init__()

        self.hidden_dim = hidden_dim
        self.linear = linear
        self.out_dim, self.in_dim = self.linear.weight.shape

        # freeze original layer
        for p in self.linear.parameters():
            p.requires_grad = False

        self.in_scaler = torch.nn.Parameter(torch.ones(self.in_dim))
        # self.out_scaler = torch.nn.Parameter(torch.ones(self.out_dim))

    def forward(self, x: torch.tensor) -> torch.tensor:
        x *= self.in_scaler
        x = self.linear(x)
        # x *= self.out_scaler

        return x


class LowRankAdapter(torch.nn.Module):
    """Low-rank adapter for linear layers."""

    def __init__(self, linear: torch.nn.Linear, hidden_dim=16):
        """Add a low rank adapter to a linear layer"""
        super().__init__()

        self.hidden_dim = hidden_dim
        self.linear = linear
        self.out_dim, self.in_dim = self.linear.weight.shape

        # freeze original layer
        for p in self.linear.parameters():
            p.requires_grad = False

        self.down = torch.nn.Linear(self.in_dim, self.hidden_dim)
        self.up = torch.nn.Linear(self.hidden_dim, self.out_dim)
        self.up.weight.data.fill_(0)
        self.up.bias.data.fill_(0)
        torch.nn.init.normal_(self.down.weight.data)

    def forward(self, x: torch.tensor) -> torch.tensor:
        x_lr = self.up(self.down(x))
        x = self.linear(x)

        return x + x_lr


class LowRankConvAdapter(torch.nn.Module):
    """Low-rank adapter for conv layers."""

    def __init__(self, conv2d: torch.nn.Conv2d, hidden_dim: int = 16):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.proj = conv2d
        self.kernel_size = conv2d.kernel_size

        assert conv2d.kernel_size == (16, 16)
        kernel_size = (4, 4)
        self.down = torch.nn.Conv2d(
            self.proj.in_channels,
            self.hidden_dim,
            kernel_size=kernel_size,
            stride=kernel_size,
        )
        self.up = torch.nn.Conv2d(
            self.hidden_dim,
            self.proj.out_channels,
            kernel_size=kernel_size,
            stride=kernel_size,
        )

        for p in self.proj.parameters():
            p.requires_grad = False

    def forward(self, x: torch.tensor) -> torch.tensor:
        x_lr = self.up(self.down(x))
        x = self.proj(x)

        x += x_lr

        return x


class IA3ConfigTimmViT:
    """Config to add IA3 adapters to a timm ViT."""

    def __init__(self):
        self.lora_scaling_rank = 1
        self.lora_rank = 0
        self.lora_init_scale = 0.0
        self.lora_modules = ".*attn|.*mlp"
        self.lora_layers = "qkv|fc1|fc2|proj"
        self.trainable_param_names = ".*lora_b.*"
        self.model_modifier = "lora"


class ScaledLowRankConfigTimmViT:
    """Config to add SLR adapters to a timm ViT."""

    def __init__(
        self, hidden_dim: int = 8, patch_embed: bool = False, norm: bool = True
    ):
        self.lora_rank = hidden_dim
        self.adapter_modules = ".*attn|.*mlp|decoder_embed|decoder_pred"
        self.adapter_layers = "qkv|fc1|fc2|proj|decoder_embed|decoder_pred"
        if patch_embed:
            self.adapter_modules += "|patch_embed"
            self.adapter_layers += "|proj"
        self.model_modifier = "adapter"
        self.extra_trainable_param_names = "fcn_high.pred.7|fcn_low.pred.7"
        if norm:
            self.extra_trainable_param_names += "|.*norm.*"
        # self.extra_trainable_param_names = ".*norm.*|.*decoder_embed.*|.*decoder_pred.*"
        # self.extra_trainable_param_names = ".*norm.*|.*decoder_embed.*"


class LowRankConfigTimmViT:
    """Config to add low-rank adapters to a timm ViT."""

    def __init__(
        self, hidden_dim: int = 8, patch_embed: bool = False, norm: bool = True
    ):
        self.lora_rank = hidden_dim
        self.adapter_modules = ".*attn|.*mlp|decoder_embed|decoder_pred"
        self.adapter_layers = "qkv|fc1|fc2|proj|decoder_embed|decoder_pred"
        if patch_embed:
            self.adapter_modules += "|patch_embed"
            self.adapter_layers += "proj"
        self.model_modifier = "adapter"
        self.extra_trainable_param_names = "fcn_high.pred.7|fcn_low.pred.7"
        if norm:
            self.extra_trainable_param_names += "|.*norm.*"
        # self.extra_trainable_param_names = ".*norm.*|.*decoder_embed.*|.*decoder_pred.*"
        # self.extra_trainable_param_names = ".*norm.*|.*decoder_embed.*"


class ScalingConfigTimmViT:
    """Config to add scaling adapters to a timm ViT."""

    def __init__(
        self, hidden_dim: int = 8, patch_embed: bool = False, norm: bool = True
    ):
        self.lora_rank = hidden_dim
        self.adapter_modules = ".*attn|.*mlp|decoder_embed|decoder_pred"
        self.adapter_layers = "qkv|fc1|fc2|proj|decoder_embed|decoder_pred"
        if patch_embed:
            self.adapter_modules += "|patch_embed"
            self.adapter_layers += "proj"
        self.model_modifier = "adapter"
        self.extra_trainable_param_names = "fcn_high.pred.7|fcn_low.pred.7"
        if norm:
            self.extra_trainable_param_names += "|.*norm.*"
        # self.extra_trainable_param_names = ".*norm.*|.*decoder_embed.*|.*decoder_pred.*"
        # self.extra_trainable_param_names = ".*norm.*|.*decoder_embed.*"


def add_extra_weights(
    model,
    config,
    adapter,
    conv_adapter=None,
    trainable=True,
    only_scaler_trainable=False,
):
    # together with config of type ScaledLowRankConfigTimmViT
    for m_name, module in dict(model.named_modules()).items():
        if re.fullmatch(config.adapter_modules, m_name):
            children = dict(module.named_children())
            set_as_module = False
            if not children:
                set_as_module = True
                # if module is a layer
                children = {m_name: module}
            for c_name, layer in children.items():
                if re.fullmatch(config.adapter_layers, c_name):
                    if isinstance(layer, torch.nn.Linear):
                        adp = adapter
                    elif isinstance(layer, torch.nn.Conv2d):
                        adp = conv_adapter
                    else:
                        raise ValueError()
                    adapter_instance = adp(layer, hidden_dim=config.lora_rank)
                    if not trainable:
                        for p in adapter_instance.parameters():
                            p.requires_grad = False
                    if only_scaler_trainable:
                        for n, p in adapter_instance.named_parameters():
                            if "scaler" in n:
                                p.requires_grad = True
                            else:
                                p.requires_grad = False
                    if set_as_module:
                        setattr(model, c_name, adapter_instance)
                    else:
                        setattr(module, c_name, adapter_instance)

        # make extra params trainable (e.g., layer norm layers)
        if re.fullmatch(config.extra_trainable_param_names, m_name):
            for p in module.parameters():
                p.requires_grad = True

    return model


def add_adapter(
    model: torch.nn.Module,
    type: str = "lora",
    shared: bool = True,
    scale: int = 1,
    hidden_dim: int = 8,
    patch_embed_adapter: bool = False,
    adapter_trainable: bool = True,
    norm_trainable: bool = True,
    only_scaler_trainable: bool = False,
) -> torch.nn.Module:
    if type == "lora":
        config = LowRankConfigTimmViT(
            hidden_dim=hidden_dim,
            patch_embed=patch_embed_adapter,
            norm=norm_trainable,
        )
        model = add_extra_weights(
            model,
            config,
            LowRankAdapter,
            LowRankConvAdapter,
            adapter_trainable,
            only_scaler_trainable,
        )

    elif type == "ia3":
        config = ScalingConfigTimmViT(
            hidden_dim=hidden_dim,
            patch_embed=patch_embed_adapter,
            norm=norm_trainable,
        )
        model = add_extra_weights(
            model,
            config,
            ScalingAdapter,
            ScalingConvAdapter,
            adapter_trainable,
            only_scaler_trainable,
        )

    elif type == "low-rank-scaling":
        assert shared == False
        config = ScaledLowRankConfigTimmViT(
            hidden_dim=hidden_dim,
            patch_embed=patch_embed_adapter,
            norm=norm_trainable,
        )
        model = add_extra_weights(
            model,
            config,
            ScaledLowRankAdapter,
            ScaledLowRankConvAdapter,
            adapter_trainable,
            only_scaler_trainable,
        )

    else:
        raise AttributeError(f"type must be in ['lora', 'ia3', 'low-rank-scaling']")

    return model


def set_adapter_requires_grad(model, train=True):
    trainable_params = 0
    for n, p in model.named_parameters():
        if "adapter" in n:
            p.requires_grad = train
            trainable_params += p.numel()

    return trainable_params


def get_vit_l_imagenet(
    pretrained=True, num_classes=1000, in_chans=3, img_size=224, patch_size=16
):
    """return a timm vit-l model
    model_type determines which (if pretrained=True) weights are loaded
    """
    assert patch_size == 16
    model = timm.create_model(
        "vit_large_patch16_224", pretrained=pretrained, num_classes=num_classes
    )
    return model


def get_mae(
    model_type,
    pretrained=True,
    num_classes=1000,
    in_chans=3,
    img_size=224,
    patch_size=16,
):
    if model_type == "mae":
        assert patch_size == 16
        assert img_size == 224
        # assert in_chans % 3 == 0
        model = src.mae.models_mae.mae_vit_large_patch16_dec512d8b(in_chans=in_chans)

        if pretrained:
            # checkpoint with decode head
            checkpoint = torch.load(
                "/netscratch/lscheibenreif/code/low-rank-da/checkpoints/mae/mae_visualize_vit_large.pth",
                map_location=torch.device("cpu"),
            )
            ckpt_patch_embed_weight = checkpoint["model"]["patch_embed.proj.weight"]
            if in_chans != ckpt_patch_embed_weight.shape[1]:
                if in_chans % ckpt_patch_embed_weight.shape[1] == 0:
                    print(
                        f"Rescaling pretrained patch_embed weight to fit new {in_chans=}"
                    )
                    new_pe_weight = upscale_patch_embed(
                        in_chans,
                        ckpt_patch_embed_weight,
                    )
                    checkpoint["model"]["patch_embed.proj.weight"] = new_pe_weight
                    if model_type == "mae":
                        # also change final decoder layer from 3*patch_size*patch_size
                        print(
                            "Rescaling final decoder layer for new number of in_chans"
                        )
                        target_pix = (
                            in_chans * patch_size**2
                        )  # number of pixels per token
                        weight, bias = upscale_decoder_pred(
                            target_pix,
                            checkpoint["model"]["decoder_pred.weight"],
                            checkpoint["model"]["decoder_pred.bias"],
                        )
                        checkpoint["model"]["decoder_pred.weight"] = weight
                        checkpoint["model"]["decoder_pred.bias"] = bias
                else:
                    print("Deleting pre-trained patch embedding weights")
                    del checkpoint["model"]["patch_embed.proj.weight"]
                    if model_type == "mae":
                        del checkpoint["model"]["decoder_pred.weight"]
                        del checkpoint["model"]["decoder_pred.bias"]
            model.load_state_dict(checkpoint["model"], strict=False)

    elif model_type == "":
        assert patch_size == 16
        assert img_size == 224
        # assert in_chans == 3
        model = src.mae.models_vit.vit_large_patch16(in_chans=in_chans)
        if pretrained:
            checkpoint = torch.load(
                "/netscratch/lscheibenreif/code/low-rank-da/checkpoints/mae/mae_pretrain_vit_large.pth"
            )
            if in_chans != 3:
                del checkpoint["model"]["patch_embed.proj.weight"]
            print(model.load_state_dict(checkpoint["model"], strict=False))

    del checkpoint
    torch.cuda.empty_cache()

    return model


def get_scale_mae(
    model_type: str,
    pretrained: bool = True,
    num_classes: int = 1000,
    in_chans: int = 3,
    img_size: int = 224,
    patch_size: int = 16,
    fixed_output_size=0,
    use_mask_token=True,
):
    """return the scale-mae model"""

    checkpoint = torch.load(
        "/netscratch/lscheibenreif/code/low-rank-da/checkpoints/scale-mae/scalemae-vitlarge-800.pth",
        map_location=torch.device("cpu"),
    )
    args = checkpoint["args"]
    args.model = args.model.replace("mae_", "")
    args.nb_classes = num_classes
    args.global_pool = False

    args.input_size = img_size
    args.in_chans = in_chans
    args.patch_size = patch_size

    print(f"{args.model=}")
    if model_type == "mae":
        model = src.scale_mae.models_mae.mae_vit_large_patch16_dec512d8b(
            img_size=args.input_size,
            in_chans=args.in_chans,
            fixed_output_size=0,
            use_mask_token=True,
            decoder_depth=3,
            fcn_layers=2,
            # fpn_layers=2,
            fcn_dim=512,
            decoder_aux_loss_layers=1,
            independent_fcn_head=True,
            loss_masking=True,
            norm_pix_loss=True,
            use_l1_loss=False,
            progressive=False,
        )
    else:
        model = src.scale_mae.models_vit.__dict__[args.model](
            img_size=args.input_size,
            num_classes=args.nb_classes,
            global_pool=args.global_pool,
            in_chans=args.in_chans,
            # patch_size=args.patch_size,
        )

    src.scale_mae.util.pos_embed.interpolate_pos_embed(model, checkpoint["model"])

    if pretrained:
        ckpt_patch_embed_weight = checkpoint["model"]["patch_embed.proj.weight"]
        if args.in_chans != ckpt_patch_embed_weight.shape[1]:
            if args.in_chans % ckpt_patch_embed_weight.shape[1] == 0:
                print(f"Rescaling pretrained patch_embed weight to fit new {in_chans=}")
                new_pe_weight = upscale_patch_embed(
                    args.in_chans,
                    ckpt_patch_embed_weight,
                )
                checkpoint["model"]["patch_embed.proj.weight"] = new_pe_weight
                if model_type == "mae":
                    # also change final decoder layer from 3*patch_size*patch_size
                    print("Rescaling final decoder layer for new number of in_chans")
                    # note: there are two decoders, for high and low frequency features
                    weight, bias = upscale_conv_decoder(
                        args.in_chans,
                        checkpoint["model"]["fcn_high.pred.7.weight"],
                        checkpoint["model"]["fcn_high.pred.7.bias"],
                    )
                    checkpoint["model"]["fcn_high.pred.7.weight"] = weight
                    checkpoint["model"]["fcn_high.pred.7.bias"] = bias

                    weight, bias = upscale_conv_decoder(
                        args.in_chans,
                        checkpoint["model"]["fcn_low.pred.7.weight"],
                        checkpoint["model"]["fcn_low.pred.7.bias"],
                    )
                    checkpoint["model"]["fcn_low.pred.7.weight"] = weight
                    checkpoint["model"]["fcn_low.pred.7.bias"] = bias
            else:
                print("Deleting pre-trained patch embedding weights")
                del checkpoint["model"]["patch_embed.proj.weight"]
                if model_type == "mae":
                    del checkpoint["model"]["fcn_low.pred.7.weight"]
                    del checkpoint["model"]["fcn_low.pred.7.bias"]
                    del checkpoint["model"]["fcn_high.pred.7.weight"]
                    del checkpoint["model"]["fcn_high.pred.7.bias"]

        checkpoint["model"]["mask_token_decoder"] = checkpoint["model"][
            "mask_token"
        ].clone()  # decoder mask token is missing in the checkpoint..
        missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)

        print(f"missing weights: {missing}")
        ux = set()
        for key in unexpected:
            if "decoder" in key:
                continue
            if "mask_token" in key:
                continue
            if "fcn" in key:
                continue
            if "fpn" in key:
                continue
            ux.add(key)
        print(f"unexpected weights: {ux}")

    del checkpoint
    torch.cuda.empty_cache()

    return model, args


def upscale_patch_embed(target_chans, weight):
    """change pre-trained 3-channel patch embed weight
    to, e.g., 12 channes
    """
    assert target_chans % weight.shape[1] == 0
    factor = target_chans // weight.shape[1]

    return torch.concat([weight] * factor, axis=1)


def upscale_decoder_pred(target_pixels, weight, bias):
    assert target_pixels % weight.shape[0] == 0
    factor = target_pixels // weight.shape[0]

    weight = torch.concat([weight] * factor)
    bias = torch.concat([bias] * factor)
    return weight, bias


def upscale_conv_decoder(target_chans, weight, bias):
    assert target_chans % weight.shape[1] == 0
    factor = target_chans // weight.shape[1]

    weight = torch.concat([weight] * factor, axis=1)
    bias = torch.concat([bias] * factor)

    return weight, bias


def get_sat_mae(
    model_type,
    pretrained=True,
    in_chans=3,
    img_size=224,
    patch_size=16,
):
    """return one of the three SatMAE model variants with or without pre-trained weights
    model_type: in ["", "temporal", "group_c"]
    returns: model and args (Namespace)
    """
    assert timm.__version__ in [
        "0.3.2",
        "0.6.12",
    ], f"timm version mismatch: {timm.__version__} != 0.3.2"

    if model_type == "temporal":
        checkpoint = torch.load(
            "/netscratch/lscheibenreif/code/low-rank-da/checkpoints/sat-mae/pretrain_fmow_temporal.pth",
            map_location=torch.device("cpu"),
        )
    elif model_type == "group_c":
        checkpoint = torch.load(
            "/netscratch/lscheibenreif/code/low-rank-da/checkpoints/sat-mae/pretrain_vit_large_group_c.pth"
        )
    else:
        # standard non temporal/spectral pre-trained checkpoint
        checkpoint = torch.load(
            "/netscratch/lscheibenreif/code/low-rank-da/checkpoints/sat-mae/fmow_pretrain.pth"
        )

    args = checkpoint["args"]
    args.model = args.model.replace("mae_", "")

    args.input_size = img_size
    args.patch_size = patch_size
    orig_patch_size = 16  # sat-mae pretrain patch size

    # args.patch_size = 16
    # args.input_size = 224
    args.in_chans = in_chans
    args.nb_classes = 1000
    args.drop_path = 0.1
    args.global_pool = False  # must be false

    if model_type == "group_c":
        assert args.in_chans == 10, "needs to be 10 to fit pretrained weights"
        args.patch_size = 8

        if len(args.grouped_bands) == 0:
            args.grouped_bands = [[0, 1, 2, 6], [3, 4, 5, 7], [8, 9]]

    if model_type == "temporal":
        model = src.sat_mae.models_vit_temporal.__dict__[args.model](
            num_classes=args.nb_classes,
            drop_path_rate=args.drop_path,
            global_pool=args.global_pool,
        )
    elif model_type == "group_c":
        model = src.sat_mae.models_vit_group_channels.__dict__[args.model](
            patch_size=args.patch_size,
            img_size=args.input_size,
            in_chans=args.in_chans,
            channel_groups=args.grouped_bands,
            num_classes=args.nb_classes,
            drop_path_rate=args.drop_path,
            global_pool=args.global_pool,
        )
    elif model_type == "mae":
        # return mae model including decoder
        model = src.sat_mae.models_mae.__dict__["MaskedAutoencoderViT"](
            patch_size=args.patch_size,
            img_size=args.input_size,
            in_chans=args.in_chans,
        )
    else:
        model = src.sat_mae.models_vit.__dict__[args.model](
            patch_size=args.patch_size,
            img_size=args.input_size,
            in_chans=args.in_chans,
            num_classes=args.nb_classes,
            drop_path_rate=args.drop_path,
            global_pool=args.global_pool,
        )

    if pretrained:
        patch_size_factor = 1
        # load weights
        ckpt_patch_embed_weight = checkpoint["model"]["patch_embed.proj.weight"]
        if args.in_chans != ckpt_patch_embed_weight.shape[1]:
            if args.in_chans % ckpt_patch_embed_weight.shape[1] == 0:
                print(f"Rescaling pretrained patch_embed weight to fit new {in_chans=}")
                new_pe_weight = upscale_patch_embed(
                    args.in_chans,
                    ckpt_patch_embed_weight,
                )
                checkpoint["model"]["patch_embed.proj.weight"] = new_pe_weight
                if model_type == "mae":
                    # also change final decoder layer from 3*patch_size*patch_size
                    print("Rescaling final decoder layer for new number of in_chans")
                    target_pix = (
                        args.in_chans * args.patch_size**2
                    )  # number of pixels per token
                    weight, bias = upscale_decoder_pred(
                        target_pix,
                        checkpoint["model"]["decoder_pred.weight"],
                        checkpoint["model"]["decoder_pred.bias"],
                    )
                    checkpoint["model"]["decoder_pred.weight"] = weight
                    checkpoint["model"]["decoder_pred.bias"] = bias
            else:
                print("Deleting pre-trained patch embedding weights")
                del checkpoint["model"]["patch_embed.proj.weight"]
                if model_type == "mae":
                    del checkpoint["model"]["decoder_pred.weight"]
                    del checkpoint["model"]["decoder_pred.bias"]
        elif (
            model.patch_embed.proj.weight.shape
            != checkpoint["model"]["patch_embed.proj.weight"].shape
        ):
            # model has different patch size than weights from pretrained checkpoint
            print(f"Resampling patch embedding for new patch size: {args.patch_size}")
            print(f"final decoder pred layer randomly initialized")
            patch_size_factor = orig_patch_size / args.patch_size
            checkpoint["model"]["patch_embed.proj.weight"] = (
                src.patch_embed.resample_patch_embed(
                    checkpoint["model"]["patch_embed.proj.weight"].cpu(),
                    model.patch_embed.proj.weight.shape[-2:],
                    interpolation="bilinear",
                    antialias=False,
                )
            )
            # if patch size changed, final decoder prediction layer (token -> pixels) does not fit anymore
            del checkpoint["model"]["decoder_pred.weight"]
            del checkpoint["model"]["decoder_pred.bias"]
        if model.pos_embed.shape != checkpoint["model"]["pos_embed"].shape:
            print(
                f"Resampling positional embeddings for new image size: {args.input_size}"
            )
            checkpoint["model"]["pos_embed"] = src.pos_embed.resample_abs_pos_embed(
                checkpoint["model"]["pos_embed"],
                new_size=(args.input_size, args.input_size),
                old_size=(224, 224),
                size_factor=patch_size_factor,
            )
            if model_type == "mae":
                checkpoint["model"]["decoder_pos_embed"] = (
                    src.pos_embed.resample_abs_pos_embed(
                        checkpoint["model"]["decoder_pos_embed"],
                        new_size=(args.input_size, args.input_size),
                        old_size=(224, 224),
                        size_factor=patch_size_factor,
                    )
                )
        missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
        print(f"missing weights: {missing}")
        ux = set()
        for key in unexpected:
            if (
                model_type != "mae" and key.startswith("decoder")
            ) or key == "mask_token":
                continue
            else:
                ux.add(key)
        print(f"unexpected weights: {ux}")

        del checkpoint
        torch.cuda.empty_cache()

    return model, args


def get_model(**hparams):
    """todo:
    * replace prithvi patch_embed and pos_embed with resampled versions of checkpoint
    * forward img_size and patch_size args to non sat_mae models
    """
    if hparams["model"] == "sat_mae":
        model, _ = get_sat_mae(
            hparams["model_type"],
            pretrained=hparams["pretrained"],
            in_chans=hparams["in_channels"],
            img_size=hparams["input_size"],
            patch_size=hparams["patch_size"],
        )
    elif hparams["model"] == "mae":
        model = get_mae(
            hparams["model_type"],
            pretrained=hparams["pretrained"],
            in_chans=hparams["in_channels"],
            img_size=hparams["input_size"],
            patch_size=hparams["patch_size"],
        )
    elif hparams["model"] == "scale_mae":
        model, _ = get_scale_mae(
            hparams["model_type"],
            pretrained=hparams["pretrained"],
            num_classes=hparams["num_classes"],
            in_chans=hparams["in_channels"],
            img_size=hparams["input_size"],
            patch_size=hparams["patch_size"],
            fixed_output_size=hparams["fixed_output_size"],
            use_mask_token=hparams["use_mask_token"],
        )
        # fix the input resolution
        model.forward = functools.partial(
            model.forward, input_res=torch.tensor([hparams["input_res"]])
        )
    elif hparams["model"] == "vit_l_imagenet":
        model = get_vit_l_imagenet(
            pretrained=hparams["pretrained"],
            num_classes=hparams["num_classes"],
            in_chans=hparams["in_channels"],
            patch_size=hparams["patch_size"],
            img_size=hparams["input_size"],
        )

    if hparams["freeze_backbone"]:
        for param in model.parameters():
            param.requires_grad = False

    if hparams["adapter"]:
        model = add_adapter(
            model,
            type=hparams["adapter_type"],
            shared=hparams["adapter_shared"],
            scale=hparams["adapter_scale"],
            hidden_dim=hparams["adapter_hidden_dim"],
            patch_embed_adapter=hparams["patch_embed_adapter"],
            adapter_trainable=hparams["adapter_trainable"],
            norm_trainable=hparams["norm_trainable"],
            only_scaler_trainable=hparams["only_scaler_trainable"],
        )
    else:
        # norm trainable but no adapter
        if hparams["norm_trainable"]:
            for n, p in model.named_parameters():
                if "norm" in n:
                    p.requires_grad = True

    if hparams["only_bias_trainable"]:
        for n, p in model.named_parameters():
            if "bias" in n:
                p.requires_grad = True

    # if hparams["patch_embed_adapter"]:
    # add_patch_embed_adapter(model, scale=hparams["patch_embed_adapter_scale"])

    if hparams["train_patch_embed"]:
        for p in model.patch_embed.parameters():
            p.requires_grad = True

    if hparams["train_all_params"]:
        for param in model.parameters():
            param.requires_grad = True

    if hparams["train_cls_mask_tokens"]:
        model.cls_token.requires_grad = True
        model.mask_token.requires_grad = True

    return model
