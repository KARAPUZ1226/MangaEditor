import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import os

try:
    from patchmatch import patch_match
    HAS_PATCHMATCH = patch_match.patchmatch_available
except Exception as e:
    HAS_PATCHMATCH = False
    print(f"[LaMa PyTorch] PyPatchMatch not available: {e}")

def set_requires_grad(module, value):
    for param in module.parameters():
        param.requires_grad = value

def get_activation(kind='tanh'):
    if kind == 'tanh':
        return nn.Tanh()
    if kind == 'sigmoid':
        return nn.Sigmoid()
    if kind is False:
        return nn.Identity()
    raise ValueError(f'Unknown activation kind {kind}')


class FFCSE_block(nn.Module):
    def __init__(self, channels, ratio_g):
        super(FFCSE_block, self).__init__()
        in_cg = int(channels * ratio_g)
        in_cl = channels - in_cg
        r = 16

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1 = nn.Conv2d(channels, channels // r,
                               kernel_size=1, bias=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv_a2l = None if in_cl == 0 else nn.Conv2d(
            channels // r, in_cl, kernel_size=1, bias=True)
        self.conv_a2g = None if in_cg == 0 else nn.Conv2d(
            channels // r, in_cg, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x if type(x) is tuple else (x, 0)
        id_l, id_g = x

        x = id_l if type(id_g) is int else torch.cat([id_l, id_g], dim=1)
        x = self.avgpool(x)
        x = self.relu1(self.conv1(x))

        x_l = 0 if self.conv_a2l is None else id_l * \
            self.sigmoid(self.conv_a2l(x))
        x_g = 0 if self.conv_a2g is None else id_g * \
            self.sigmoid(self.conv_a2g(x))
        return x_l, x_g


class FourierUnit(nn.Module):
    def __init__(self, in_channels, out_channels, groups=1, spatial_scale_factor=None, spatial_scale_mode='bilinear',
                 spectral_pos_encoding=False, use_se=False, se_kwargs=None, ffc3d=False, fft_norm='ortho'):
        super(FourierUnit, self).__init__()
        self.groups = groups

        self.conv_layer = torch.nn.Conv2d(in_channels=in_channels * 2 + (2 if spectral_pos_encoding else 0),
                                          out_channels=out_channels * 2,
                                          kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False)
        self.bn = torch.nn.BatchNorm2d(out_channels * 2)
        self.relu = torch.nn.ReLU(inplace=True)

        self.use_se = use_se
        self.spatial_scale_factor = spatial_scale_factor
        self.spatial_scale_mode = spatial_scale_mode
        self.spectral_pos_encoding = spectral_pos_encoding
        self.ffc3d = ffc3d
        self.fft_norm = fft_norm

    def forward(self, x):
        batch = x.shape[0]

        if self.spatial_scale_factor is not None:
            orig_size = x.shape[-2:]
            x = F.interpolate(x, scale_factor=self.spatial_scale_factor, mode=self.spatial_scale_mode, align_corners=False)

        r_size = x.size()
        fft_dim = (-3, -2, -1) if self.ffc3d else (-2, -1)

        if x.dtype in (torch.float16, torch.bfloat16):
            x = x.type(torch.float32)

        ffted = torch.fft.rfftn(x, dim=fft_dim, norm=self.fft_norm)
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])

        if self.spectral_pos_encoding:
            height, width = ffted.shape[-2:]
            coords_vert = torch.linspace(0, 1, height)[None, None, :, None].expand(batch, 1, height, width).to(ffted)
            coords_hor = torch.linspace(0, 1, width)[None, None, None, :].expand(batch, 1, height, width).to(ffted)
            ffted = torch.cat((coords_vert, coords_hor, ffted), dim=1)

        if self.use_se:
            ffted = self.se(ffted)

        ffted = self.conv_layer(ffted)
        ffted = self.relu(self.bn(ffted))

        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(
            0, 1, 3, 4, 2).contiguous()
        if ffted.dtype in (torch.float16, torch.bfloat16):
            ffted = ffted.type(torch.float32)
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])

        ifft_shape_slice = x.shape[-3:] if self.ffc3d else x.shape[-2:]
        output = torch.fft.irfftn(ffted, s=ifft_shape_slice, dim=fft_dim, norm=self.fft_norm)

        if self.spatial_scale_factor is not None:
            output = F.interpolate(output, size=orig_size, mode=self.spatial_scale_mode, align_corners=False)

        return output


class SpectralTransform(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, groups=1, enable_lfu=True, **fu_kwargs):
        super(SpectralTransform, self).__init__()
        self.enable_lfu = enable_lfu
        if stride == 2:
            self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        else:
            self.downsample = nn.Identity()

        self.stride = stride
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=1, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True)
        )
        self.fu = FourierUnit(out_channels // 2, out_channels // 2, groups, **fu_kwargs)
        if self.enable_lfu:
            self.lfu = FourierUnit(out_channels // 2, out_channels // 2, groups)
        self.conv2 = torch.nn.Conv2d(out_channels // 2, out_channels, kernel_size=1, groups=groups, bias=False)

    def forward(self, x):
        x = self.downsample(x)
        x = self.conv1(x)
        output = self.fu(x)

        if self.enable_lfu:
            n, c, h, w = x.shape
            split_no = 2
            split_s = h // split_no
            xs = torch.cat(torch.split(x[:, :c // 4], split_s, dim=-2), dim=1).contiguous()
            xs = torch.cat(torch.split(xs, split_s, dim=-1), dim=1).contiguous()
            xs = self.lfu(xs)
            xs = xs.repeat(1, 1, split_no, split_no).contiguous()
        else:
            xs = 0

        output = self.conv2(x + output + xs)
        return output


class FFC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 ratio_gin, ratio_gout, stride=1, padding=0,
                 dilation=1, groups=1, bias=False, enable_lfu=True,
                 padding_type='reflect', gated=False, **spectral_kwargs):
        super(FFC, self).__init__()

        assert stride == 1 or stride == 2, "Stride should be 1 or 2."
        self.stride = stride

        in_cg = int(in_channels * ratio_gin)
        in_cl = in_channels - in_cg
        out_cg = int(out_channels * ratio_gout)
        out_cl = out_channels - out_cg

        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout
        self.global_in_num = in_cg

        module = nn.Identity if in_cl == 0 or out_cl == 0 else nn.Conv2d
        self.convl2l = module(in_cl, out_cl, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cl == 0 or out_cg == 0 else nn.Conv2d
        self.convl2g = module(in_cl, out_cg, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cg == 0 or out_cl == 0 else nn.Conv2d
        self.convg2l = module(in_cg, out_cl, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cg == 0 or out_cg == 0 else SpectralTransform
        self.convg2g = module(in_cg, out_cg, stride, 1 if groups == 1 else groups // 2, enable_lfu, **spectral_kwargs)

        self.gated = gated
        module = nn.Identity if in_cg == 0 or out_cl == 0 or not self.gated else nn.Conv2d
        self.gate = module(in_channels, 2, 1)

    def forward(self, x):
        x_l, x_g = x if type(x) is tuple else (x, 0)
        out_xl, out_xg = 0, 0

        if self.gated:
            total_input_parts = [x_l]
            if torch.is_tensor(x_g):
                total_input_parts.append(x_g)
            total_input = torch.cat(total_input_parts, dim=1)

            gates = torch.sigmoid(self.gate(total_input))
            g2l_gate, l2g_gate = gates.chunk(2, dim=1)
        else:
            g2l_gate, l2g_gate = 1, 1

        if self.ratio_gout != 1:
            out_xl = self.convl2l(x_l) + self.convg2l(x_g) * g2l_gate
        if self.ratio_gout != 0:
            out_xg = self.convl2g(x_l) * l2g_gate + self.convg2g(x_g)

        return out_xl, out_xg


class FFC_BN_ACT(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size, ratio_gin, ratio_gout,
                 stride=1, padding=0, dilation=1, groups=1, bias=False,
                 norm_layer=nn.BatchNorm2d, activation_layer=nn.Identity,
                 padding_type='reflect', enable_lfu=True, **kwargs):
        super(FFC_BN_ACT, self).__init__()
        self.ffc = FFC(in_channels, out_channels, kernel_size,
                       ratio_gin, ratio_gout, stride, padding, dilation,
                       groups, bias, enable_lfu, padding_type=padding_type, **kwargs)
        lnorm = nn.Identity if ratio_gout == 1 else norm_layer
        gnorm = nn.Identity if ratio_gout == 0 else norm_layer
        global_channels = int(out_channels * ratio_gout)
        self.bn_l = lnorm(out_channels - global_channels)
        self.bn_g = gnorm(global_channels)

        lact = nn.Identity if ratio_gout == 1 else activation_layer
        gact = nn.Identity if ratio_gout == 0 else activation_layer
        self.act_l = lact(inplace=True)
        self.act_g = gact(inplace=True)

    def forward(self, x):
        x_l, x_g = self.ffc(x)
        x_l = self.act_l(self.bn_l(x_l))
        x_g = self.act_g(self.bn_g(x_g))
        return x_l, x_g


class FFCResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation_layer=nn.ReLU, dilation=1,
                 spatial_transform_kwargs=None, inline=False, **conv_kwargs):
        super().__init__()
        self.conv1 = FFC_BN_ACT(dim, dim, kernel_size=3, padding=dilation, dilation=dilation,
                                norm_layer=norm_layer, activation_layer=activation_layer,
                                padding_type=padding_type, **conv_kwargs)
        self.conv2 = FFC_BN_ACT(dim, dim, kernel_size=3, padding=dilation, dilation=dilation,
                                norm_layer=norm_layer, activation_layer=activation_layer,
                                padding_type=padding_type, **conv_kwargs)
        self.inline = inline

    def forward(self, x):
        if self.inline:
            x_l, x_g = x[:, :-self.conv1.ffc.global_in_num], x[:, -self.conv1.ffc.global_in_num:]
        else:
            x_l, x_g = x if type(x) is tuple else (x, 0)

        id_l, id_g = x_l, x_g

        x_l, x_g = self.conv1((x_l, x_g))
        x_l, x_g = self.conv2((x_l, x_g))

        x_l, x_g = id_l + x_l, id_g + x_g
        out = x_l, x_g
        if self.inline:
            out = torch.cat(out, dim=1)
        return out


class MaskedSinusoidalPositionalEmbedding(nn.Embedding):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__(num_embeddings, embedding_dim)
        self.weight = self._init_weight(self.weight)

    @staticmethod
    def _init_weight(out: nn.Parameter):
        n_pos, dim = out.shape
        position_enc = np.array(
            [[pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)] for pos in range(n_pos)]
        )
        out.requires_grad = False
        sentinel = dim // 2 if dim % 2 == 0 else (dim // 2) + 1
        out[:, 0:sentinel] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
        out[:, sentinel:] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
        out.detach_()
        return out

    @torch.no_grad()
    def forward(self, input_ids):
        return super().forward(input_ids)


class MultiLabelEmbedding(nn.Module):
    def __init__(self, num_positions: int, embedding_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(num_positions, embedding_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight)

    def forward(self, input_ids):
        out = torch.matmul(input_ids, self.weight)
        return out


class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc=3, ndf=64, n_layers=4, norm_layer=nn.BatchNorm2d,):
        super().__init__()
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
                     nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)

            cur_model = []
            cur_model += [
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf),
                nn.LeakyReLU(0.2, True)
            ]
            sequence.append(cur_model)

        nf_prev = nf
        nf = min(nf * 2, 512)

        cur_model = []
        cur_model += [
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]
        sequence.append(cur_model)

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        for n in range(len(sequence)):
            setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))

    def get_all_activations(self, x):
        res = [x]
        for n in range(self.n_layers + 2):
            model = getattr(self, 'model' + str(n))
            res.append(model(res[-1]))
        return res[1:]

    def forward(self, x):
        act = self.get_all_activations(x)
        return act[-1], act[:-1]


class ConcatTupleLayer(nn.Module):
    def forward(self, x):
        assert isinstance(x, tuple)
        x_l, x_g = x
        assert torch.is_tensor(x_l) or torch.is_tensor(x_g)
        if not torch.is_tensor(x_g):
            return x_l
        return torch.cat(x, dim=1)


class FFCResNetGenerator(nn.Module):
    def __init__(self, input_nc=4, output_nc=3, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d,
                 padding_type='reflect', activation_layer=nn.ReLU,
                 up_norm_layer=nn.BatchNorm2d, up_activation=nn.ReLU(True),
                 init_conv_kwargs={}, downsample_conv_kwargs={}, resnet_conv_kwargs={}, spatial_transform_kwargs={},
                 add_out_act=True, max_features=1024, out_ffc=False, out_ffc_kwargs={}):
        assert (n_blocks >= 0)
        super().__init__()

        model = [nn.ReflectionPad2d(3),
                 FFC_BN_ACT(input_nc, ngf, kernel_size=7, padding=0, norm_layer=norm_layer,
                            activation_layer=activation_layer, **init_conv_kwargs)]

        for i in range(n_downsampling):
            mult = 2 ** i
            if i == n_downsampling - 1:
                cur_conv_kwargs = dict(downsample_conv_kwargs)
                cur_conv_kwargs['ratio_gout'] = resnet_conv_kwargs.get('ratio_gin', 0)
            else:
                cur_conv_kwargs = downsample_conv_kwargs
            model += [FFC_BN_ACT(min(max_features, ngf * mult),
                                 min(max_features, ngf * mult * 2),
                                 kernel_size=3, stride=2, padding=1,
                                 norm_layer=norm_layer,
                                 activation_layer=activation_layer,
                                 **cur_conv_kwargs)]

        mult = 2 ** n_downsampling
        feats_num_bottleneck = min(max_features, ngf * mult)

        for i in range(n_blocks):
            cur_resblock = FFCResnetBlock(feats_num_bottleneck, padding_type=padding_type, activation_layer=activation_layer,
                                          norm_layer=norm_layer, **resnet_conv_kwargs)
            model += [cur_resblock]

        model += [ConcatTupleLayer()]

        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(min(max_features, ngf * mult),
                                         min(max_features, int(ngf * mult / 2)),
                                         kernel_size=3, stride=2, padding=1, output_padding=1),
                      up_norm_layer(min(max_features, int(ngf * mult / 2))),
                      up_activation]

        if out_ffc:
            model += [FFCResnetBlock(ngf, padding_type=padding_type, activation_layer=activation_layer,
                                     norm_layer=norm_layer, inline=True, **out_ffc_kwargs)]

        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        if add_out_act:
            model.append(get_activation('tanh' if add_out_act is True else add_out_act))

        self.model = nn.Sequential(*model)

    def forward(self, img, mask, rel_pos=None, direct=None) -> Tensor:
        masked_img = torch.cat([img * (1 - mask), mask], dim=1)
        if rel_pos is None:
            return self.model(masked_img)
        else:
            x_l, x_g = self.model[:2](masked_img)
            x_l = x_l.to(torch.float32)
            x_l += rel_pos
            x_l += direct
            return self.model[2:]((x_l, x_g))


class MPE(nn.Module):
    def __init__(self):
        super().__init__()
        self.rel_pos_emb = MaskedSinusoidalPositionalEmbedding(num_embeddings=128, embedding_dim=64)
        self.direct_emb = MultiLabelEmbedding(num_positions=4, embedding_dim=64)
        self.alpha5 = nn.Parameter(torch.tensor(0, dtype=torch.float32), requires_grad=True)
        self.alpha6 = nn.Parameter(torch.tensor(0, dtype=torch.float32), requires_grad=True)

    def forward(self, rel_pos=None, direct=None):
        b, h, w = rel_pos.shape
        rel_pos = rel_pos.reshape(b, h * w)
        rel_pos_emb = self.rel_pos_emb(rel_pos).reshape(b, h, w, -1).permute(0, 3, 1, 2) * self.alpha5
        direct = direct.reshape(b, h * w, 4).to(torch.float32)
        direct_emb = self.direct_emb(direct).reshape(b, h, w, -1).permute(0, 3, 1, 2) * self.alpha6

        return rel_pos_emb, direct_emb


class LamaFourier:
    def __init__(self, build_discriminator=True, use_mpe=False, large_arch: bool = False) -> None:
        n_blocks = 9
        if large_arch:
            n_blocks = 18

        self.generator = FFCResNetGenerator(4, 3, add_out_act='sigmoid', 
                            n_blocks = n_blocks,
                            init_conv_kwargs={
                            'ratio_gin': 0,
                            'ratio_gout': 0,
                            'enable_lfu': False
                        }, downsample_conv_kwargs={
                            'ratio_gin': 0,
                            'ratio_gout': 0,
                            'enable_lfu': False
                        }, resnet_conv_kwargs={
                            'ratio_gin': 0.75,
                            'ratio_gout': 0.75,
                            'enable_lfu': False
                        }, 
                    )

        self.discriminator = NLayerDiscriminator() if build_discriminator else None
        self.inpaint_only = False
        if use_mpe:
            self.mpe = MPE()
        else:
            self.mpe = None

    def to(self, device):
        self.generator.to(device)
        if self.discriminator is not None:
            self.discriminator.to(device)
        if self.mpe is not None:
            self.mpe.to(device)
        return self

    def eval(self):
        self.inpaint_only = True
        self.generator.eval()
        if self.mpe is not None:
            self.mpe.eval()
        return self

    def __call__(self, img: Tensor, mask: Tensor, rel_pos=None, direct=None):
        if self.mpe is not None:
            rel_pos, _, direct = self.load_masked_position_encoding(mask[0][0].cpu().numpy())
            rel_pos = torch.LongTensor(rel_pos).unsqueeze_(0).to(img.device)
            direct = torch.LongTensor(direct).unsqueeze_(0).to(img.device)
            rel_pos, direct = self.mpe(rel_pos, direct)
        else:
            rel_pos, direct = None, None
        predicted_img = self.generator(img, mask, rel_pos, direct)

        if self.inpaint_only:
            return predicted_img * mask + (1 - mask) * img
        return predicted_img

    def load_masked_position_encoding(self, mask):
        mask = (mask * 255).astype(np.uint8)
        ones_filter = np.ones((3, 3), dtype=np.float32)
        d_filter1 = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]], dtype=np.float32)
        d_filter2 = np.array([[0, 0, 0], [1, 1, 0], [1, 1, 0]], dtype=np.float32)
        d_filter3 = np.array([[0, 1, 1], [0, 1, 1], [0, 0, 0]], dtype=np.float32)
        d_filter4 = np.array([[0, 0, 0], [0, 1, 1], [0, 1, 1]], dtype=np.float32)
        str_size = 256
        pos_num = 128

        ori_mask = mask.copy()
        ori_h, ori_w = ori_mask.shape[0:2]
        ori_mask = ori_mask / 255
        mask = cv2.resize(mask, (str_size, str_size), interpolation=cv2.INTER_AREA)
        mask[mask > 0] = 255
        h, w = mask.shape[0:2]
        mask3 = mask.copy()
        mask3 = 1. - (mask3 / 255.0)
        pos = np.zeros((h, w), dtype=np.int32)
        direct = np.zeros((h, w, 4), dtype=np.int32)
        i = 0

        if mask3.max() > 0:
            while np.sum(1 - mask3) > 0:
                i += 1
                mask3_ = cv2.filter2D(mask3, -1, ones_filter)
                mask3_[mask3_ > 0] = 1
                sub_mask = mask3_ - mask3
                pos[sub_mask == 1] = i

                m = cv2.filter2D(mask3, -1, d_filter1)
                m[m > 0] = 1
                m = m - mask3
                direct[m == 1, 0] = 1

                m = cv2.filter2D(mask3, -1, d_filter2)
                m[m > 0] = 1
                m = m - mask3
                direct[m == 1, 1] = 1

                m = cv2.filter2D(mask3, -1, d_filter3)
                m[m > 0] = 1
                m = m - mask3
                direct[m == 1, 2] = 1

                m = cv2.filter2D(mask3, -1, d_filter4)
                m[m > 0] = 1
                m = m - mask3
                direct[m == 1, 3] = 1

                mask3 = mask3_

        abs_pos = pos.copy()
        rel_pos = pos / (str_size / 2)
        rel_pos = (rel_pos * pos_num).astype(np.int32)
        rel_pos = np.clip(rel_pos, 0, pos_num - 1)

        if ori_w != w or ori_h != h:
            rel_pos = cv2.resize(rel_pos, (ori_w, ori_h), interpolation=cv2.INTER_NEAREST)
            rel_pos[ori_mask == 0] = 0
            direct = cv2.resize(direct, (ori_w, ori_h), interpolation=cv2.INTER_NEAREST)
            direct[ori_mask == 0, :] = 0

        return rel_pos, abs_pos, direct


def load_lama_mpe(model_path, device, use_mpe: bool = True, large_arch: bool = False) -> LamaFourier:
    model = LamaFourier(build_discriminator=False, use_mpe=use_mpe, large_arch=large_arch)
    sd = torch.load(model_path, map_location='cpu')
    model.generator.load_state_dict(sd['gen_state_dict'])
    if use_mpe:
        model.mpe.load_state_dict(sd['str_state_dict'])
    model.eval().to(device)
    return model


class LamaMPEPyTorchInpainter:
    def __init__(self, model_path):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"[LaMa PyTorch] Loading checkpoint {model_path} to {self.device}...")
        self.model = load_lama_mpe(model_path, self.device)
        self.model.eval()
        
        # Загружаем text segmenter (U-Net) для точного удаления только текста
        segmenter_path = os.path.join(os.path.dirname(model_path), "segmenter.onnx")
        if os.path.exists(segmenter_path):
            import onnxruntime as ort
            self.segmenter = ort.InferenceSession(segmenter_path, providers=['CPUExecutionProvider'])
            print(f"[LaMa PyTorch] Loaded text segmenter from {segmenter_path}")
        else:
            self.segmenter = None
            print(f"[LaMa PyTorch] Warning: text segmenter not found at {segmenter_path}")

    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        # BGR (H, W, 3) and mask (H, W) [255 = inpaint]
        img_original = np.copy(image)
        text_mask_raw = np.zeros_like(mask)
        y0_box, x0_box = 0, 0
        
        # === 0. Интеллектуальное детектирование текста (U-Net + Связные компоненты) ===
        gray_orig = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
        
        # 1. Запуск детектора связных компонентов (сглаживание медианным фильтром от шумов/растра)
        gray_smooth = cv2.medianBlur(gray_orig, 5)
        binary_dark = (gray_smooth < 140).astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_dark, connectivity=8)
        cc_text_mask = np.zeros_like(binary_dark)
        for i in range(1, num_labels):
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            ar = max(w / (h + 1e-5), h / (w + 1e-5))
            area = stats[i, cv2.CC_STAT_AREA]
            # Ограничиваем максимальный размер символа (w < 45, h < 45) для отсечения фолд-линий одежды и рамок
            if area > 15 and w < 45 and h < 45 and ar < 3.0:
                cc_text_mask[labels == i] = 255
        cc_text_mask[binary_dark == 0] = 0
        
        # Записываем отладочные маски
        cv2.imwrite("gray_orig_debug.png", gray_orig)
        cv2.imwrite("binary_dark_debug.png", binary_dark * 255)
        cv2.imwrite("cc_text_mask_debug.png", cc_text_mask)
        
        # 2. Запуск U-Net сегментера (если загружен)
        unet_mask = np.zeros_like(gray_orig)
        if self.segmenter is not None:
            try:
                ch, cw = img_original.shape[:2]
                print(f"[LaMa PyTorch DEBUG] Crop shape: {ch}x{cw}")
                y_indices, x_indices = np.where(mask >= 127)
                if len(y_indices) > 0:
                    y0_box, y1_box = y_indices.min(), y_indices.max() + 1
                    x0_box, x1_box = x_indices.min(), x_indices.max() + 1
                    print(f"[LaMa PyTorch DEBUG] BBox found: {y1_box-y0_box}x{x1_box-x0_box} at ({x0_box},{y0_box})")
                    bbox_gray = gray_orig[y0_box:y1_box, x0_box:x1_box]
                    bbox_resized = cv2.resize(bbox_gray, (256, 256))
                    input_blob = bbox_resized.astype(np.float32) / 255.0
                    input_blob = np.expand_dims(input_blob, axis=0)
                    input_blob = np.expand_dims(input_blob, axis=0)
                    outputs = self.segmenter.run(None, {"input": input_blob})
                    logits = outputs[0][0][0]
                    probs = 1.0 / (1.0 + np.exp(-logits))
                    mask_256 = (probs > 0.5).astype(np.uint8) * 255
                    bbox_mask = cv2.resize(mask_256, (x1_box - x0_box, y1_box - y0_box), interpolation=cv2.INTER_NEAREST)
                    unet_mask[y0_box:y1_box, x0_box:x1_box] = bbox_mask
            except Exception as e:
                print(f"[LaMa PyTorch] U-Net failed, using CC only: {e}")
                
        # 3. Объединяем результаты U-Net и связных компонентов
        combined_text = (unet_mask > 0) | (cc_text_mask > 0)
        
        # 4. Умный поиск обводки вокруг объединенного текста
        kernel = np.ones((3, 3), np.uint8)
        near_text = cv2.dilate(combined_text.astype(np.uint8) * 255, kernel, iterations=2)
        white_outline = (gray_orig > 185) & (near_text > 0)
        
        text_mask_raw = np.zeros_like(gray_orig)
        text_mask_raw[combined_text] = 255
        text_mask_raw[white_outline] = 255
        
        # 5. Слегка расширяем для сглаживания краев и полного удаления белых ореолов
        text_mask_dilated = cv2.dilate(text_mask_raw, kernel, iterations=4)
        
        # 6. Ограничиваем областью выделения пользователя
        mask_refined = np.copy(mask)
        mask_refined[text_mask_dilated == 0] = 0
        
        if np.sum(mask_refined >= 127) < 10:
            mask_refined = np.zeros_like(mask)

        # HoughLinesP и вычитание удалены, защита рамок теперь строится 
        # исключительно на финальном этапе через overlap-фильтрацию dilated_edges

        mask_original = np.copy(mask_refined)
        mask_original[mask_original < 127] = 0
        mask_original[mask_original >= 127] = 1
        mask_original_3d = mask_original[:, :, None]

        height, width, c = image.shape

        # Сохраняем отладочную маску на диск для визуальной проверки
        cv2.imwrite(f"mask_debug_{y0_box}_{x0_box}.png", (mask_original * 255).astype(np.uint8))

        # === 1. Паддинг до кратного 8 БЕЗ ресайза (используем отражение) ===
        pad_size = 8
        pad_h = (pad_size - (height % pad_size)) % pad_size
        pad_w = (pad_size - (width % pad_size)) % pad_size

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Предобработка: ОЧЕНЬ слабый bilateral filter или его отсутствие.
        # Скринтоны должны остаться видимыми для LaMa, иначе модель не сможет их восстановить.
        # Если скринтоны сильно мелкие — можно включить, но с d=5 и малыми сигмами.
        use_bilateral = False  # Поставь True, если LaMa слишком шумит на скринтонах
        if use_bilateral:
            image_smooth = cv2.bilateralFilter(image_rgb, d=5, sigmaColor=25, sigmaSpace=25)
        else:
            image_smooth = image_rgb.copy()

        if pad_h > 0 or pad_w > 0:
            image_padded = cv2.copyMakeBorder(image_smooth, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
            mask_padded = cv2.copyMakeBorder(mask_refined, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
        else:
            image_padded = image_smooth
            mask_padded = mask_refined

        img_torch = torch.from_numpy(image_padded).permute(2, 0, 1).unsqueeze_(0).float() / 255.
        mask_torch = torch.from_numpy(mask_padded).unsqueeze_(0).unsqueeze_(0).float() / 255.0
        mask_torch[mask_torch < 0.5] = 0
        mask_torch[mask_torch >= 0.5] = 1

        img_torch = img_torch.to(self.device)
        mask_torch = mask_torch.to(self.device)

        with torch.no_grad():
            img_torch *= (1 - mask_torch)
            img_inpainted_torch = self.model(img_torch, mask_torch)
            img_inpainted_torch = img_inpainted_torch.to(torch.float32)
            img_inpainted = (img_inpainted_torch.cpu().squeeze_(0).permute(1, 2, 0).numpy() * 255.).astype(np.uint8)

        img_inpainted = cv2.cvtColor(img_inpainted, cv2.COLOR_RGB2BGR)
        # Принудительно конвертируем в grayscale и обратно, чтобы убить цветной тон на ч/б манге
        gray_inpainted = cv2.cvtColor(img_inpainted, cv2.COLOR_BGR2GRAY)
        img_inpainted = cv2.cvtColor(gray_inpainted, cv2.COLOR_GRAY2BGR)

        # Обрезаем паддинг обратно
        if pad_h > 0 or pad_w > 0:
            img_inpainted = img_inpainted[0:height, 0:width]

        # === 2. FFT-based поиск периода скринтона на ОРИГИНАЛЕ ===
        # Сканируем 9 областей кропа, чтобы найти область с самой чистой текстурой без текста
        sub_size = min(256, height, width)
        half_sub = sub_size // 2
        
        # Находим центр выделения для локального анализа
        y_indices, x_indices = np.where(mask_original == 1)
        if len(y_indices) > 0:
            cy_box = int(y_indices.mean())
            cx_box = int(x_indices.mean())
        else:
            cy_box, cx_box = height // 2, width // 2
            
        best_peak_ratio = 0.0
        best_dy = 0
        best_dx = 0
        
        # Регулярная сетка по всему кропу (не привязана к координатам текста)
        grid_points = [
            (height // 4, width // 4),
            (height // 4, width // 2),
            (height // 4, 3 * width // 4),
            (height // 2, width // 4),
            (height // 2, width // 2),
            (height // 2, 3 * width // 4),
            (3 * height // 4, width // 4),
            (3 * height // 4, width // 2),
            (3 * height // 4, 3 * width // 4)
        ]
        
        for cy, cx in grid_points:
            y0_sub = max(0, cy - half_sub)
            y1_sub = min(height, cy + half_sub)
            x0_sub = max(0, cx - half_sub)
            x1_sub = min(width, cx + half_sub)
            
            # Корректируем размеры до ровного квадрата sub_size x sub_size
            if (y1_sub - y0_sub) < sub_size:
                if y0_sub == 0:
                    y1_sub = min(height, sub_size)
                else:
                    y0_sub = max(0, y1_sub - sub_size)
            if (x1_sub - x0_sub) < sub_size:
                if x0_sub == 0:
                    x1_sub = min(width, sub_size)
                else:
                    x0_sub = max(0, x1_sub - sub_size)
                    
            h_s, w_s = y1_sub - y0_sub, x1_sub - x0_sub
            if h_s < 64 or w_s < 64:
                continue
                
            # Считаем процент текста в окне
            mask_area = np.sum(mask_original[y0_sub:y1_sub, x0_sub:x1_sub] > 0)
            if mask_area > h_s * w_s * 0.15:
                continue
                
            gray_sub = gray_orig[y0_sub:y1_sub, x0_sub:x1_sub].astype(np.float32)
            mask_sub = (mask_original[y0_sub:y1_sub, x0_sub:x1_sub] == 0).astype(np.float32)
            
            # Маскируем и центрируем
            gray_sub -= gray_sub.mean()
            gray_sub *= mask_sub
            
            # FFT
            f_gray = np.fft.fft2(gray_sub)
            f_conj = np.conj(f_gray)
            power = np.fft.ifft2(f_gray * f_conj).real
            power_shifted = np.fft.fftshift(power)
            
            cy_s, cx_s = h_s // 2, w_s // 2
            search_half = 16
            y_start = max(0, cy_s - search_half)
            y_end = min(h_s, cy_s + search_half + 1)
            x_start = max(0, cx_s - search_half)
            x_end = min(w_s, cx_s + search_half + 1)
            power_shifted[y_start:y_end, x_start:x_end] = 0
            # Подавляем горизонтальные и вертикальные гармоники (линии рисунка)
            power_shifted[max(0, cy_s - 2):min(h_s, cy_s + 3), :] = 0
            power_shifted[:, max(0, cx_s - 2):min(w_s, cx_s + 3)] = 0
            
            max_idx = np.unravel_index(np.argmax(power_shifted), power_shifted.shape)
            dy = max_idx[0] - cy_s
            dx = max_idx[1] - cx_s
            
            peak_val = power_shifted[max_idx]
            power_mean = np.mean(np.abs(power_shifted))
            ratio = peak_val / (power_mean + 1e-5)
            
            if ratio > best_peak_ratio:
                best_peak_ratio = ratio
                best_dy = dy
                best_dx = dx
                
        # Если ни одно окно не подошло, берем центр без ограничений на маску
        if best_peak_ratio == 0.0:
            y0_sub = max(0, height // 2 - half_sub)
            y1_sub = min(height, height // 2 + half_sub)
            x0_sub = max(0, width // 2 - half_sub)
            x1_sub = min(width, width // 2 + half_sub)
            h_s, w_s = y1_sub - y0_sub, x1_sub - x0_sub
            gray_sub = gray_orig[y0_sub:y1_sub, x0_sub:x1_sub].astype(np.float32)
            gray_sub -= gray_sub.mean()
            gray_sub *= (mask_original[y0_sub:y1_sub, x0_sub:x1_sub] == 0)
            f_gray = np.fft.fft2(gray_sub)
            f_conj = np.conj(f_gray)
            power = np.fft.ifft2(f_gray * f_conj).real
            power_shifted = np.fft.fftshift(power)
            cy_s, cx_s = h_s // 2, w_s // 2
            search_half = 16
            power_shifted[max(0, cy_s - search_half):min(h_s, cy_s + search_half + 1),
                          max(0, cx_s - search_half):min(w_s, cx_s + search_half + 1)] = 0
            # Подавляем горизонтальные и вертикальные гармоники (линии рисунка)
            power_shifted[max(0, cy_s - 2):min(h_s, cy_s + 3), :] = 0
            power_shifted[:, max(0, cx_s - 2):min(w_s, cx_s + 3)] = 0
            
            max_idx = np.unravel_index(np.argmax(power_shifted), power_shifted.shape)
            best_dy = max_idx[0] - cy_s
            best_dx = max_idx[1] - cx_s
            best_peak_ratio = power_shifted[max_idx] / (np.mean(np.abs(power_shifted)) + 1e-5)
            
        has_screentone = best_peak_ratio > 3.0
        print(f"[LaMa PyTorch DEBUG] FFT peak: dy={best_dy}, dx={best_dx}, strength={best_peak_ratio:.2f}")
        
        # === 3. Улучшенное выделение структурных линий (без поглощения растра) ===
        # Используем ОРИГИНАЛЬНОЕ изображение для детекции линий, чтобы не потерять стертые LaMa куски рамок
        gray_lines = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
        
        # Бинаризируем темные участки
        _, binary_dark = cv2.threshold(gray_lines, 150, 255, cv2.THRESH_BINARY_INV)
        
        # Разрываем случайные мостики между точками растра
        kernel_cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        binary_separated = cv2.erode(binary_dark, kernel_cross, iterations=1)
        
        # Ищем связные компоненты строго по 4 соседям (чтобы диагонали не слипались)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_separated, connectivity=4)
        dilated_edges = np.zeros_like(gray_lines)
        for i in range(1, num_labels):
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]
            
            # Точки растра маленькие, линии - протяженные
            if area > 20 or w > 15 or h > 15:
                comp_mask = (labels == i)
                # Вычисляем процент пересечения компонента с маской букв
                overlap = np.sum(comp_mask & (text_mask_raw > 0)) / (area + 1e-5)
                
                if overlap > 0.3:
                    # Это буква текста (высокое перекрытие), исключаем ее полностью
                    continue
                else:
                    # Это рамка кадра или контур рисунка (низкое перекрытие), защищаем его без вырезания букв
                    dilated_edges[comp_mask] = 255
                
        # Возвращаем линиям исходную толщину + небольшой запас
        kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated_edges = cv2.dilate(dilated_edges, kernel_rect, iterations=3)
        cv2.imwrite("edges_debug.png", dilated_edges)

        # Оставляем mask_original полной, чтобы не пропускать буквы около линий
        mask_original_3d = mask_original[:, :, None]

        # === 4. Мгновенный локальный синтез растра (FFT Period Shift) ===
        text_mask_u8 = (mask_original * 255).astype(np.uint8)
        text_mask_dilated = cv2.dilate(text_mask_u8, np.ones((7, 7), np.uint8), iterations=2)
        
        hp_texture = None
        if has_screentone and (best_dx != 0 or best_dy != 0):
            # Извлекаем точки высокой частоты НАПРЯМУЮ из оригинала вокруг бабла
            orig_smooth = cv2.GaussianBlur(img_original, (5, 5), 0).astype(np.float32)
            hp_orig = img_original.astype(np.float32) - orig_smooth
            
            hp_texture = np.zeros_like(hp_orig)
            mask_to_fill = (text_mask_dilated > 0)
            
            # Запрещаем брать доноры из букв, чёрных линий, белых облаков и тёмных теней
            gray_orig = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
            dirty_donor_mask = (
                (text_mask_dilated > 0) | 
                (dilated_edges > 0) | 
                (gray_orig > 230) | 
                (gray_orig < 25)
            ).astype(np.float32)
            
            # 2D базис решётки скринтона (основной вектор + перпендикулярный)
            v1 = (best_dx, best_dy)
            v2 = (-best_dy, best_dx)
            
            # Поиск спиралью от ближнего к дальнему окружению бабла
            period_shifts = []
            for r in range(1, 35):
                for k1 in range(-r, r + 1):
                    for k2 in range(-r, r + 1):
                        if abs(k1) == r or abs(k2) == r:
                            sx = k1 * v1[0] + k2 * v2[0]
                            sy = k1 * v1[1] + k2 * v2[1]
                            period_shifts.append((sx, sy))
            
            for sx, sy in period_shifts:
                if not np.any(mask_to_fill):
                    break
                M = np.float32([[1, 0, sx], [0, 1, sy]])
                shifted_hp = cv2.warpAffine(hp_orig, M, (width, height), borderMode=cv2.BORDER_REFLECT)
                shifted_dirty = cv2.warpAffine(dirty_donor_mask, M, (width, height), borderMode=cv2.BORDER_CONSTANT, borderValue=1.0)
                
                # Копируем ТОЛЬКО из 100% чистых участков
                copy_map = mask_to_fill & (shifted_dirty < 0.5)
                if np.any(copy_map):
                    hp_texture[copy_map] = shifted_hp[copy_map]
                    mask_to_fill[copy_map] = False
            
            print(f"[LaMa PyTorch DEBUG] Local FFT-Period shift complete! Unfilled pixels={np.sum(mask_to_fill)}")
            print(f"[LaMa PyTorch DEBUG] hp_texture stats: min={np.min(hp_texture):.2f}, max={np.max(hp_texture):.2f}, mean_abs={np.mean(np.abs(hp_texture)):.2f}")

        # === 5. Финальное совмещение: LaMa = структура, Донор = текстура ===
        feathered_mask = cv2.GaussianBlur(mask_original_3d.astype(np.float32), (15, 15), 0)
        if len(feathered_mask.shape) == 2:
            feathered_mask = feathered_mask[:, :, None]

        img_blended = img_inpainted.astype(np.float32) * feathered_mask + img_original.astype(np.float32) * (1.0 - feathered_mask)

        if hp_texture is not None:
            # Определяем, где в ОРИГИНАЛЕ есть текстура (скринтон)
            orig_gray_f = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY).astype(np.float32)
            mean_gray = cv2.GaussianBlur(orig_gray_f, (11, 11), 0)
            variance = cv2.GaussianBlur((orig_gray_f - mean_gray)**2, (11, 11), 0)
            stddev = np.sqrt(np.clip(variance, 0, None))
            
            # stddev < 1.5 -> сплошной цвет (белый/черный), stddev > 5.5 -> скринтон
            texture_presence = np.clip((stddev - 1.5) / 4.0, 0, 1)
            texture_presence_u8 = (texture_presence * 255).astype(np.uint8)
            
            # Inpaint маски присутствия текстуры, чтобы предсказать, где она должна быть под текстом
            scale = 0.25
            small_h, small_w = max(1, int(height * scale)), max(1, int(width * scale))
            small_texture = cv2.resize(texture_presence_u8, (small_w, small_h), interpolation=cv2.INTER_AREA)
            
            # Маска инпейнта должна покрывать весь текст, иначе inpaint оставит края букв
            small_mask = cv2.resize(text_mask_dilated, (small_w, small_h), interpolation=cv2.INTER_NEAREST)
            
            small_inpainted = cv2.inpaint(small_texture, small_mask, 5, cv2.INPAINT_TELEA)
            
            texture_presence_inpainted = cv2.resize(small_inpainted, (width, height), interpolation=cv2.INTER_LINEAR)
            f_texture = (texture_presence_inpainted.astype(np.float32) / 255.0)[:, :, np.newaxis]
            
            # Дополнительно подстрахуемся яркостью от LaMa
            lama_gray_smooth = cv2.GaussianBlur(
                cv2.cvtColor(img_inpainted, cv2.COLOR_BGR2GRAY), (15, 15), 0).astype(np.float32)
            f_white = np.clip((240.0 - lama_gray_smooth) / 20.0, 0, 1)  # гаснет 220->240
            f_black = np.clip((lama_gray_smooth - 20.0) / 20.0, 0, 1)   # гаснет 40->20
            
            f_texture = f_texture * f_white[:, :, np.newaxis] * f_black[:, :, np.newaxis]
            
            clean_texture_mask = feathered_mask * f_texture
            clean_texture_mask[dilated_edges[:, :, None] > 0] = 0
            
            ans = img_blended + hp_texture * clean_texture_mask
        else:
            ans = img_blended
            
        # Восстанавливаем оригинальные линии рисунка строго вне сырой маски букв (text_mask_raw == 0),
        # чтобы вернуть резкие линии и границы кадра, но не восстановить стертый текст.
        restore_mask = (dilated_edges > 0) & (text_mask_raw == 0)
        ans[restore_mask] = img_original[restore_mask]
            
        ans = np.clip(ans, 0, 255).astype(np.uint8)
        return ans
