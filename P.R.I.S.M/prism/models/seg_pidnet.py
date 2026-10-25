"""PIDNet-S, written from scratch in plain PyTorch (no timm, no einops).

Reference: Xu, Xiong, Bhattacharyya, "PIDNet: A Real-time Semantic Segmentation Network
Inspired by PID Controllers", CVPR 2023.  The network borrows the structure of a PID
controller: a **P**roportional branch that keeps high-resolution detail, an **I**ntegral
branch that accumulates global context (and therefore over-shoots at boundaries), and a
**D**erivative branch that predicts the boundary map used to correct that over-shoot.

Modules implemented here, all faithful to the paper / reference implementation:
  BasicBlock, Bottleneck            residual units (Bottleneck expansion = 2)
  PagFM   (Pixel-attention-guided fusion)   P <- I, gated by feature similarity
  Light_Bag / Bag  (Boundary-attention-guided fusion)  P, I, D -> final feature
  DAPPM / PAPPM   (Deep/Parallel Aggregation Pyramid Pooling)   context module
  segmenthead                        BN-ReLU-3x3-BN-ReLU-1x1 head

PIDNet-S configuration used by prism: m=2, n=3, planes=32, ppm_planes=96,
head_planes=128, PAPPM + Light_Bag (exactly what the reference calls PIDNet-S).

Training here is from random init (no ImageNet-pretrained PIDNet weights are available
offline); the supervision is distillation from the SegFormer-B0/ADE20K teacher, see
`prism/training/distill_seg.py`.
"""
from __future__ import annotations

import time

import torch
import torch.nn as nn
import torch.nn.functional as F

BN_MOM = 0.1
ALIGN_CORNERS = False


# --------------------------------------------------------------------------- residual units
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, no_relu=False):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOM)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.no_relu = no_relu

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out = out + residual
        return out if self.no_relu else self.relu(out)


class Bottleneck(nn.Module):
    expansion = 2                    # PIDNet uses 2, not the ResNet 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, no_relu=True):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOM)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=BN_MOM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.no_relu = no_relu

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out = out + residual
        return out if self.no_relu else self.relu(out)


class segmenthead(nn.Module):
    def __init__(self, inplanes, interplanes, outplanes, scale_factor=None):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(inplanes, momentum=BN_MOM)
        self.conv1 = nn.Conv2d(inplanes, interplanes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(interplanes, momentum=BN_MOM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(interplanes, outplanes, 1, padding=0, bias=True)
        self.scale_factor = scale_factor

    def forward(self, x):
        x = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(x)))
        if self.scale_factor is not None:
            out = F.interpolate(out, scale_factor=self.scale_factor,
                                mode="bilinear", align_corners=ALIGN_CORNERS)
        return out


# --------------------------------------------------------------------------- fusion modules
class PagFM(nn.Module):
    """Pixel-attention-guided fusion: pull I-branch context into P, per pixel.

    A sigmoid similarity map between the (embedded) P feature and the (embedded, upsampled)
    I feature decides, pixel by pixel, how much context to trust:
        out = (1 - sim) * p + sim * i
    so detail survives where the two branches disagree.
    """

    def __init__(self, in_channels, mid_channels, after_relu=False, with_channel=False):
        super().__init__()
        self.with_channel = with_channel
        self.after_relu = after_relu
        self.f_x = nn.Sequential(nn.Conv2d(in_channels, mid_channels, 1, bias=False),
                                 nn.BatchNorm2d(mid_channels, momentum=BN_MOM))
        self.f_y = nn.Sequential(nn.Conv2d(in_channels, mid_channels, 1, bias=False),
                                 nn.BatchNorm2d(mid_channels, momentum=BN_MOM))
        if with_channel:
            self.up = nn.Sequential(nn.Conv2d(mid_channels, in_channels, 1, bias=False),
                                    nn.BatchNorm2d(in_channels, momentum=BN_MOM))
        if after_relu:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x, y):
        size = x.shape[-2:]
        if self.after_relu:
            x = self.relu(x)
            y = self.relu(y)
        y_q = F.interpolate(self.f_y(y), size=size, mode="bilinear", align_corners=ALIGN_CORNERS)
        x_k = self.f_x(x)
        if self.with_channel:
            sim = torch.sigmoid(self.up(x_k * y_q))
        else:
            sim = torch.sigmoid((x_k * y_q).sum(1, keepdim=True))
        y = F.interpolate(y, size=size, mode="bilinear", align_corners=ALIGN_CORNERS)
        return (1.0 - sim) * x + sim * y


class Light_Bag(nn.Module):
    """Boundary-attention-guided fusion (light variant used by PIDNet-S).

    The D branch's boundary logit becomes an attention map: on boundaries trust P
    (detail), inside regions trust I (context).
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_p = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                    nn.BatchNorm2d(out_channels, momentum=BN_MOM))
        self.conv_i = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                    nn.BatchNorm2d(out_channels, momentum=BN_MOM))

    def forward(self, p, i, d):
        edge_att = torch.sigmoid(d)
        p_add = self.conv_p((1.0 - edge_att) * i + p)
        i_add = self.conv_i(i + edge_att * p)
        return p_add + i_add


class Bag(nn.Module):
    """Full Bag fusion (PIDNet-M/L). Kept for completeness / ablation."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(nn.BatchNorm2d(in_channels, momentum=BN_MOM),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False))

    def forward(self, p, i, d):
        edge_att = torch.sigmoid(d)
        return self.conv(edge_att * p + (1.0 - edge_att) * i)


# --------------------------------------------------------------------------- context modules
class DAPPM(nn.Module):
    """Deep Aggregation Pyramid Pooling (cascaded), PIDNet-M/L."""

    def __init__(self, inplanes, branch_planes, outplanes):
        super().__init__()
        def _bn_relu_conv(cin, cout, k=1, pad=0):
            return nn.Sequential(nn.BatchNorm2d(cin, momentum=BN_MOM), nn.ReLU(inplace=True),
                                 nn.Conv2d(cin, cout, k, padding=pad, bias=False))
        self.scale1 = nn.Sequential(nn.AvgPool2d(5, 2, 2), _bn_relu_conv(inplanes, branch_planes))
        self.scale2 = nn.Sequential(nn.AvgPool2d(9, 4, 4), _bn_relu_conv(inplanes, branch_planes))
        self.scale3 = nn.Sequential(nn.AvgPool2d(17, 8, 8), _bn_relu_conv(inplanes, branch_planes))
        self.scale4 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                    _bn_relu_conv(inplanes, branch_planes))
        self.scale0 = _bn_relu_conv(inplanes, branch_planes)
        self.process1 = _bn_relu_conv(branch_planes, branch_planes, 3, 1)
        self.process2 = _bn_relu_conv(branch_planes, branch_planes, 3, 1)
        self.process3 = _bn_relu_conv(branch_planes, branch_planes, 3, 1)
        self.process4 = _bn_relu_conv(branch_planes, branch_planes, 3, 1)
        self.compression = _bn_relu_conv(branch_planes * 5, outplanes)
        self.shortcut = _bn_relu_conv(inplanes, outplanes)

    def forward(self, x):
        size = x.shape[-2:]
        xs = [self.scale0(x)]
        for i, sc in enumerate([self.scale1, self.scale2, self.scale3, self.scale4]):
            up = F.interpolate(sc(x), size=size, mode="bilinear", align_corners=ALIGN_CORNERS)
            proc = [self.process1, self.process2, self.process3, self.process4][i]
            xs.append(proc(up + xs[-1]))
        return self.compression(torch.cat(xs, 1)) + self.shortcut(x)


class PAPPM(nn.Module):
    """Parallel Aggregation PPM - the fast variant PIDNet-S uses.

    All pyramid levels are processed in parallel by one grouped 3x3 convolution instead
    of the DAPPM cascade, which is what makes the S model real-time.
    """

    def __init__(self, inplanes, branch_planes, outplanes):
        super().__init__()
        def _bn_relu_conv(cin, cout, k=1, pad=0, groups=1):
            return nn.Sequential(nn.BatchNorm2d(cin, momentum=BN_MOM), nn.ReLU(inplace=True),
                                 nn.Conv2d(cin, cout, k, padding=pad, groups=groups, bias=False))
        self.scale1 = nn.Sequential(nn.AvgPool2d(5, 2, 2), _bn_relu_conv(inplanes, branch_planes))
        self.scale2 = nn.Sequential(nn.AvgPool2d(9, 4, 4), _bn_relu_conv(inplanes, branch_planes))
        self.scale3 = nn.Sequential(nn.AvgPool2d(17, 8, 8), _bn_relu_conv(inplanes, branch_planes))
        self.scale4 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                    _bn_relu_conv(inplanes, branch_planes))
        self.scale0 = _bn_relu_conv(inplanes, branch_planes)
        self.scale_process = _bn_relu_conv(branch_planes * 4, branch_planes * 4, 3, 1, groups=4)
        self.compression = _bn_relu_conv(branch_planes * 5, outplanes)
        self.shortcut = _bn_relu_conv(inplanes, outplanes)

    def forward(self, x):
        size = x.shape[-2:]
        x0 = self.scale0(x)
        outs = []
        for sc in (self.scale1, self.scale2, self.scale3, self.scale4):
            up = F.interpolate(sc(x), size=size, mode="bilinear", align_corners=ALIGN_CORNERS)
            outs.append(up + x0)
        scale_out = self.scale_process(torch.cat(outs, 1))
        return self.compression(torch.cat([x0, scale_out], 1)) + self.shortcut(x)


# --------------------------------------------------------------------------- the network
class PIDNet(nn.Module):
    def __init__(self, m=2, n=3, num_classes=7, planes=32, ppm_planes=96,
                 head_planes=128, augment=True):
        super().__init__()
        self.augment = augment
        self.num_classes = num_classes
        self.cfg = dict(m=m, n=n, num_classes=num_classes, planes=planes,
                        ppm_planes=ppm_planes, head_planes=head_planes)

        # ------------------------------------------------------------ stem (shared)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, planes, 3, 2, 1), nn.BatchNorm2d(planes, momentum=BN_MOM),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, 3, 2, 1), nn.BatchNorm2d(planes, momentum=BN_MOM),
            nn.ReLU(inplace=True),
        )
        self.relu = nn.ReLU(inplace=False)

        # ------------------------------------------------------------ I branch (context)
        self.layer1 = self._make_layer(BasicBlock, planes, planes, m)
        self.layer2 = self._make_layer(BasicBlock, planes, planes * 2, m, stride=2)
        self.layer3 = self._make_layer(BasicBlock, planes * 2, planes * 4, n, stride=2)
        self.layer4 = self._make_layer(BasicBlock, planes * 4, planes * 8, n, stride=2)
        self.layer5 = self._make_layer(Bottleneck, planes * 8, planes * 8, 2, stride=2)

        # ------------------------------------------------------------ P branch (detail)
        self.compression3 = nn.Sequential(
            nn.Conv2d(planes * 4, planes * 2, 1, bias=False),
            nn.BatchNorm2d(planes * 2, momentum=BN_MOM))
        self.compression4 = nn.Sequential(
            nn.Conv2d(planes * 8, planes * 2, 1, bias=False),
            nn.BatchNorm2d(planes * 2, momentum=BN_MOM))
        self.pag3 = PagFM(planes * 2, planes)
        self.pag4 = PagFM(planes * 2, planes)
        self.layer3_ = self._make_layer(BasicBlock, planes * 2, planes * 2, m)
        self.layer4_ = self._make_layer(BasicBlock, planes * 2, planes * 2, m)
        self.layer5_ = self._make_layer(Bottleneck, planes * 2, planes * 2, 1)

        # ------------------------------------------------------------ D branch (boundary)
        if m == 2:      # PIDNet-S
            self.layer3_d = self._make_single_layer(BasicBlock, planes * 2, planes)
            self.layer4_d = self._make_layer(Bottleneck, planes, planes, 1)
            self.diff3 = nn.Sequential(
                nn.Conv2d(planes * 4, planes, 3, padding=1, bias=False),
                nn.BatchNorm2d(planes, momentum=BN_MOM))
            self.diff4 = nn.Sequential(
                nn.Conv2d(planes * 8, planes * 2, 3, padding=1, bias=False),
                nn.BatchNorm2d(planes * 2, momentum=BN_MOM))
            self.spp = PAPPM(planes * 16, ppm_planes, planes * 4)
            self.dfm = Light_Bag(planes * 4, planes * 4)
        else:           # PIDNet-M / L
            self.layer3_d = self._make_single_layer(BasicBlock, planes * 2, planes * 2)
            self.layer4_d = self._make_single_layer(BasicBlock, planes * 2, planes * 2)
            self.diff3 = nn.Sequential(
                nn.Conv2d(planes * 4, planes * 2, 3, padding=1, bias=False),
                nn.BatchNorm2d(planes * 2, momentum=BN_MOM))
            self.diff4 = nn.Sequential(
                nn.Conv2d(planes * 8, planes * 2, 3, padding=1, bias=False),
                nn.BatchNorm2d(planes * 2, momentum=BN_MOM))
            self.spp = DAPPM(planes * 16, ppm_planes, planes * 4)
            self.dfm = Bag(planes * 4, planes * 4)
        self.layer5_d = self._make_layer(Bottleneck, planes * 2, planes * 2, 1)

        # ------------------------------------------------------------ heads
        if self.augment:
            self.seghead_p = segmenthead(planes * 2, head_planes, num_classes)
            self.seghead_d = segmenthead(planes * 2, planes, 1)
        self.final_layer = segmenthead(planes * 4, head_planes, num_classes)

        self._init_weights()

    # ------------------------------------------------------------------ builders
    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion, 1, stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOM))
        layers = [block(inplanes, planes, stride, downsample)]
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes, no_relu=(i == blocks - 1)))
        return nn.Sequential(*layers)

    def _make_single_layer(self, block, inplanes, planes, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion, 1, stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOM))
        return block(inplanes, planes, stride, downsample, no_relu=True)

    def _init_weights(self):
        for mod in self.modules():
            if isinstance(mod, nn.Conv2d):
                nn.init.kaiming_normal_(mod.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(mod, nn.BatchNorm2d):
                nn.init.constant_(mod.weight, 1)
                nn.init.constant_(mod.bias, 0)

    # ------------------------------------------------------------------ forward
    def forward(self, x):
        h8, w8 = x.shape[-2] // 8, x.shape[-1] // 8

        x = self.conv1(x)                                  # 1/4
        x = self.layer1(x)                                 # 1/4
        x = self.relu(self.layer2(self.relu(x)))           # 1/8   I

        x_ = self.layer3_(x)                               # 1/8   P
        x_d = self.layer3_d(x)                             # 1/8   D

        x = self.relu(self.layer3(x))                      # 1/16  I
        x_ = self.pag3(x_, self.compression3(x))
        x_d = x_d + F.interpolate(self.diff3(x), size=(h8, w8),
                                  mode="bilinear", align_corners=ALIGN_CORNERS)
        temp_p = x_

        x = self.relu(self.layer4(x))                      # 1/32  I
        x_ = self.layer4_(self.relu(x_))
        x_d = self.layer4_d(self.relu(x_d))
        x_ = self.pag4(x_, self.compression4(x))
        x_d = x_d + F.interpolate(self.diff4(x), size=(h8, w8),
                                  mode="bilinear", align_corners=ALIGN_CORNERS)
        temp_d = x_d

        x_ = self.layer5_(self.relu(x_))                   # 1/8   P, 4*planes
        x_d = self.layer5_d(self.relu(x_d))                # 1/8   D, 4*planes
        x = F.interpolate(self.spp(self.layer5(x)), size=(h8, w8),
                          mode="bilinear", align_corners=ALIGN_CORNERS)   # 1/8 I

        out = self.final_layer(self.dfm(x_, x, x_d))       # Bag fusion -> logits @1/8

        if self.augment and self.training:
            return self.seghead_p(temp_p), out, self.seghead_d(temp_d)
        return out


def PIDNetS(num_classes: int = 7, augment: bool = True) -> PIDNet:
    """The prism student: PIDNet-S with base 32 / ppm 96 / head 128."""
    return PIDNet(m=2, n=3, num_classes=num_classes, planes=32,
                  ppm_planes=96, head_planes=128, augment=augment)


def PIDNetM(num_classes: int = 7, augment: bool = True) -> PIDNet:
    return PIDNet(m=2, n=3, num_classes=num_classes, planes=64,
                  ppm_planes=96, head_planes=128, augment=augment)


# --------------------------------------------------------------------------- profiling
def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def count_flops(model: nn.Module, shape=(1, 3, 288, 512), device="cpu") -> int:
    """MAC-based FLOPs (2 x MACs) for Conv2d / Linear, via forward hooks. No deps."""
    total = [0]

    def hook(mod, inp, out):
        if isinstance(mod, nn.Conv2d):
            oh, ow = out.shape[-2:]
            k = mod.kernel_size[0] * mod.kernel_size[1]
            total[0] += 2 * oh * ow * mod.out_channels * (mod.in_channels // mod.groups) * k
            if mod.bias is not None:
                total[0] += oh * ow * mod.out_channels
        elif isinstance(mod, nn.Linear):
            total[0] += 2 * mod.in_features * mod.out_features

    handles = [m.register_forward_hook(hook) for m in model.modules()
               if isinstance(m, (nn.Conv2d, nn.Linear))]
    was_training = model.training
    model.eval().to(device)
    model(torch.zeros(*shape, device=device))
    for h in handles:
        h.remove()
    model.train(was_training)
    return total[0]


# --------------------------------------------------------------------------- self-test
if __name__ == "__main__":
    from ..config import SEG_INPUT, N_TERRAIN

    W, H = SEG_INPUT
    net = PIDNetS(num_classes=N_TERRAIN, augment=True)
    p = count_params(net)
    fl = count_flops(net, (1, 3, H, W))
    print(f"PIDNet-S  classes={N_TERRAIN} planes=32 ppm=96 head=128")
    print(f"  params : {p:,} ({p/1e6:.2f} M)   fp16 weights {p*2/2**20:.1f} MiB")
    print(f"  FLOPs  : {fl/1e9:.2f} GFLOPs @ {W}x{H}  ({fl/2/1e9:.2f} GMACs)")
    for name, sub in [("stem+I", [net.conv1, net.layer1, net.layer2, net.layer3,
                                  net.layer4, net.layer5, net.spp]),
                      ("P", [net.layer3_, net.layer4_, net.layer5_, net.pag3, net.pag4,
                             net.compression3, net.compression4]),
                      ("D", [net.layer3_d, net.layer4_d, net.layer5_d, net.diff3, net.diff4]),
                      ("fusion+heads", [net.dfm, net.final_layer, net.seghead_p, net.seghead_d])]:
        n = sum(count_params(s) for s in sub)
        print(f"  {name:<13s} {n/1e6:5.2f} M")

    x = torch.randn(2, 3, H, W)
    net.train()
    o = net(x)
    print(f"  train() -> {len(o)} outputs: "
          + ", ".join(str(tuple(t.shape)) for t in o))
    net.eval()
    with torch.no_grad():
        o = net(x)
    print(f"  eval()  -> {tuple(o.shape)}  (1/8 stride, upsampled by the stage)")

    # latency
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = net.to(dev).eval()
    xt = torch.randn(1, 3, H, W, device=dev)
    if dev == "cuda":
        net = net.half()
        xt = xt.half()
        for _ in range(10):
            with torch.no_grad():
                net(xt)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(50):
            with torch.no_grad():
                net(xt)
        torch.cuda.synchronize()
        print(f"  GPU fp16 latency: {(time.time()-t0)/50*1000:.2f} ms/frame @ {W}x{H}")
        print(f"  peak VRAM {torch.cuda.max_memory_allocated()/2**20:.0f} MiB")
    netc = PIDNetS(N_TERRAIN, augment=False).eval()
    xc = torch.randn(1, 3, H, W)
    with torch.no_grad():
        for _ in range(3):
            netc(xc)
        t0 = time.time()
        for _ in range(10):
            netc(xc)
    print(f"  CPU fp32 latency: {(time.time()-t0)/10*1000:.1f} ms/frame @ {W}x{H}")
