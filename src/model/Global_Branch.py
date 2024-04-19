import torch
import torch.nn as nn


def convert_to_lgconv(model, ignore_key=None):
    named_children = list(model.named_children())
    next_bn = False
    for k, m in named_children:
        if k == '':
            continue
        if ignore_key is not None and k.startswith(ignore_key):
            continue
        if isinstance(
                m, nn.Conv2d
        ) and m.kernel_size[0] == 3 and m.kernel_size[0] == m.kernel_size[1]:
            setattr(
                model, k,
                LocalGlobalConv(m)
            )
        convert_to_lgconv(m, ignore_key=None)


class LocalGlobalConv(nn.Module):
    def __init__(self, ori_conv, num_heads=8, reduction_ratio=4):
        super().__init__()
        self.conv = ori_conv
        self.global_branch = GlobalBranch(
            ori_conv.in_channels,
            ori_conv.out_channels,
            ori_conv.stride,
            num_heads,
            reduction_ratio
        )
        # init with small values so that the original pretrained features
        # would not be affected much at the beggining
        self.global_branch.norm.weight.data.normal_(0, 0.0001)
        self.global_branch.norm.bias.data.zero_()

    def forward(self, x):
        x = self.conv(x) + self.global_branch(x)
        return x


class GlobalBranch(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, num_heads=8, reduction_ratio=4):
        super().__init__()
        self.num_heads = num_heads
        mid_channels = int(in_channels / reduction_ratio)
        self.reduction = nn.Conv2d(in_channels, mid_channels, 1)
        self.nonlinear = nn.ReLU()
        self.agg_attn = nn.Conv2d(mid_channels * 2, num_heads, 1)
        self.agg_gate = nn.Softmax(2)
        self.broad_attn = nn.Conv2d(mid_channels * 2, num_heads, 1)
        self.broad_gate = nn.Sigmoid()
        self.v = nn.Conv2d(in_channels, in_channels, 1)
        self.proj = nn.Conv2d(in_channels, out_channels, 1, stride=stride)
        #self.proj = nn.Conv2d(in_channels * 2, out_channels, 1, stride=stride)
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        B, C, H, W = x.shape
        x_local = self.nonlinear(self.reduction(x))
        x_global = x_local.mean([2, 3], keepdim=True)
        x_hidden = torch.cat([x_local, x_global.expand(-1, -1, H, W)], 1)
        agg_attn = self.agg_attn(x_hidden)
        agg_attn = self.agg_gate(agg_attn.view(B, -1, H*W)).view(B, -1, H, W)  # [B, heads, H, W]
        broad_attn = self.broad_attn(x_hidden)
        broad_attn = self.broad_gate(broad_attn)  # [B, heads, H, W]
        # aggregate the global features
        v = self.v(x).view(B, self.num_heads, -1, H, W)  # [B, heads, C//heads, H, W]
        v = v * agg_attn.unsqueeze(2)    # [B, heads, C//heads, H, W]
        v = v.sum([-1, -2]).view(B, self.num_heads, -1, 1, 1) # [B, heads, C//heads, 1, 1]
        # broadcast the global features
        v = v.expand(-1, -1, -1, H, W) * broad_attn.unsqueeze(2)
        v = v.view(B, C, H, W)
        x = x + v
        #x = torch.cat([x, v.expand(-1, -1, H, W)], 1)
        x = self.proj(x)
        x = self.norm(x)
        return x
    

if __name__ == '__main__':
    
