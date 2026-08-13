# Variant Comparison

| Variant | Global Context | Multi-Scale Attention | Detect Inputs | Parameters |
|---|---|---|---|---:|
| Baseline | C2PSA | Standard FPN/PAN | P3, P4, P5 | 9.432M |
| MobileViT | MobileViT at P5 | Standard FPN/PAN | P3, P4, P5 | 10.634M |
| MSCA | C2PSA | MSCA after P3 fusion | P3, P4, P5 | 9.518M |
| MobileViT-MSCA | MobileViT at P5 | MSCA after P3 fusion | P3, P4, P5 | 10.720M |
| MobileViT-MSCA-P2 | MobileViT at P5 | MSCA after P3 fusion | P2, P3, P4, P5 | 10.866M |

Depthwise-separable local and fusion paths keep the MobileViT variant within 13% of the baseline parameter count.
