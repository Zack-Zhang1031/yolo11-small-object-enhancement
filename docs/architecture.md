# Architecture

The project extends the YOLO11s detection graph provided by Ultralytics 8.4.118. Custom modules are registered in the active `ultralytics.nn.tasks` namespace before the upstream parser builds the network. Model YAML files ship inside the Python package and are loaded with `importlib.resources`.

## MobileViT

MobileViT is inserted after the final P5 C3k2 block. A depthwise-separable local path reduces spatial convolution cost before features are converted into patch-position token sequences. Transformer encoder layers model long-range interactions, and a depthwise-separable fusion path merges the result with the original P5 feature.

## MSCA

MSCA follows the P3 top-down fusion. Parallel 3x3, 5x5, and 7x7 depthwise convolutions capture different receptive fields. Channel and spatial gates refine the fused feature before the residual addition.

## P2 Head

The P2 path upsamples P3 and concatenates it with backbone layer 2 at stride 4. The bottom-up path then returns through P3, P4, and P5. Detect consumes layers 21, 24, 27, and 30 at strides `[4, 8, 16, 32]`.

## Tensor Input Contract

The FPN/PAN graph expects dimensions aligned to the maximum model stride. `predict_tensor` pads BCHW tensors to a stride multiple, while `auto_pad=False` provides strict validation for callers that manage preprocessing themselves.
