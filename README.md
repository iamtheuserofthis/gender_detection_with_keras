### Flatten Layer Vs GlobalAveragePooling2D
Both Flatten and GlobalAveragePooling2D are valid options. So is GlobalMaxPooling2D.
Flatten will result in a larger Dense layer afterwards, which is more expensive and may result in worse overfitting. 
But if you have lots of data, it might also perform better.
As usual, it depends completely on your problem.

__Effect On Dimensions__
H X W X D tensor, GAP will average the H X W features into a single number and reduce the tensor into a 1 X 1 X D tensor.


## Layer Architecture
|Layers|N(Layers)|
|------|---|
|Xception Model (Feature Extraction)|71|
|GlobalAveragePooling2D|1|
|Dense 256 RELU units|1|
|Dense 256 RELU units|1|
|Dense n_classes UNITS Softmax Activation|1|

