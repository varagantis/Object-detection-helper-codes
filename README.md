# Object-detection-helper-codes

The prediction network inputs the features from the backbone network, and outputs the classification scores and offsets for each bounding box.

For each position in the  7 x 7  grid of features from the backbone, the prediction network outputs C numbers to be interpreted as classification scores over the C object categories for the bounding boxes with centers in that grid cell.

In addition, for each of the A bounding boxes at each position, the prediction network outputs offsets (4 numbers, to represent the bounding box) and a confidence score (where large positive values indicate high probability that the bounding box contains an object, and large negative values indicate low probability that the bounding box contains an object).

Collecting all of these outputs, we see that for each position in the  7 x 7  grid of features we need to output a total of 5A+C numbers, so the prediction network receives an input tensor of shape (B, 1280, 7, 7) and produces an output tensor of shape (B, 5A+C, 7, 7). We can achieve this with two 1x1 convolution layers operating on the input tensor, where the number of filters in the second layer is 5A+C.

This module outputs the prediction scores (see figure below). Each grid cell predicts  A  bounding boxes (A=2) and confidence scores for those boxes. These confidence scores reflect how confident the model is that the box contains an object. Each bounding box consists of 5 predictions: x, y, w, h, confidence. The (x, y) coordinates represent the offset from the center of the grid cell which it belongs to relative to the bounds of the grid cell and should thus be in the range (-0.5, 0.5). The width w and height h are normalised and predicted relative to the whole activation map and are thus in the range (0, 1).
