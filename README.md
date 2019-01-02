# Object Detection with Single Shot MultiBox Detector

## A PyTorch implementation of Single Shot MultiBox Detector Algorithm

<h2>

```diff
+ General Introduction to SSD 
```

</h2>

<p1>
Here I would like to discuss only the high-level intuition of Single Shot Multibox Detection Algorithm approach in the regards of the object detection.
By using SSD, we only need to take one single shot to detect multiple objects within the image, while regional proposal network (RPN) based approaches such as R-CNN series that need two shots, one for generating region proposals, one for detecting the object of each proposal. Thus, SSD is much faster compared with two-shot RPN-based approaches.

</p1>

<p2>

The SSD detector differs from others single shot detectors due to the usage of multiple layers that provide a finer accuracy on objects with different scales. (Each deeper layer will see bigger objects).
The SSD normally start with a VGG on Resnet pre-trained model that is converted to a fully convolution neural network. 

</p2>

<p3>

Then we attach some extra conv layers, which will actually help to handle bigger objects. The SSD architecture can in principle be used with any deep network base model.
One important point to notice is that after the image is passed on the **VGG network**, some conv layers are added producing feature maps of sizes 19x19, 10x10, 5x5, 3x3, 1x1. These, together with the 38x38 feature map produced by VGG’s conv4_3, are the feature maps which will be used to predict bounding boxes.
There the **conv4_3** is responsible to detect the **smallest objects** while the **conv11_2** is responsible for the **biggest objects.**

</p3>













