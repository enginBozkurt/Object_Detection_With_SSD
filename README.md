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

### SSD Network Architecture

![1](https://user-images.githubusercontent.com/30608533/50615335-2b6f4680-0ef5-11e9-966b-710526972251.jpg)


**Single Shot: Object localization and classification is done in single forward pass of network**

**MultiBox: Technique for bounding box regression**

**Detector: Classify the detected objects**

<p4>
  
The architecture of SSD is built based on the VGG-16 architecture. But here is a little tweak on the VGG-16, we use the set of **auxiliary convolutional layers from Conv6 layer onwards instead of fully connected layers**. The reason of using VGG-16 as foundational network is its high quality image classification and transfer learning to improve results. Using the auxiliary convolutional layers we can **extract features at multiple scales and progressively decrease the size at each following layer**. I have discussed how this works in following section. You can see the following image for VGG-16 architecture. It contains fully connected layers.

</p4>

### VGG-16 architecture
![2](https://user-images.githubusercontent.com/30608533/50616043-f0224700-0ef7-11e9-944b-15f857bfb615.png)



