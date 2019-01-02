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


<h3>

```diff
+ Dealing With Scale Problem
```

</h3>

![4](https://user-images.githubusercontent.com/30608533/50616413-6b382d00-0ef9-11e9-9827-e9d884ccbaa9.png)

<p5>
  
In the above picture,  we have an image with few horses. We have divided our input image into the set of grids. Then we make couple of rectangles of different aspect ratio around those grids. Then we apply convolution in those boxes to find if there is an object or not in those grids. Here one of the black horse is closer to the camera in the image. So the rectangle we draw is unable to identify if that is horse or not because the rectangle does not have any features that are identifying to horses.

</p5>

![5](https://user-images.githubusercontent.com/30608533/50616727-c4549080-0efa-11e9-8798-3983dd1e750a.png)

<p6>

If we see the above architecture of SSD, we can see in each step after conv6 layer the size of images gets reduced substantially. Then every operation we discussed on making grids and finding objects on those grids applies in every single step of the convolution going from back to front of the network. The classifiers are applied in every single step to detect the objects too. So since the objects become smaller in each steps they gets easily identified.

  
</p6>
