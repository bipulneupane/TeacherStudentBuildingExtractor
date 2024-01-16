# TeacherStudents

This repo will provide the codes to 65 lightweight CNNs used as encoders for eight encoder-decoder networks (EDNs) for the following paper that is currently "submitted" to an academic journal for possible publication.
"A Comparative Study of Knowledge Transfer Methods for Misaligned Urban Building Labels". The paper is archived at [https://arxiv.org/abs/2303.09064](https://arxiv.org/abs/2311.03867)

The codes for domain adaptation, knowledge distillation, and deep mutual learning will be available in this repo after the paper is published. For now, a brief explanation of the EDNs and 65 CNNs is provided below:

## Encoder-decoder networks (EDNs)
### U-Net 
U-Net is an architecture characterised by its symmetrical structure, consisting of an encoder CNN and a decoder equipped with corresponding upsampling layers. This symmetrical design is implemented to effectively retain and preserve spatial information throughout the network. It leverages skip connections to transfer and concatenate low-level features from the encoder to the corresponding decoder layers with matching dimensions. Each encoder block consists of convolutions, Rectified Linear Unit (ReLU) and max pooling layers. Similarly, the decoder blocks consist of upsampling, concatenation and convolution operations. The encoder can be replaced by other CNNs.
```json
Ronneberger, O., Fischer, P., & Brox, T. (2015). U-net: Convolutional networks for biomedical image segmentation. In Medical Image Computing and Computer-Assisted Intervention–MICCAI 2015: 18th International Conference, Munich, Germany, October 5-9, 2015, Proceedings, Part III 18 (pp. 234-241). Springer International Publishing.
```
    
### U-Net++ 
U-Net++ is a reiteration of U-Net with nested dense skip connections that enrich the information transported to the decoder layers and reduce the inter-encoder-decoder semantic gaps. The encoder and decoder blocks are similar to the U-Net, however, the skip connections have additional convolution layers.
```json
Zhou, Z., Siddiquee, M. M. R., Tajbakhsh, N., & Liang, J. (2019). Unet++: Redesigning skip connections to exploit multiscale features in image segmentation. IEEE transactions on medical imaging, 39(6), 1856-1867.
```

### U-Net3+ 
U-Net3+ is the latest reiteration of the U-Net family. Once again, its focus is on improving the skip connections. It offers a full-scale skip connection to reduce the inter-encoder-decoder and intra-decoder semantic gaps. Its decoder layer receives the feature maps of (i) the same-scale encoder layer, (ii) smaller-scale encoder layers supported by non-overlapping max pooling operations, and (iii) larger-scale decoder layers supported by bilinear interpolation. All incoming feature maps are unified with 64 filters of 3x3 size. Subsequently, a feature aggregation mechanism is employed on these 320 (64 $\times$ 5) concatenated maps of five scales. This unification reduces the number of network parameters compared to both U-Net and U-Net++, however, it takes more time to compute because of the increased concatenation operations.
```json
Huang, H., Lin, L., Tong, R., Hu, H., Zhang, Q., Iwamoto, Y., ... & Wu, J. (2020, May). Unet 3+: A full-scale connected unet for medical image segmentation. In ICASSP 2020-2020 IEEE international conference on acoustics, speech and signal processing (ICASSP) (pp. 1055-1059). IEEE.
```

### LinkNet 
LinkNet has a similar architecture to U-Net, except the features transported by skip connections are added to the decoder layers. The input of the encoder layer is bypassed to the output of the decoder layer with matching dimensions. This recovers the spatial information lost in downsampling and further reduces the network parameters. The original LinkNet uses ResNet-18 CNN as its encoder, but it can be replaced by others.
```json
Chaurasia, A., & Culurciello, E. (2017, December). Linknet: Exploiting encoder representations for efficient semantic segmentation. In 2017 IEEE visual communications and image processing (VCIP) (pp. 1-4). IEEE.
```

### PSPNet 
PSPNet is different from the symmetrical EDNs above. A CNN is used to collect the feature maps that are then input to a pyramid pooling module (PPM). PPM extracts multi-scale contextual information from the features by dividing them into cells, performing pooling operations within each cell to capture features at different scales, and then concatenating these results. This enriched information is integrated into the network's decoder, improving the accuracy of semantic image segmentation by considering details and context at various levels of scale.
```json
Zhao, H., Shi, J., Qi, X., Wang, X., & Jia, J. (2017). Pyramid scene parsing network. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 2881-2890).
```

### FPN
FPN captures multi-scale features with bottom-top and top-bottom pyramid structures. The bottom-top pathway enhances feature maps obtained from a CNN by applying lateral connections, facilitating information propagation from lower-resolution to higher-resolution feature maps. The top-bottom pathway upsamples higher-level feature maps and fuses them with the bottom-up feature maps. This fusion creates a multi-scale feature pyramid, offering semantically rich and precisely localised information at various resolutions. FPN's feature pyramid enables effective handling of objects of different sizes within images. Originally dedicated to object detection, it is useful to segment objects of different sizes with precise boundary extraction.
```json
Lin, T. Y., Dollár, P., Girshick, R., He, K., Hariharan, B., & Belongie, S. (2017). Feature pyramid networks for object detection. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 2117-2125).
```
    
### DeepLabv3+
DeepLabv3+ \cite{chen2018encoder} is the latest reiteration of the DeepLab family. The encoder comprises a CNN responsible for gathering high-level features from the input image. To capture multi-resolution feature maps, an atrous spatial pyramid pooling (ASPP) module is integrated into the encoder, utilizing atrous convolutions (also known as dilated convolutions) with different dilation rates in parallel. The results of these parallel convolutions are combined to create a holistic feature representation, preserving both local and global contextual information. In the decoder, feature maps from the encoder are upsampled using either bilinear interpolation or transposed convolutions.
```json
Chen, L. C., Zhu, Y., Papandreou, G., Schroff, F., & Adam, H. (2018). Encoder-decoder with atrous separable convolution for semantic image segmentation. In Proceedings of the European conference on computer vision (ECCV) (pp. 801-818).
```
    
### MANet 
MANet \cite{fan2020manet} uses a CNN to generate multi-scale hierarchical feature maps of different scales. The maps are then integrated (generally concatenated) into multiple attention heads, each dedicated to a single scale of feature maps. These attention heads operate in parallel, allowing the network to attend to various features simultaneously. Deconvolution layers upsample the outputs of attention heads from the smallest to the input size feature maps to produce the final segmented output.
```json
Fan, T., Wang, G., Li, Y., & Wang, H. (2020). Ma-net: A multi-scale attention network for liver and tumor segmentation. IEEE Access, 8, 179656-179665.
```

## Lightweight CNNs for the EDNs
### ResNet
ResNet \cite{he2016deep_resnet} is one of the earliest CNNs. Its success comes from the introduction of residual learning and skip connections into CNN that allowed the training of deeper models. They are either two-layers deep (ResNet-18, -34) or three-layers deep (ResNet - 50, -101, -152), with ResNet-18 as the lightest one.
```json

```
    
### DenseNet
DenseNet \cite{huang2017densely} employs dense blocks to establish dense connections between layers, enabling the linkage of all the layers with matching feature map sizes. DenseNet-121 is the lightest among DenseNet-121, -169, and -201, where the numbers denote the depth of the networks.
```json

```
    
### MobileNet
MobileNet \cite{howard2017mobilenets} is a lightweight CNN designed for vision tasks on mobile and embedded devices, emphasizing low latency. It leverages depthwise separable convolutions for efficiency. Its iteration, MobileNet-v2 \cite{sandler2018mobilenetv2}, introduces inverted residual blocks to its bottleneck to minimise network parameters. The third iteration, MobileNet-v3 \cite{howard2019searching_mobilenetv3}, is fine-tuned for mobile phone CPUs using network architecture search (NAS) \cite{zoph2016neural} and NetAdapt algorithms to achieve efficient design. Multiple versions of MobileNet-v3 are accessible, catering to diverse resource constraints, and offering both compact and more resource-intensive options.
```json

```

### MnasNet
MnasNet \cite{tan2019mnasnet} is a CNN tailored specifically for mobile devices, discovered through automated mobile NAS. It incorporates model latency as its main objective to identify a model that achieves a balance between precision and latency. MnasNet relies on inverted residual blocks, originally hailing from MobileNet-v2, as its fundamental building blocks. MnasNet-small is the lightest version of it.
```json

```
    
### EfficientNet
EfficientNet \cite{tan2019efficientnet} follows the structure of MnasNet, but with FLOPs (floating point operations per second) as the main rewarding parameter. This is the baseline for the EfficientNetB0, which is further scaled from B1 to B7 with added depth, width, and image resolution. EfficientNet-lite versions were introduced dedicated to mobile devices with ReLU6 activation functions and removed squeeze-and-excitation blocks. EfficientNetv2 \cite{tan2021efficientnetv2} later added Fused-MBConv convolutional blocks along with the NAS component. Similar to v1, several versions are scaled up for v2.
```json

```

### SK-ResNet
SK-ResNet is an upgrade to ResNet with Selective Kernel (SK) unit \cite{li2019selective_skresnet} replacing the large kernel convolutions in its bottleneck, and allowing adaptive selection of receptive field size. SK-ResNet-18 is its lightest version.
```json

```

### Dual path network (DPN)
DPN \cite{chen2017dualpathnetworks} blends ResNet's feature re-usage capacity with DenseNet's features exploration strategy, developing a new topology of internal connection paths. DPN-68 is its lightest version.
```json

```

### ResNeSt
ResNeSt \cite{zhang2022resnest} applies channel-wise attention to different branches of ResNet to capture cross-feature interactions and learn diverse representations. ResNeSt-18 is its lightest version.
```json

```

### GERNet
GERNet is the GPU-efficient network (GE-Net) \cite{lin2020neural_gernet} implementation on ResNet that allows a more inexpensive search for GPU-efficient networks compared to NAS. Available versions are small, medium, and large.
```json

```

### MobileOne
MobileOne \cite{vasu2023mobileone} from Apple is built upon Google's MobileNet-v1 and MobileNet-v2, dedicated to its mobile devices. They achieved a 1 ms inference in iPhone-12 time by removing the multi-branched architecture during the inference. Five versions of it are available from smallest to largest.
```json

```

### High-Resolution Net (HRNet)
HRNet \cite{wang2020deep_hrnet} is a CNN focused on preserving high-resolution features throughout the network. HRNet-18 is its lightest version.
```json

```

### MobileViT
MobileViT \cite{mehta2021mobilevit} is a lightweight vision transformer with low latency from Apple dedicated to mobile devices. It offers an alternative approach to handling global information processing using transformers, namely, treating transformers as convolutional units. It comes with small versions ranging from s, xs, and xxs.
```json

```

### Facebook-Berkeley-Net (FBNet)
FBNet \cite{wu2019fbnet} is a mobile CNN discovered through automated mobile differentiable NAS. It employs an image block with depthwise convolutions and an inverted residual structure inspired by MobileNetv2. 
```json

```

### HardCoRe-NAS
HardCoRe-NAS \cite{nayman2021hardcore} is developed to address the ``soft'' enforcement of constraint by NAS. It employs a scalable search that adheres to hard constraints throughout the search and is based on an accurate definition of the anticipated resource requirement. 
```json

```

### MixNet
MixNet \cite{tan2019mixconv} is a mobile CNN from Google Brain that proposes mixed depthwise convolution (MixConv) that is capable of mixing multiple kernel sizes in a single convolution in MobileNets. Three versions of MixNet are available: small, medium, and large.
```json

```

### TinyNet
TinyNet \cite{han2020model} is a mobile CNN from Huawei, that is inspired by the scalability of EfficientNet with three dimensions of depth, width, and image resolution. They summarise and derive several TinyNets from EfficientNetB0 based only on depth and image resolution. The size of TinyNet ranges from \textit{a} to \textit{e}.
```json

```


# Acknowledgement
The authors also express their special gratitude to Segmentation Models Pytorch and Hugging Face for their continuous support towards open and accessible AI - the extensive study would not have been possible without their commendable work.
