# TeacherStudents

This repo will provide the codes to 65 lightweight CNNs used as encoders for 8 encoder-decoder networks for the following paper that is currently "submitted" to an academic journal for possible publication.
"A Comparative Study of Knowledge Transfer Methods for Misaligned Urban Building Labels". The paper is archived at [https://arxiv.org/abs/2303.09064](https://arxiv.org/abs/2311.03867)

The codes for domain adaptation, knowledge distillation, and deep mutual learning will be available in this repo after the paper is published. For now, a brief explanation of the encoder-decoder networks (EDNs) and 65 CNNs is provided below:

## Encoder-decoder networks (EDNs)
### U-Net 
U-Net \cite{ronneberger2015u} is an architecture characterised by its symmetrical structure, consisting of an encoder CNN and a decoder equipped with corresponding upsampling layers. This symmetrical design is implemented to effectively retain and preserve spatial information throughout the network. It leverages skip connections to transfer and concatenate low-level features from the encoder to the corresponding decoder layers with matching dimensions. Each encoder block consists of convolutions, Rectified Linear Unit (ReLU) and max pooling layers. Similarly, the decoder blocks consist of upsampling, concatenation and convolution operations. The encoder can be replaced by other CNNs.
    
### U-Net++ 
U-Net++ \cite{zhou2019unet++} is a reiteration of U-Net with nested dense skip connections that enrich the information transported to the decoder layers and reduce the inter-encoder-decoder semantic gaps. The encoder and decoder blocks are similar to the U-Net, however, the skip connections have additional convolution layers.

### U-Net3+ 
U-Net3+ \cite{huang2020unet} is the latest reiteration of the U-Net family. Once again, its focus is on improving the skip connections. It offers a full-scale skip connection to reduce the inter-encoder-decoder and intra-decoder semantic gaps. Its decoder layer receives the feature maps of (i) the same-scale encoder layer, (ii) smaller-scale encoder layers supported by non-overlapping max pooling operations, and (iii) larger-scale decoder layers supported by bilinear interpolation. All incoming feature maps are unified with 64 filters of 3x3 size. Subsequently, a feature aggregation mechanism is employed on these 320 (64 $\times$ 5) concatenated maps of five scales. This unification reduces the number of network parameters compared to both U-Net and U-Net++, however, it takes more time to compute because of the increased concatenation operations.

### LinkNet 
LinkNet \cite{chaurasia2017linknet} has a similar architecture to U-Net, except the features transported by skip connections are added to the decoder layers. The input of the encoder layer is bypassed to the output of the decoder layer with matching dimensions. This recovers the spatial information lost in downsampling and further reduces the network parameters. The original LinkNet uses ResNet-18 CNN as its encoder, but it can be replaced by others.

### PSPNet 
PSPNet \cite{zhao2017pyramid} is different from the symmetrical EDNs above. A CNN is used to collect the feature maps that are then input to a pyramid pooling module (PPM). PPM extracts multi-scale contextual information from the features by dividing them into cells, performing pooling operations within each cell to capture features at different scales, and then concatenating these results. This enriched information is integrated into the network's decoder, improving the accuracy of semantic image segmentation by considering details and context at various levels of scale.

### FPN
FPN \cite{lin2017feature} captures multi-scale features with bottom-top and top-bottom pyramid structures. The bottom-top pathway enhances feature maps obtained from a CNN by applying lateral connections, facilitating information propagation from lower-resolution to higher-resolution feature maps. The top-bottom pathway upsamples higher-level feature maps and fuses them with the bottom-up feature maps. This fusion creates a multi-scale feature pyramid, offering semantically rich and precisely localised information at various resolutions. FPN's feature pyramid enables effective handling of objects of different sizes within images. Originally dedicated to object detection, it is useful to segment objects of different sizes with precise boundary extraction.
    
### DeepLabv3+
DeepLabv3+ \cite{chen2018encoder} is the latest reiteration of the DeepLab family. The encoder comprises a CNN responsible for gathering high-level features from the input image. To capture multi-resolution feature maps, an atrous spatial pyramid pooling (ASPP) module is integrated into the encoder, utilizing atrous convolutions (also known as dilated convolutions) with different dilation rates in parallel. The results of these parallel convolutions are combined to create a holistic feature representation, preserving both local and global contextual information. In the decoder, feature maps from the encoder are upsampled using either bilinear interpolation or transposed convolutions.
    
### MANet 
MANet \cite{fan2020manet} uses a CNN to generate multi-scale hierarchical feature maps of different scales. The maps are then integrated (generally concatenated) into multiple attention heads, each dedicated to a single scale of feature maps. These attention heads operate in parallel, allowing the network to attend to various features simultaneously. Deconvolution layers upsample the outputs of attention heads from the smallest to the input size feature maps to produce the final segmented output.




If you use the codes from the repo, we appreciate your citation to the paper as:

```json
@misc{neupane2023comparative,
      title={A Comparative Study of Knowledge Transfer Methods for Misaligned Urban Building Labels}, 
      author={Bipul Neupane and Jagannath Aryal and Abbas Rajabifard},
      year={2023},
      eprint={2311.03867},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
Acknowledgement:
The authors also express their special gratitude to Segmentation Models Pytorch and Hugging Face for their continuous support towards open and accessible AI - the extensive study would not have been possible without their commendable work.
