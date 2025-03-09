# CamPoint :camera:

> **Paper and code will come soon...**

This repository is the official implementation of the paper:

**[\[CVPR 2025\] CamPoint: Boosting Point Cloud Segmentation with Virtual Camera](/#/)**.

> TL;DR: The ***CamPoint*** employs **Camera Visibility Feature(CVF)** to encode points as feature vector via virtual cameras, representing the visibility from multiple camera views. Mainly works include:
> - **Camera Perspective Slice Distance(CPSD)**: Identifies semantically related neighbors rather than just spatially closest points to enhance local feature aggregation.
> - **Camera Parameter Embedding(CPE)**: Integrates camera prior features into point representations to enhance global information perception.
> 
> The CamPoint achieves SOTA performance on multiple datasets (e.g., 83.3% mIoU on S3DIS and 77.7% mIoU on ScanNetV2, without any spacial strategies like voting, pre-training, or joint training), with fewer parameters, lower training costs, and faster inference speed.
>  

<div style="text-align: center;">
    <img src="./assets/framework.png" alt="framework">
</div>
