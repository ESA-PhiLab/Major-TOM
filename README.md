
![image/png](https://cdn-uploads.huggingface.co/production/uploads/6304c06eeb6d777a838eab63/BJKsLwX0GG4W3-gdf40TJ.png)

# 🗺️ Major TOM: Expandable Datasets for Earth Observation
[![HF](https://img.shields.io/badge/%F0%9F%A4%97-Datasets-yellow)](https://www.huggingface.co/Major-TOM) [![HF](https://img.shields.io/badge/%F0%9F%A4%97-Spaces_Demo-yellow)](https://huggingface.co/spaces/Major-TOM/MajorTOM-Core-Viewer) [![paper](https://img.shields.io/badge/arXiv-2402.12095-D12424)](https://www.arxiv.org/abs/2402.12095) <a href="https://colab.research.google.com/github/ESA-PhiLab/Major-TOM/blob/main/03-Filtering-in-Colab.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

A standard for curating large-scale (Terabyte-scale) EO datasets.

This organisation provides a platform for making further contributions to the project with the aim of building large and compatible Earth observation datasets.

## 📊 Available Datasets
| Dataset    | Modality            | Number of Patches | Sensing Type | Comments |
|------------|---------------------|-------------------|--------------|----------|
| [Core-S2L2A](https://huggingface.co/datasets/Major-TOM/Core-S2L2A) | Sentinel-2 Level 2A | 2,245,886 | Multi-Spectral | General-Purpose Global (about 23 TB) |
| [Core-S2L1C](https://huggingface.co/datasets/Major-TOM/Core-S2L1C) | Sentinel-2 Level 1C | 2,245,886 | Multi-Spectral | General-Purpose Global (about 23 TB) |

## 🔭 Demo
You can view samples from the `MajorTOM-Core` dataset instantly in your browser here:
https://huggingface.co/spaces/Major-TOM/MajorTOM-Core-Viewer
[![huggingface co_spaces_Major-TOM_MajorTOM-Core-Viewer (1)](https://github.com/ESA-PhiLab/Major-TOM/assets/13435425/6a82c999-23f6-42e7-ad12-1e86869d17a4)](https://huggingface.co/spaces/Major-TOM/MajorTOM-Core-Viewer)

## 📌 Open Access Manuscript
This project has been outlined in [https://arxiv.org/abs/2402.12095/](https://arxiv.org/abs/2402.12095/).
<details>
<summary>Read Abstract</summary>
  
  > Deep learning models are increasingly data-hungry, requiring significant resources to collect and compile the datasets needed to train them, with Earth Observation (EO) models being no exception. However, the landscape of datasets in EO is relatively atomised, with interoperability made difficult by diverse formats and data structures. If ever larger datasets are to be built, and duplication of effort minimised, then a shared framework that allows users to combine and access multiple datasets is needed. Here, Major TOM (Terrestrial Observation Metaset) is proposed as this extensible framework. Primarily, it consists of a geographical indexing system based on a set of grid points and a metadata structure that allows multiple datasets with different sources to be merged. Besides the specification of Major TOM as a framework, this work also presents a large, open-access dataset, MajorTOM-Core, which covers the vast majority of the Earth's land surface. This dataset provides the community with both an immediately useful resource, as well as acting as a template for future additions to the Major TOM ecosystem.
</details>

![image/jpeg](https://cdn-uploads.huggingface.co/production/uploads/6304c06eeb6d777a838eab63/dfw2xVubATDEGj9--4i1D.jpeg)

### If you found this useful for your research, please cite accordingly as:
```latex
@inproceedings{Major_TOM,
  title={Major TOM: Expandable Datasets for Earth Observation}, 
  author={Alistair Francis and Mikolaj Czerkawski},
  year={2024},
  eprint={2402.12095},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
```

Powered by [Φ-lab, European Space Agency (ESA) 🛰️](https://huggingface.co/ESA-philab)

---
## FAQ
<details>
  <summary><b>Is Major TOM just another EO dataset?</b></summary>

  Almost. Major TOM is **not a dataset**, but a project aiming to standardize some of the future EO datasets. As an example of what such a dataset could be like, **MajorTOM-Core** is released as a nearly global dataset of Sentinel-2 data.
  
  Scroll up to the 📊 **Available Datasets** section of this file to see the list of current datasets.
</details>

<details>
<summary><b>Who is going to contribute to upcoming Major TOM datasets?</b></summary>
  
  Anyone can contribute. The original authors of the Major TOM paper are already working on a few other datasets that will join the Major TOM initiative.
</details>

<details>
<summary><b>Can I join Major TOM organisation on HuggingFace?</b></summary>
  
  Anyone can join the organisation with reading rights. In order to gain contributor rights, you will need to contact one of the admins and verify who you are and how you would like to contribute (you should be allowed to contribute with any dataset that follows Major TOM standard).
</details>
