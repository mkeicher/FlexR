# FlexR: Few-shot Classification with Language Embeddings for Structured Reporting of Chest X-rays
[![arxiv](https://img.shields.io/badge/arXiv-2203.15723-red)](https://arxiv.org/abs/2203.15723)
[![MIDL 2023](https://img.shields.io/badge/MIDL-2023-orange)](https://2023.midl.io/papers/p162)

This is the official repository for the paper [FlexR: Few-shot Classification with Language Embeddings for Structured Reporting of Chest X-rays](https://openreview.net/pdf?id=wiN5LQThnIV), which is presented at [MIDL 2023](https://2023.midl.io/).

**Authors**: [Matthias Keicher][mk], [Kamilia Zaripova][kz], [Tobias Czempiel][tc], [Kristina Mach][km], [Ashkan Khakzar][ak], [Nassir Navab][nn]

**Group**: [Vision-Language](https://github.com/CAMP-ViL) @ [CAMP](https://www.cs.cit.tum.de/camp/)

[mk]:https://www.cs.cit.tum.de/camp/members/matthias-keicher/
[kz]:https://www.cs.cit.tum.de/camp/members/kamilia-zaripova/
[tc]:https://www.cs.cit.tum.de/camp/members/tobias-czempiel/
[km]:https://scholar.google.com/citations?user=nvMY9T0AAAAJ&hl=en
[ak]:https://ashk-on.github.io/
[nn]:https://www.cs.cit.tum.de/camp/members/cv-nassir-navab/nassir-navab/

### Abstract
The automation of chest X-ray reporting has garnered significant interest due to the time-consuming nature of the task. However, the clinical accuracy of free-text reports has proven challenging to quantify using natural language processing metrics, given the complexity of medical information, the variety of writing styles, and the potential for typos and inconsistencies. Structured reporting with standardized reports, conversely, can provide consistency and formalize the evaluation of clinical correctness. However, high-quality annota- tions of standardized reports are scarce. Therefore, we propose a data-efficient method based on contrastive pretraining with free-text radiology reports to predict clinical findings as defined by structured reporting templates. The method can be used to fill such templates for generating standardized reports.

### Method

FlexR is a method for fine-tuning vision-language models when zero-shot performance proves insufficient. It builds on zero-shot classification by initializing class embeddings using their textual descriptions and subsequently optimizes these embeddings with a limited number of annotated samples. This method's versatility allows for applications in the classification of fine-grained clinical findings such as disease grading and localization as well as the prediction of rare findings in long-tailed class distributions.

Applied to the prediction of clinical findings defined by sentences in standardized reports, the method consists of the following steps:
1. Contrastive language-image pretraining (CLIP) on a dataset of radiology images and free-text reports
2. Encode possible clinical findings of the structured report into language embeddings (zero-shot initialization)
3. Fine-tune these language embeddings by optimizing cosine similarity with image embeddings using the LogSumExpSign-Loss designed for long-tailed distributions

![Graphical Abstract](pics/graphical_abstract.png)


### Citation

```
@inproceedings{keicher2023flexr,
  title={FlexR: Few-shot Classification with Language Embeddings for Structured Reporting of Chest X-rays},
  author={Keicher, Matthias and Zaripova, Kamilia and Czempiel, Tobias and Mach, Kristina and Khakzar, Ashkan and Navab, Nassir},
  booktitle={Medical Imaging with Deep Learning},
  year={2023}
}
```

### Instructions

***Code coming soon..***