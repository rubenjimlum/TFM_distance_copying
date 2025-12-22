# Distance-based copying of machine learning classifiers
This repository contains the code generated during the Master's Thesis titled: "Distance-based copying of machine learning classifiers", developed in partial fulfillment of the requirements for the degree of MSc in Fundamental Principles of Data Science of the University of Barcelona.

- **Author**: Rubén Jiménez Lumbreras
- **Program**: MSc in Fundamental Principles of Data Science
- **Institution**: University of Barcelona
- **Advisor**: Prof. Dr. Oriol Pujol Vila

## Abstract
Copying machine learning black box classifiers is a key framework that allows practitioners to upgrade their old models, enriching them with new properties, changing their architectures or adapting them to comply with the current AI legislations. Thanks to the copying techniques and assumptions, these improvements can be done even in settings where retraining the original system from scratch is not possible, due to resource, protocol or availability constraints. In this work, we propose the use of signed distances to the decision boundary as a replacement of the black box hard labels used to build the copies, and introduce two different algorithms to compute these distances. In addition, we observe that distance-based copying could behave as a model-agnostic regularization technique and develop a flexible framework to reduce the generalization error of the copies. Then, we validate these proposals through a series of experiments on synthetic datasets and real problems. Results show that distance-based copying is successful across multiple relevant settings and evaluation metrics. Furthermore, results also validate the quality of the predicted distances and their potential as uncertainty measures.

## Contact
Feel free to contact me to discuss any issues, questions or comments.

- Email: [jimenezlum.ruben@gmail.com](mailto:jimenezlum.ruben@gmail.com)
- Github: [rubenjimlum](https://github.com/rubenjimlum)

## Citations
If you find it useful, please cite this thesis using the following references:

### **Written Report**
```bibtex
@masterthesis{DistanceCopyingJimenezRuben2026Report,
  author       = {Rubén Jiménez Lumbreras},
  title        = {Distance-based copying of machine learning classifiers},
  year         = {2026},
  institution  = {University of Barcelona},
  url          = {https://github.com/rubenjimlum/TFM_distance_copying/blob/main/distance_copying_report.pdf}
}
```

### **Code**
Use the `CITATION.cff` file or the following BibTeX reference:
```bibtex
@misc{DistanceCopyingJimenezRuben2026Code,
  title={TFM_distance_copying},
  url={https://github.com/rubenjimlum/TFM_distance_copying},
  author={Rubén Jiménez Lumbreras},
  year={2026}
}
```