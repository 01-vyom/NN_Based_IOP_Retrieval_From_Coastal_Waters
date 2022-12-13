# Neural Network Based Retrieval of Inherent Optical Properties (IOPs) Of Coastal Waters of Oceans
## InGARSS 2021: IEEE International India Geoscience and Remote Sensing Symposium

### [[Paper]](https://doi.org/10.1109/InGARSS51564.2021.9792013) | [[Slides]](https://docs.google.com/presentation/d/1hkdjKePRyWepQvj3BGB2BE_Gf6-HZAL7T6SP8YEHfr0/edit?usp=sharing)

[Vyom Pathak](https://www.linkedin.com/in/01-vyom/)<sup>1</sup> | [Brijesh Bhatt](https://scholar.google.com/citations?user=aEkOFcUAAAAJ)<sup>1</sup> | [Arvind Sahay](https://scholar.google.com/citations?user=WBD49gwAAAAJ)<sup>2</sup> | [Mini Raman](https://scholar.google.com/citations?user=FAJZ1qsAAAAJ)<sup>2</sup>

[Dharmsinh Desai University, Nadiad](https://ddu.ac.in)<sup>1</sup> | [Indian Space Research Organization, Ahmedabad](https://www.isro.gov.in)<sup>2</sup>

<details>
  <summary>Abstract</summary>

  Inherent optical properties (IOPs) of the coastal oceans are modulated independently by the in-water optical constituents, which cause variations in the water leaving radiances or re-mote sensing reflectances. Accurate determination of IOPs and the optical constituents from water-leaving radiances or reflectances using conventional empirical ratio approaches fail in the coastal oceans. Alternate non-parametric approaches such as neural network (NN) based approaches can be developed to derive the parameters of interest using training datasets. Further NN based approaches can deal with the non-linearity of functional dependence between optical constituents and the IOPs.To retrieve the IOPs, earlier NN models used Levenberg-Marquardt with Bayesian Regularization as an optimizer for learning the weights of the model, which has a slow learning rate. Moreover, with low-resource datasets while retrieving IOPs till the third level, the probability of error propagation becomes high. To overcome these two problems, we present a Modified Neural Network (MNN) algorithm (modification of NN model [Ioannou et. al.](https://doi.org/10.1016/j.rse.2013.02.015) to retrieve Inherent Optical Properties (IOPs) of ocean waters, in which three Neural Networks (NN) were developed in parallel. Our method is based on the approach where we use the Adam optimizer, instead of the Levenberg-Marquardt since it has a faster training time. Also, the error propagation is observed to be very less even with low-resource data while retrieving IOPs at the third level, with a decent R 2 score.Results of the MNN algorithm indicate that MNN retrieves IOPs with an R 2 = 0.99 between measured and predicted values for b bp (443) and R 2 = 0.99 for a pg (443) at Level 1. Level-2 products give R 2 = 0.98 and R 2 = 0.99 between measured and predicted values for a pg (443) and a dg (443) respectively. Similarly Level-3 products give R 2 = 0.97 and R 2 = 0.51 between measured and predicted values for a g (443) and a d (443) respectively. The algorithm retrieves better R 2 score for all parameters except a d (443) compared to Semi Analytical Algorithm, Quasi Analytical Algorithm and NN algorithm by [Ioannou et. al.](https://doi.org/10.1016/j.rse.2013.02.015). The new technique has the advantage of faster convergence and better generalization capacity for deriving IOPs from complex waters. The new algorithm is also able to separate gelbstoff and detrital absorption.
</details>


If you find this work useful, please cite this work using the following BibTeX:

```bibtex
@inproceedings{pathak2021neural,
  title={Neural Network Based Retrieval of Inherent Optical Properties (IOPs) Of Coastal Waters of Oceans},
  author={Pathak, Vyom and Bhatt, Brijesh and Sahay, Arvind and Raman, Mini},
  booktitle={2021 IEEE International India Geoscience and Remote Sensing Symposium (InGARSS)},
  pages={285--288},
  year={2021},
  organization={IEEE}
}
```

## Setup

### System & Requirements

- Linux OS
- Python-3.6
- TensorFlow-2.2.0

### Setting up repository

  ```shell
  git clone https://github.com/01-vyom/NN_Based_IOP_Retrieval_From_Coastal_Waters.git
  python -m venv nn_iop_env
  source $PWD/nn_iop/bin/activate
  ```

### Installing Dependencies

Change directory to the root of the repository.

  ```shell
  pip install --upgrade pip
  pip install -r requirements.txt
  ```

## Running Code

Change directory to the root of the repository.

### Training, Evaluation and Inference

To train the model in the paper, run this command:

```shell
python ./src/model.py
```

Note:

- The training data is stored in `./data` directory.
- The results are stored in `./results` directory.
- The model is stored in `./model` directory.
## Results

Our algorithm achieves the following performance:

| Technique name | b_bp(443) | a_pg(443) | a_pg(443) | a_dg(443) | a_g(443) | a_d(443) |
| --------------------------------------- | ---------------- | ---------------- | ---------------- | ---------------- | ---------------- | ---------------- |
| Semi Analytical Algorithm               | 0.99             | 0.96             | 0.97             | 0.96             | do not exist | do not exist |
| Quasi Analytical Algorithm              | 0.98             | 0.93             | 0.98             | 0.93             | do not exist | do not exist |
| NN algorithm by [Ioannou et. al.](https://doi.org/10.1016/j.rse.2013.02.015)                     | 0.99             | 0.92             | 0.98             | 0.93             | 0.92             | 0.89             |
| Proposed Modified-NN algorithm                    | **0.99**             | **0.98**             | **0.99**             | **0.98**             | **0.97**             | **0.51**             |

## Acknowledgement

This work was supported by the Indian Space Research Organization (ISRO), Ahmedabad, India.

Licensed under the [MIT License](LICENSE.md).
