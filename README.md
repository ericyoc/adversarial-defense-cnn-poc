# Convolutional Neural Network (CNN) Attack and Defense

This repository contains Python code for training a Convolutional Neural Network (CNN) model on the MNIST or EMNIST datasets and evaluating its performance under adversarial attacks. The code also includes adversarial training as a defense mechanism against the attacks.

## Motivating Article
Yocam, E., Alomari, A., Gawanmeh, A., Mansoor, W. (2023). A privacy-preserving system design for digital presence protection. Computers, Materials & Continua, 75(2), 3091-3110. https://doi.org/10.32604/cmc.2023.032826

D. J. Miller, Z. Xiang and G. Kesidis, "Adversarial Learning Targeting Deep Neural Network Classification: A Comprehensive Review of Defenses Against Attacks," in Proceedings of the IEEE, vol. 108, no. 3, pp. 402-433, March 2020, doi: 10.1109/JPROC.2020.2970615. https://ieeexplore.ieee.org/document/9013065

## Adversarial Attacks

The code implements three common adversarial attacks:

1. **FGSM (Fast Gradient Sign Method)**: FGSM is a fast and simple attack that adds a small perturbation to the input image in the direction of the gradient of the loss function with respect to the input. It is an untargeted attack that aims to fool the model into making incorrect predictions.

2. **CW (Carlini-Wagner)**: The CW attack is an iterative attack that optimizes a loss function to find the smallest perturbation that can cause misclassification. It is a targeted attack, meaning it tries to make the model predict a specific target class different from the true class.

3. **PGD (Projected Gradient Descent)**: PGD is an iterative attack that performs multiple steps of gradient ascent to find adversarial examples. It projects the perturbations onto an epsilon-ball around the original input to ensure that the perturbations are bounded.

These attacks can be combined to create a compounded attack, where multiple attacks are applied sequentially to generate more robust adversarial examples. The code provides three options for compounded attacks: `fgsm_cw_attack`, `fgsm_pgd_attack`, and `cw_pgd_attack`.

The attacks implemented in this code are white-box attacks, meaning they have access to the model's architecture and parameters during the attack generation process.

## Adversarial Training Defense

The code includes adversarial training as a defense mechanism against adversarial attacks. Adversarial training involves augmenting the training dataset with adversarial examples generated using the attacks mentioned above. By training the model on both clean and adversarial examples, the model learns to be more robust and resistant to adversarial perturbations.

## Requirements

- Python version: 3.8.18
- torch: 2.2.1
- torchvision: 0.17.1
- torchattacks: 3.5.1
- numpy: 1.23.5
- tabulate: 0.9.0

## Dataset

The code supports the following datasets:

- MNIST
- EMNIST
- SVHN
- USPS
- Semeion

Make sure to have the dataset files in the appropriate directory structure on your Google Drive.

## Usage

1. Set the runtime to GPU (e.g., T4 GPU - Google Colabs) instead of CPU.
2. Mount your Google Drive and ensure that the dataset files are in the correct directory structure.
3. Select the desired compounded adversarial attack by setting the `compounded_attack_name` variable.
4. Set the desired dataset by setting the `dataset_name` variable.
5. Run the code cells in the provided order.

## Workflow

1. Train the clean CNN model on the selected dataset.
2. Evaluate the clean model's performance on the test set.
3. Perform the selected compounded adversarial attack on the clean model.
4. Evaluate the attacked model's performance on the test set.
5. Generate an adversarial dataset using the attacked model.
6. Perform adversarial training by combining the clean and adversarial datasets.
7. Evaluate the adversarially trained model's performance on the test set.
8. Perform the adversarial attack on the adversarially trained model.
9. Evaluate the performance of the adversarially trained model under attack.
10. Summarize and compare the performance metrics and misclassified examples for the clean, attacked, and adversarially trained models.

## Results

The code provides a summary of the model's performance at different stages:

- Clean model performance
- Performance under adversarial attack without defense
- Performance under adversarial attack with defense (adversarial training)

The summary includes metrics such as loss, accuracy, precision, recall, F1-score, and ROC AUC score. It also displays examples of misclassified images at each stage [CNN Results](https://github.com/ericyoc/adversarial-defense-cnn/tree/main/cnn_results).

## Note

The code assumes that the dataset files are stored on Google Drive and that Google Drive will be mounted with the default directory structure. Make sure to adjust the file paths and directory structure if necessary.

For detailed information on the code and its functionality, please refer to the code comments and the provided Python code.

**Disclaimer**
This repository is intended for educational and research purposes.

