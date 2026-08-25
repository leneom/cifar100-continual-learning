"""PyTorch loaders for ciFAIR-10 and ciFAIR-100.

The dataset metadata below is adapted from the ciFAIR loader distributed in
ZhangLab-DeepNeuroCogLab/Integrating-Curricula-with-Replays (MIT licensed),
which in turn points to the official cvjena/ciFAIR v1.0 release.
"""

from __future__ import annotations

from torchvision.datasets import CIFAR10, CIFAR100


class CiFAIR10(CIFAR10):
    base_folder = "ciFAIR-10"
    url = "https://github.com/cvjena/cifair/releases/download/v1.0/ciFAIR-10.zip"
    filename = "ciFAIR-10.zip"
    tgz_md5 = "ca08fd390f0839693d3fc45c4e49585f"
    test_list = [["test_batch", "01290e6b622a1977a000eff13650aca2"]]


class CiFAIR100(CIFAR100):
    base_folder = "ciFAIR-100"
    url = "https://github.com/cvjena/cifair/releases/download/v1.0/ciFAIR-100.zip"
    filename = "ciFAIR-100.zip"
    tgz_md5 = "ddc236ab4b12eeb8b20b952614861a33"
    test_list = [["test", "8130dae8d6fc6a436437f0ebdb801df1"]]
