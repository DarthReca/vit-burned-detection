<h1 align="center">
  Vision transformers in burned area delineation
</h1>

<div align="center">
<br />

[![Project license](https://img.shields.io/github/license/DarthReca/vit-burned-detection.svg?style=flat-square)](LICENSE)

</div>

<details open="open">
<summary>Table of Contents</summary>

- [About](#about)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
- [Usage](#usage)
- [License](#license)
- [Acknowledgements](#acknowledgements)

</details>

---

## About

Code for _Vision transformers in burned area delineation_ by Daniele Rege Cambrin, Luca Colomba and Paolo Garza.
**The code will be released once ready for use. Some parts may be missing.**

## Getting Started

### Prerequisites

To a full list of dependency look at _requirements.txt_. They can be installed with `pip install -r requirements.txt`.

## Usage

Simply run `main.py` with arguments:

- ckpt_path: path of the checkpoint to load (default: None)
- steps: list of steps to execute (default: train)
- tag: comet experiment tag (default: test)
- config_file: configuration file path (default: configs/reduced_config.yaml)
- seed: for reproducibility (default: 7)
- test_set: the name of the fold used as test set. (default: purple)

## License

This project is licensed under the **Apache 2.0 license**. See [LICENSE](LICENSE) for more information.

This repository use code from other repository. See their licenses in `licenses` folder.

## Acknowledgements

We have to thanks HPC@POLITO and SmartData@POLITO for the provided computational resources.
