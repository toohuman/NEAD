# NeuroEvolution of Ant Dynamics (NEAD)

[![arXiv](https://img.shields.io/badge/paper-arxiv-red?style=for-the-badge)](https://arxiv.org)

## Authors
- Michael Crosscombe ([@toohuman](https://github.com/toohuman))
- Ilya Horiguchi ([@NeoGendaijin](https://github.com/NeoGendaijin))

## Project Overview
| Real Ants | Simulated Behaviour |
|:-:|:-:|
| <img src="video_screenshot.png" alt="Video recordings of real ants" height="300"> | <img src="sim_screenshot.png" alt="Simulated ant behaviour" height="300"> |

The purpose of this project is to evolve neural networks that can accurately reproduce realistic ant dynamics and, eventually, colony-level collective behaviours.


## Project Structure
- `WANNTool/`: Core WANN implementation and tools
- `prettyNeatWann/`: Main implementation directory containing:
  - Domain-specific environments
  - Training and testing scripts
  - Analysis tools
  - State space visualisation

## Features
- Ant trajectory analysis and state space representation
- Behavioural clustering and pattern recognition
- Neural network evolution for ant behaviour reproduction
- Colony-level behavioural analysis
- Visualisation tools for behavioural state space

## Dependencies
- Python 3.11+
- NumPy
- Pandas
- SciPy
- Scikit-learn
- Matplotlib
- Seaborn
- Gymnasium
- PyTorch

## Usage
Main scripts can be found in the `prettyNeatWann/` directory:
- `wann_train.py`: Train WANN models
- `wann_test.py`: Test trained models
- `ant_state_space.py`: Analyse ant behavioural states

## License
[To be added]

## Acknowledgements
This project builds upon the Weight Agnostic Neural Networks (WANN) implementation from the [brain-tokyo-workshop](https://github.com/google/brain-tokyo-workshop/tree/master/WANNRelease) repository by Google Research. The original WANN implementation is described in:

```bibtex
@article{wann2019,
  author = {Adam Gaier and David Ha},
  title  = {Weight Agnostic Neural Networks},
  eprint = {arXiv:1906.04358},
  url    = {https://weightagnostic.github.io},
  note   = "\url{https://weightagnostic.github.io}",
  year   = {2019}
}
```
