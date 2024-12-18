# NeuroEvolution of Ant Dynamics (NEAD)
# cCan you center the title and the arxiv link? AI!
[![arXiv](https://img.shields.io/badge/paper-arxiv-red?style=for-the-badge)](https://arxiv.org)

## Authors
- Michael Crosscombe ([@toohuman](https://github.com/toohuman))
- Ilya Horiguchi ([@NeoGendaijin](https://github.com/NeoGendaijin))

## Project Overview
The purpose of this project is to evolve neural networks that can accurately reproduce realistic ant dynamics and, eventually, colony-level collective behaviours.

## Project Structure
- `WANNTool/`: Core WANN implementation and tools
- `prettyNeatWann/`: Main implementation directory containing:
  - Domain-specific environments
  - Training and testing scripts
  - Analysis tools
  - State space visualization

## Features
- Ant trajectory analysis and state space representation
- Behavioral clustering and pattern recognition
- Neural network evolution for ant behavior reproduction
- Colony-level behavioral analysis
- Visualization tools for behavioral state space

## Dependencies
- Python 3.11+
- NumPy
- Pandas
- SciPy
- Scikit-learn
- Matplotlib
- Seaborn
- OpenAI Gym
- PyTorch

## Usage
Main scripts can be found in the `prettyNeatWann/` directory:
- `wann_train.py`: Train WANN models
- `wann_test.py`: Test trained models
- `ant_state_space.py`: Analyze ant behavioral states

## License
[To be added]
