# Trade Decomposition: Environmental Gains from Trade

This repository contains simplified code to generate the simulations and compute the decomposition of emissions reductions into scale, composition, and green sourcing effects, as discussed in the paper:

**"Greening Ricardo: Environmental Comparative Advantage and the Environmental Gains From Trade"**  
*Authors: Mathilde Le Moigne, Simon Lepot, Ralph Ossa, Marcos Ritel, and Dora Simon*

---

## Overview

This simplified code focuses **on the decomposition of emissions reductions** resulting from carbon taxes. The decomposition breaks down emissions reductions into three effects:

1. **Scale Effect**: Emissions reduction due to the overall contraction of production.
2. **Composition Effect**: Emissions reduction due to a shift towards greener sectors.
3. **Green Sourcing Effect**: Emissions reduction due to a shift in sourcing towards greener economies (the environmental gains from trade).

---

## Requirements

### Software and Libraries

- **Python 3.x**
- **Required Libraries**:
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `scipy`
  - `seaborn`

Install dependencies with:

```bash
pip install -r requirements.txt
```
---

## Repository Structure

```plaintext
trade_decomposition/
│
├── data/ # Input data for emissions and trade flows
│
├── results/
│
├── compute_decomposition.py     # Script to compute the decomposition and generate the article plot
├── run_simulations.py           # Simulates a counterfactual economy for increasing values of the uniform carbon tax
├── utils.py                     # Utility functions and classes
├── requirements.txt             # Python dependencies
```

---

## Citations and References

If you use this code, please cite the original paper:

**Le Moigne, M., Lepot, S., Ossa, R., Ritel, M., & Simon, D. (2024). "Greening Ricardo: Environmental Comparative Advantage and the Environmental Gains From Trade."**

---
