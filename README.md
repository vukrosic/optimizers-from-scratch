# Optimizers from scratch

Code for the video **Why Adam Beats SGD — Every Optimizer From Scratch**.
Seven optimizers, each written as a few lines of numpy, then raced on the same network.
Python 3.9+ and numpy. No deep-learning library: the network and its backward pass are in `net.py`.

```bash
python3 tests.py      # checks every step, about 5 seconds
./run_all.sh          # runs all eight, about two minutes; expected output is in demo-output.txt
```

| File | Optimizer | Year | What the run shows |
| --- | --- | --- | --- |
| `step1_gradient_descent.py` | Gradient descent | 1847 | A valley 100 times steeper across than along. Learning rate 0.001: loss still 6.76 after 1,000 steps. 0.019: bottom in 223 steps, bouncing. 0.021: blows up. |
| `step2_sgd.py` | SGD | 1951 | Same number of pictures looked at. 10 exact steps: loss 2.06, 19.2 % of test digits. 405 steps of 32 pictures: loss 0.097, 93.4 %. |
| `step3_momentum.py` | Momentum | 1964 | Same valley, same learning rate: 223 steps become 83. Learning rates that reach the bottom, out of 101: 14 → 39. |
| `step4_rmsprop.py` | RMSProp | 2012 | Every weight gets its own step size. In the digit network a large gradient (top 1 %) is 22,372 times a small one (bottom 1 %). |
| `step5_adam.py` | Adam | 2014 | Momentum and RMSProp together, plus the start-up fix (first steps 3.16, 4.25, 4.95 … become 1.00, 1.00, 1.00). Working learning rates out of 101: 54. |
| `step6_adamw.py` | AdamW | 2017 | Three weights with gradient sizes 1, 0.1, 0.01. Decay inside the gradient leaves 0.80, 0.17, 0.02. AdamW leaves 0.80, 0.80, 0.80. With 84 of 300 labels wrong: 67.0 % (no decay), 72.4 % (old way), 75.0 % (AdamW). |
| `step7_muon.py` | Muon | 2024 | A gradient matrix has 6 of 32 directions above a tenth of the strongest. After five Newton–Schulz rounds: 27 of 32. |
| `step8_race.py` | The race | | 5 optimizers × 17 learning rates × 5 starting weights = 425 trainings. |

The race, 400 steps of 32 pictures, best learning rate for each (median of five starting weights):

| Optimizer | Best rate | Final loss | Under 0.1 at | Test digits | Rates that work |
| --- | --- | --- | --- | --- | --- |
| SGD | 0.1 | 0.083 | step 350 | 92.2 % | 1 of 17 |
| Momentum | 0.0178 | 0.059 | step 340 | 94.8 % | 3 of 17 |
| RMSProp | 0.00316 | 0.064 | step 290 | 93.2 % | 4 of 17 |
| Adam | 0.01 | 0.051 | step 300 | 94.6 % | 4 of 17 |
| Muon | 0.00562 | 0.026 | step 190 | 96.0 % | 4 of 17 |

With a library's default learning rates and no search: SGD at 0.01 reads 82.0 %, Adam at 0.001 reads 94.2 %.

`net.py` holds the network (64 pixels in, five hidden layers of 32, 10 scores out: 6,634 weights), backpropagation and the
training loop; an optimizer is any object with `step(params, grads)`. `data.py` loads the digits.
`values.py` saves the numbers and curves the video draws.

## Exercise

In `net.py`, change `SIZES = [64, 32, 32, 32, 32, 32, 10]` to `SIZES = [64, 32, 32, 10]` (two hidden layers) and run
`python3 step8_race.py`. Does the order change?

Measured: Muon 0.014, momentum 0.037, Adam 0.041, RMSProp 0.045, SGD 0.059. Muon stays first and momentum passes Adam.
Every optimizer does better on the shallower network, and SGD's best learning rate moves from 0.1 to 0.316.
At the default rates SGD reads 49.4 % and Adam 92.4 %.

## What is simplified

- **Data.** scikit-learn's 8 × 8 digits, 1,297 for training and 500 for testing, so 425 trainings run in about a minute.
- **No schedule.** Every run uses one fixed learning rate: no warm-up, no decay.
- **Momentum.** The plain heavy-ball form with 0.9. Nesterov's variant is left out (Muon uses it internally).
- **Muon.** The core rule only: momentum 0.95, five Newton–Schulz rounds with the reference coefficients
  (3.4445, −4.7750, 2.0315), update scaled by √(rows / columns) when that is above 1. Bias vectors use Adam at 0.01.
- **"Works".** A learning rate works when the typical run (median of five starting weights) ends under a loss of 0.1.
- **Scale.** On this small network, tuned momentum is close to Adam. On transformer language models the gap between SGD and
  Adam is much larger.

## Data

`digits.csv.gz` is the test part of the UCI "Optical Recognition of Handwritten Digits" data set
(E. Alpaydin and C. Kaynak, 1998, CC BY 4.0), as shipped with scikit-learn.

## Licence

MIT for the code.
