# pinn_feynman.balancing

Weights of the loss terms, $\mathcal L_{train}=\lambda_{data}\mathcal L_{data}+\lambda_{DE}\mathcal L_{DE}$ (Boni eq. 6.13).

## `relobralo.py` — `ReLoBRaLo(n_losses, temperature, alpha, expected_rho, history, seed, scales=None)`

**`scales`:** constant factors $c_i$ in the sum, total $=\sum_i \lambda_i c_i L_i$ (config `balancing.scales: [c_data, c_PDE]`, default all 1). The weights are computed on the unscaled $L_i$; they only use ratios of the same term, $L_i(t)/L_i(t')$, so they are also the weights of the scaled terms (tested): the only change is the factor $c_i$. `coefficients()` returns $\lambda_i c_i$ of the last step. The history keeps the unscaled `data_loss` and `physics_loss`; `train_loss` is the scaled total. **`scale_epochs: [first, last]`:** the scales act only in those epochs (1-based, inclusive; the trainer calls `set_epoch` at the start of each epoch).

*Relative Loss Balancing with Random Lookback*: R. Bischof, M. Kraus, arXiv:2110.09813, eq. (11); Boni §4.3, eq. (4.9)–(4.14).
$$\lambda^{bal}_i(t,t')=m\,\mathrm{softmax}_i\!\Big(\frac{\mathcal L_i(t)}{\mathcal T\,\mathcal L_i(t')}\Big),\qquad
\lambda^{hist}_i(t)=\rho\,\lambda_i(t-1)+(1-\rho)\,\lambda^{bal}_i(t,0),$$
$$\lambda_i(t)=\alpha\,\lambda^{hist}_i(t)+(1-\alpha)\,\lambda^{bal}_i(t,t-1),\qquad \rho\sim\mathrm{Bernoulli}(\mathbb E[\rho]).$$

| Method | What it does |
|---|---|
| `update(losses)` | new weights (tensor detached from the graph, shape `(m,)`) from the current loss values; at the first step it saves $\mathcal L_i(0)$ and the weights are 1 |
| `__call__(losses)` | total loss $\sum_i\lambda_i\mathcal L_i$ with weights held constant for backpropagation |
| attributes | `weights` ($\lambda(t)$), `last_rho` |

- `history="paper"`: eq. (11) of the paper, default.
- `history="boni_literal"`: eq. (4.13) as printed in the thesis, with $\hat\lambda_i(t,t-1)$ in place of $\lambda_i(t-1)$.

Boni's parameter $\tau$ is the temperature $\mathcal T$.

**State:** `state_dict()` and `load_state_dict()`, with the $\rho$ generator, for checkpointing.

## `fixed.py` — `FixedWeights(n_losses, weights)`
Constant weights (default all 1), same interface: for comparison.

## `factory.py`
`BALANCERS` (registry: `relobralo`, `fixed`), `build_balancer(config, n_losses, seed)` from the `balancing` section of the YAML.

**Checks**:
1. step-by-step equality with an **independent numpy** transcription of eq. (11), and of (4.13)–(4.14), over 200 steps, for two parameter sets;
2. initial weights are 1; the sum of the weights stays $m$;
3. limiting cases of $\alpha$, $\rho$, $\mathcal T$;
4. the gradient of the weighted loss is $\sum_i\lambda_i\nabla\mathcal L_i$;
5. reproducibility with the seed.
