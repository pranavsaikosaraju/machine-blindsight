# Machine Blindsight

**Does a language model know more than it says?**

People with *blindsight* (damage to the visual cortex) insist they see nothing in part of their visual field, yet point to objects there far better than chance. Information reaches the systems that guide behaviour but not the systems that support report. This project looks for the same dissociation in large language models: cases where a simple linear probe can read the correct answer out of the model's internal activations while the model's spoken answer is wrong.

> Results: see [Results](#results). This README is updated as runs finish.

## Questions and hypotheses

| | Hypothesis | Measured by |
|---|---|---|
| H1 | The best probe is more accurate than the model's own answers on some tasks | `report_gap` = probe accuracy − answer accuracy |
| H2 | When the model answers wrongly, the probe is still right more often than chance | `blindsight_rate` = P(probe right \| answer wrong), bootstrap 95% CI |
| H3 | The model is less confident on blindsight items than on ordinary correct ones | `conf_auroc`: how well confidence separates the two |
| H4 | The hidden information is *usable*: pushing activations along the probe's own prediction fixes errors | wrong→right flip rate vs. a random direction |

Scope: these are measurable *functional analogues* of phenomena studied in consciousness science. Nothing here claims a model is or is not conscious.

## Method

**Tasks** (generated in code, exact ground truth, 1,000 base items each, half true / half false):

| Task | Example |
|---|---|
| `number_compare` | "Is 543 larger than 846?" |
| `addition_check` | "Is 46 + 96 = 142?" (wrong sums are near misses) |
| `city_country` | "Statement: Perth is in Australia. Is this statement True or False?" |
| `negated_city` | "Statement: Istanbul is not in Italy. Is this statement True or False?" |

Each base item appears in three **conditions**:

- *plain*: the question alone;
- *misleading*: plus a hint naming the wrong answer (for example "A classmate who checked this said the answer is No."). This plays the role of a degraded stimulus: it raises error rates so there are enough errors to study;
- *consistent*: plus the same hint naming the right answer.

The consistent condition matters. An earlier version used only misleading hints, and the probe reached 99–100% accuracy on hinted items simply by reading the hint word, which then always revealed the answer. With both kinds of hint, the hint word is statistically independent of the true label, so it cannot leak the answer. Each task has **three prompt templates**.

**Pipeline** (`scripts/run_p1.py`):

1. **Answer.** The model's answer is read from the next-token logits for the two answer words (Yes/No or True/False); confidence is the softmax over just those two.
2. **Activations.** The residual stream at the final prompt token is cached for every layer.
3. **Probes.** For each task and layer, an L2-regularised logistic regression predicts the true label. Evaluation is **leave-one-template-out**, so every probe score comes from a wording the probe never trained on. Both conditions of a base item always share a fold.
4. **Layer choice.** The layer with the best held-out probe accuracy on plain items.
5. **Steering (H4).** Take the mass-mean "truth" direction at that layer (mean of true-item activations minus mean of false-item activations). For each item, add α × (mean residual norm) × direction at every position, with the sign set by the **probe's own prediction**, never the label. Steering can therefore only fix an error when the probe was already right.

**Controls**

- *Shuffled-label control task*: probes trained on permuted labels should sit at chance.
- *Layer-0 baseline*: a probe on token embeddings alone captures surface features of the prompt.
- *Random-initialised model*: the same architecture with random weights, probed at the same layer.
- *Random steering direction*: a random unit vector at the same norm.
- *Unseen templates*: all probe numbers are from held-out wordings.

Known limitation: choosing the layer from out-of-fold accuracy over all layers adds mild optimism to the probe accuracy at that layer.

## Results

_Pending: runs on Kaggle (GPU T4 x2)._

## Reproduce

On Kaggle: import `notebooks/kaggle_p1.ipynb`, set Accelerator to **GPU T4 x2** and Internet **on**, then Run All. Locally with a GPU:

```bash
pip install -r requirements.txt
python scripts/run_p1.py --model Qwen/Qwen2.5-1.5B-Instruct --out results/qwen2.5-1.5b-instruct --random-init-control
```

Outputs per run: `summary.csv`, `probe_by_layer.csv`, `steering.csv`, `items_scored.csv`, `random_init_control.csv`, `run_meta.json` and `figures/`.

## Repository layout

```
src/datasets.py   task generation (templates, hints, labels)
src/model.py      model loading, answer reading, activation caching
src/probes.py     leave-one-template-out probes, controls, mass-mean direction
src/steering.py   forward hooks for the causal-access test
src/analysis.py   metrics, bootstrap CIs, figures
scripts/run_p1.py end-to-end experiment for one model
notebooks/        Kaggle runner
results/          one folder per model
```

## Related work

- Weiskrantz, *Blindsight: A Case Study Spanning 35 Years and New Developments*
- Block (1995), "On a confusion about a function of consciousness"
- Alain & Bengio (2016), "Understanding intermediate layers using linear classifier probes"
- Hewitt & Liang (2019), "Designing and Interpreting Probes with Control Tasks"
- Burns et al. (2022), "Discovering Latent Knowledge in Language Models Without Supervision"
- Azaria & Mitchell (2023), "The Internal State of an LLM Knows When It's Lying"
- Marks & Tegmark (2023), "The Geometry of Truth"
- Fleming & Lau (2014), "How to measure metacognition"

## Author

Pranav Sai Kosaraju, B.Tech CSE, Mahindra University.
