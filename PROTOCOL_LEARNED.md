# Frozen v2 protocol — learned future memory without a privileged basis

Fixed before the scored run on 2026-10-13. This is a synthetic state-learning test, **not** a learned natural-image model or biological-memory result.

## Observation contract
Two nonlinear, seeded 32-dimensional systems contain four immediately relevant task coordinates, a latent bridge, and 27 nuisance coordinates. Both the physical basis and random 12-channel sensor are hidden from the learner. Each model sees only a **three-frame sensor window**, the two preceding actions, and task-response training labels for actual sampled future action sequences. No model sees the hidden state, rotation, hand-written oracle encoder, true update matrices, or the hidden partition. After the initial sensor window, six-step evaluation is strictly **open loop**.

The bridge world uses gain 0.28 from bridge into core; the closed world uses 0. All 27 nuisance coordinates continue evolving, but never feed back into the task-relevant state. This v2 fixture is more persistent than the original v1 law to avoid the six-step task becoming nearly zero.

## Predeclared split and models

Fixed independent apparatus seeds **41 and 97**, 1,200 train, 240 validation, 400 test trajectories per world/seed; disjoint random initial states and action strings. Six response queries, four actions, 12 sensor channels, three-frame histories, horizon 6. Target predictions are supervised only from observed future outcomes during training.

All models share the same action-dependent tanh-affine state updater and six-query affine reader. Adam lr 0.006, 160 epochs, batch 300, gradient clipping 3, validation every 10 epochs, checkpoint selected by validation future-response MSE; no tuning on test.

1. **Predictive 4**: train a linear history-to-4 encoder jointly with the 4D updater and reader on six-step query prediction.
2. **Predictive 5**: identical architecture/objective, but five retained coordinates.
3. **PCA 5**: train-only PCA on the identical observation histories (image/sensor reconstruction objective), frozen five-dimensional encoder, and the same trainable 5D updater/reader.
4. **GRU 5**: conventional three-frame GRU with five recurrent units as encoder, then the same 5D updater/reader. Counts extra shared parameters; equal per-episode retained code size.

Sensor and response normalization use **training-only** statistics. All four see the same training labels and held-out inputs. Encoder/decoder weights, observation history and fitting cost must be accounted separately from the 4- or 5-float *marginal retained state*.

## Metrics and adversarial questions
Measure pooled held-out normalized RMSE on current answers, steps 1–3, steps 4–6, and all six future steps; all seeds retained and report paired episode errors. Primary comparisons: Predictive5 vs Predictive4 in each world (is the fifth coordinate useful when causal?), Predictive5 vs PCA5 (does predictive training beat reconstruction at equal code size?), Predictive5 vs GRU5 (does a standard recurrent encoder perform as well or better?). Novel action strings and initial states, no future input during rollout. The v1 analytical oracle is **not** a fair learner comparison.

## Interpretation / kill
A learned success is **not presumed**. Publish all failures and losses, including if PCA/GRU win or the extra coordinate never helps. This fixture is controlled and its sensor may make information unavailable: no amount of latent dimension fixes unobservability. Closed-loop sufficiency outside the tested action horizon remains unproven. Do not equate the rank of a 32D operator with the number of per-memory scalars or claim any storage saving without shared model costs.
