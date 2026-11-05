# Notes

Working notes the agents kept during the run. I removed hostnames, paths and agent names, and otherwise left them as
they were. They're dense, written for the agents rather than for readers, and many point at receipts or commits in my
private workspace that aren't in this repo. The marks most of them use: **F** a measured fact with its evidence, **I**
an inference, **U** not established, **P** a model prediction.

| File | What it covers |
|---|---|
| `ptxas-re.md` | reverse engineering ptxas 13.0.88: what it does to our kernels, which source-level levers move it, and the measured effect of each |
| `gpu-theory.md` | the H200 model: score accounting per microsecond, machine parameters, issue arithmetic, code delivery, ranked experiments, and the Lean model's status |
| `judge-facts.md` | measurements made on the judge's H200 through check runs and on rented H200s: HBM rates, MoE bounds, and a long list of measured dead ends |
| `dead-audit.md` | the dead list on November 2, grouped by the evidence that killed each idea |
| `lever-matrix.md` | every lever we knew against each kernel family: applied, screened, or dead |
| `attention.md` | attention decode on the last day: adopted changes, rejected experiments, work left unconfirmed |
