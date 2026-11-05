# Algebra and search

The searches and proofs behind the NVFP4 decode ("dqx") and the whole-layer e-graph work described in the post.

| File | What it does |
|---|---|
| `fp4_layout3.c` | the search that found dqx: 4 masked LOP3 registers `r = M & (u ^ v)`, chained, with one right-moving source (rotate or logical shift) and up to 3 left shifts of `q` or of the rotated word, under a free invertible GF(2) storage layout; registers must each add 8 independent rows (the pruning) |
| `fp4_layout.c`, `fp4_layout2.c` | the earlier, narrower families: left shifts only, then one logical right shift without chaining (both 0 solutions) |
| `fp4_layout4.py` to `fp4_layout6.py` | permutation-only layouts (0 solutions) |
| `fp4_shr31_result.txt` | the logical-shift (IMAD.HI-able, 4-ALU) family exhausted to shifts up to 31: 0 |
| `fp4_layout_verify.py` | the exactness proof: the decode is GF(2)-linear, so checking the 32 basis vectors and inverting the 32 x 32 map proves all 2^32 words; plus every byte value at every position and random words |
| `layer_egraph.py`, `dag_extract.py` | egglog saturation of the layer's front and back halves (33 rewrite rules), then sharing-aware extraction (greedy DAG and an exact MILP) |
| `egraph_out.txt`, `dag_out.txt` | their outputs under the two cost models |
| `proofs.py` | SymPy proofs of the vector rewrites and Z3 proofs of two scalar identities (the RMS-epsilon rescale and the rs1² expansion) |
| `hpr_accuracy.py` | stored-precision error of the hoisted pooler/router form against production's |

```
cc -O2 -o fp4_layout3 algebra/fp4_layout3.c && ./fp4_layout3 20     # max left shift 20; prints the first solutions
python3 algebra/fp4_layout_verify.py                                  # "invertible; 101024 checks bit-exact"
```

Every negative result here holds within the family that file searches, and no wider.
