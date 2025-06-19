Unofficial implementation of [Pointer Networks](https://arxiv.org/abs/1506.03134)

`ptr_net.py` - minimal implementation, with masking of already chosen indices (useful for generating only valid TSP solutions)

`tsp_bench.ipynb` - training and testing PtrNet on available TSP dataset provided by the paper's authors [here](https://drive.google.com/drive/folders/0B2fg8yPGn2TCMzBtS0o4Q2RJaEU)

Benchmark summary

| N | Test set solution | paper Ptr-Net | this Ptr-Net
|------|-----|------|------|
| 5 (5 trained) | 2.12 | 2.12 | 2.12 | 2.12
| 20 (5-20 trained) | 4.24 | 3.88 | 4.28
| 40 (5-20 trained) | 5.82 | 5.91 | 8.10
| 50 (5-20 trained) | 6.43 | 7.66 | 10.24

The difference in performance may be caused by the lack of beam search, which authors mention was used in the paper to generate tsp solutions.