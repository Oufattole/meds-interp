# meds-interp

Interpreting embeddings from MEDS data

Current setup for knn sweep:

```
meds-knn modalities=[m1,m2] modality_weights=[1,1] data_path=/path/to/data
```

Let's assume this file structure:

data_path
|-train.parquet
|-val.parquet
|-test.parquet

Every parquet has columns `m1`,`m2`, and `labels`

We should store the output model somewhere
