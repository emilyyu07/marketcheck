# MarketCheck Demo

## Running the Demo

1. Generate a clean synthetic dataset:

```bash
python benchmarks/generate_synthetic.py --rows 10000 --output demo/clean.csv
```

2. Inject faults:

```bash
python demo/inject_faults.py --input demo/clean.csv --output demo/faulted.csv
```

3. Validate the faulted dataset:

```bash
marketcheck validate demo/faulted.csv
```

4. Compare with the clean dataset:

```bash
marketcheck validate demo/clean.csv
```
