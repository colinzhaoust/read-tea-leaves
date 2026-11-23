A made-up run for trying `tea read`: `curve.csv` is a synthetic training-loss curve (a
power-law decay with noise, logged every 50 steps for 20,000 steps) and `config.json` holds
invented settings. It is not a run of the benchmark.

    tea read --csv examples/my-run/curve.csv --budget 20000 --config examples/my-run/config.json
