import time

import pandas as pd


VARIANTS = {
	"V1_full": {"use_scoring_matrix": True, "use_hypothesizer": True, "use_provenance": True},
	"V2_no_matrix": {"use_scoring_matrix": False, "use_hypothesizer": True, "use_provenance": True},
	"V3_no_hypothesizer": {"use_scoring_matrix": True, "use_hypothesizer": False, "use_provenance": True},
	"V4_no_provenance": {"use_scoring_matrix": True, "use_hypothesizer": True, "use_provenance": False},
	"V5_static": {"use_scoring_matrix": False, "use_hypothesizer": False, "use_provenance": False},
}

DRIFT_LEVELS = [999, 2, 4, 7, 0]
SEEDS = [42, 43, 44, 45, 46]


def run_single_experiment(config: dict) -> dict:
	"""Run one configured experiment and return metrics dict."""
	raise NotImplementedError("Implement run_single_experiment in your experiment pipeline")


def run_all_experiments(base_config: dict) -> pd.DataFrame:
	results = []
	total = len(VARIANTS) * len(DRIFT_LEVELS) * len(SEEDS)
	i = 0

	for variant_name, variant_flags in VARIANTS.items():
		for drift_after in DRIFT_LEVELS:
			for seed in SEEDS:
				i += 1
				print(f"[{i}/{total}] variant={variant_name} drift_after={drift_after} seed={seed}")

				config = {**base_config}
				config["streaming"]["drift_after"] = drift_after
				config["streaming"]["seed"] = seed
				config.update(variant_flags)

				t0 = time.time()
				result = run_single_experiment(config)
				runtime = time.time() - t0

				results.append(
					{
						"variant": variant_name,
						"drift_after": drift_after,
						"seed": seed,
						"runtime_s": round(runtime, 3),
						**result,
					}
				)

	return pd.DataFrame(results)
