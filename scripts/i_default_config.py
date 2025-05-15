base_config = {
    "num_processes": 4,
    "batching_options": {
        "batch_size___fixed_batching": 16,
    },
    "latency_simulation": {
        "simulate": False,        # Whether to introduce artificial delays.
        "delay_min": 0,           # Minimum delay in seconds (e.g., 0.05 for 50ms).
        "delay_max": 0,           # Maximum delay in seconds (e.g., 0.3 for 300ms).
        "simulate_burst": False,  # Whether to simulate bursty traffic conditions.
        "burst_interval": 0.0,    # Delay (seconds) after a burst.
        "burst_size": 0           # Number defining a burst.
    },
    "decoder_config": {
        "decoding_mode": None,         # Default is None; updated when decoder variations apply (e.g., "greedy", "top_k", "top_p").
        "decoder_temperature": 1.0,      # Default temperature.
        "decoder_top_k": None,           # Set to None if top_k sampling is not applicable.
        "decoder_top_p": None            # Set to None if top_p sampling is not applicable.
    },
    "fp_precision": "float32",           # Can be updated to float16 in experiments.
    "quantization_config": {
        "quantization": None,          # Default None, updated when experiments specify quantisation settings.
        "load_in_8bit": None,          # None if not applicable.
        "load_in_4bit": None,          # None if not applicable.
        "cached_flops_for_quantised_models": 52638582308864  # FLOPs value for quantized models.
    }
}
