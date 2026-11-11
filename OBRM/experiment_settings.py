"""Final experiment settings shared by both OBRM designs.

The runner uses these settings by default. Each call returns an independent
mapping; per-size overrides take precedence within the selected dataset.
"""


def load_settings():
    return {
        "data_root": "datasets",
        "sgx_sdk": "../sgx-sdk/sgxsdk",
        "sizes": [
            20000,
            40000,
            60000,
            80000,
            100000
        ],
        "query_count": 200,
        "minimum_recall": 0.9,
        "datasets": {
            "sift": {
                "base": "sift/{size_k}K/base.bvecs",
                "query": "sift/{size_k}K/query.bvecs",
                "groundtruth": "sift/{size_k}K/groundtruth.ivecs",
                "dimension": 128,
                "parameters": {
                    "K": 12,
                    "N": 512,
                    "B": 48,
                    "m": 30,
                    "r": 6,
                    "hidden": 32,
                    "epochs": 10,
                    "lr": 0.01,
                    "seed": 20260929,
                    "data_encoding": "uint8"
                },
                "by_size": {
                    "20000": {
                        "m": 16,
                        "epochs": 5
                    },
                    "40000": {
                        "K": 8,
                        "B": 64,
                        "r": 5,
                        "m": 16,
                        "epochs": 5
                    },
                    "60000": {
                        "K": 8,
                        "B": 64,
                        "r": 5,
                        "m": 16,
                        "epochs": 5
                    },
                    "80000": {
                        "B": 64,
                        "m": 16,
                        "epochs": 5,
                        "r": 7
                    },
                    "100000": {
                        "B": 64,
                        "m": 16,
                        "epochs": 5,
                        "r": 7
                    }
                }
            },
            "deep": {
                "base": "deep/{size_k}K/base.hvecs",
                "query": "deep/{size_k}K/query.hvecs",
                "groundtruth": "deep/{size_k}K/groundtruth.ivecs",
                "dimension": 96,
                "parameters": {
                    "K": 8,
                    "N": 2560,
                    "B": 320,
                    "m": 30,
                    "r": 4,
                    "hidden": 32,
                    "epochs": 10,
                    "lr": 0.01,
                    "seed": 20260929,
                    "data_encoding": "float16"
                },
                "by_size": {
                    "80000": {
                        "K": 10,
                        "B": 256,
                        "r": 5
                    },
                    "100000": {
                        "K": 10,
                        "B": 256,
                        "r": 5
                    }
                }
            },
            "emnist": {
                "base": "emnist/{size_k}K/base.bvecs",
                "query": "emnist/{size_k}K/query.bvecs",
                "groundtruth": "emnist/{size_k}K/groundtruth.ivecs",
                "dimension": 784,
                "parameters": {
                    "K": 12,
                    "N": 512,
                    "B": 48,
                    "m": 30,
                    "r": 5,
                    "hidden": 32,
                    "epochs": 10,
                    "lr": 0.01,
                    "seed": 20260929,
                    "data_encoding": "uint8"
                },
                "by_size": {
                    "40000": {"r": 6},
                    "60000": {"r": 6},
                    "80000": {"r": 6},
                    "100000": {"r": 6}
                }
            },
            "laion": {
                "base": "laion/{size_k}K/base.bvecs",
                "query": "laion/{size_k}K/query.bvecs",
                "groundtruth": "laion/{size_k}K/groundtruth.ivecs",
                "dimension": 512,
                "parameters": {
                    "K": 8,
                    "N": 2560,
                    "B": 320,
                    "m": 16,
                    "r": 5,
                    "hidden": 32,
                    "epochs": 10,
                    "lr": 0.01,
                    "seed": 20260929,
                    "data_encoding": "uint8"
                },
                "by_size": {
                    "80000": {"m": 30},
                    "100000": {"m": 30}
                }
            }
        }
    }
