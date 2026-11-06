# Artifacts

Training writes new, isolated run directories containing configuration, split dates, fold-specific scalers/checkpoints, histories, predictions, metrics, runtime metadata and SHA-256 manifests. An existing run must not be overwritten.

Development fold checkpoints are not automatically deployable challengers. Registration requires an independently verified bundle for `stationary-return-v2`, matching symbol, features, scalers, preprocessing version and checksums. The V2 registry remains unavailable without such a bundle. Checksums establish file integrity, not trust in arbitrary pickle files; only load project-controlled artifacts.
