
import warnings
from ml.preprocess import load_cmapss

for subset in ["FD001", "FD002", "FD003", "FD004"]:
    print(f"\nChecking {subset}...")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = load_cmapss(subset)

    print("Loaded successfully:", result is not None)
    print("Warnings:", len(caught))

    for warning in caught[:3]:
        print(type(warning.message).__name__, ":", warning.message)

print("\nCheck complete. No models trained.")