"""Task 1 - print and save the dataset validation report.

Run:  python -m demos.task01_dataset
"""
import config  # noqa: F401

import dataset
from demos._transcript import transcript

if __name__ == "__main__":
    with transcript("task01_dataset"):
        dataset.report()
        # Determinism check: regenerating with the same seed yields identical records.
        assert dataset.generate_orders() == dataset.ORDERS
        print("\nDeterminism: regenerating with SEED=42 reproduces all 60 records exactly.")
