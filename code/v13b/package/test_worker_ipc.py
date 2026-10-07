"""Exercise CPU DataLoader worker tensor IPC with the production strategy."""
import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset

# Importing the production entry point configures the strategy before workers
# are created, matching the actual sbatch process.
import run_official  # noqa: F401,E402


class WorkerIPCTests(unittest.TestCase):
    def test_file_system_strategy_transfers_worker_batches(self):
        self.assertEqual(torch.multiprocessing.get_sharing_strategy(), 'file_system')
        values = torch.arange(64 * 128, dtype=torch.float32).reshape(64, 128)
        labels = torch.arange(64)
        loader = DataLoader(TensorDataset(values, labels), batch_size=8,
                            num_workers=2, timeout=30, persistent_workers=False)
        received = 0
        for batch_values, batch_labels in loader:
            self.assertEqual(batch_values.shape, (8, 128))
            self.assertEqual(batch_labels.shape, (8,))
            received += len(batch_labels)
        self.assertEqual(received, 64)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
