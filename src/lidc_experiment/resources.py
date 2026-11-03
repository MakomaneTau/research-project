"""Sample process memory while recording synchronized accelerator peaks."""
import threading

import psutil


class ResourceMonitor:
    def __init__(self, cuda=False):
        self.cuda = cuda
        self.peak_rss = 0
        self.peak_cuda = 0
        self.stop = threading.Event()

    def _sample(self):
        process = psutil.Process()
        while not self.stop.is_set():
            self.peak_rss = max(self.peak_rss, process.memory_info().rss)
            self.stop.wait(.02)

    def __enter__(self):
        if self.cuda:
            import torch
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        self.peak_rss = psutil.Process().memory_info().rss
        self.worker = threading.Thread(target=self._sample, daemon=True)
        self.worker.start()
        return self

    def __exit__(self, *_):
        if self.cuda:
            import torch
            torch.cuda.synchronize()
            self.peak_cuda = torch.cuda.max_memory_allocated()
        self.stop.set()
        self.worker.join()
        self.peak_rss = max(self.peak_rss, psutil.Process().memory_info().rss)

    def as_dict(self):
        return {"peak_process_rss_bytes": self.peak_rss,
                "peak_cuda_allocated_bytes": self.peak_cuda,
                "rss_sampling_interval_seconds": .02}
