# Step 1: remove DeploymentStates and related multi-deployment logic.
# Reason: you only need to track a single model's replica count, not manage multiple deployments

import logging
import threading
import time
from datetime import datetime
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from aibrix.gpu_optimizer.optimizer import GPUProfile, Optimizer
from aibrix.gpu_optimizer.utils import DelayedLog
from .clusterer import MovingDBSCANClusterer
from .helpers import Centeroid, DataBuffer, DataPoint
from .load_reader import LoadReader, LoadRecord
from .profile_reader import ProfileReader

Empty_Array: Iterable = []

logger = logging.getLogger("aibrix.gpu_optimizer.load_monitor")

class ModelMonitor:
    def __init__(
        self,
        model_name: str,
        load_reader: LoadReader,
        window: int = 240,
        profile_reader: Optional[ProfileReader] = None,
        gpu_fraction: float = 100.0,
        debug: bool = False,
    ):
        self.model_name = model_name
        self.thread = None
        self.debug = debug
        self.done = False
        self.window = float(window)
        self.gpu_fraction = gpu_fraction
        self._lock = threading.Lock()

        self._load_reader = load_reader
        self._profile_reader: Optional[ProfileReader] = profile_reader

        self._profiles: dict[str, GPUProfile] = {}
        self._optimizer = Optimizer(self.gpu_fraction)

        self._centers: Iterable[Centeroid] = Empty_Array
        self._labels: Iterable[int] = Empty_Array
        self._data: Optional[DataBuffer] = None
        self._progress: float = 0.0
        self._cost = 0.0

        self._replicas: int = 0

        if profile_reader is not None:
            self.load_profiles(profile_reader)
        elif self.debug:
            self._optimizer.set_profile(GPUProfile("default", cost=1.0, tputs=[[100]], indexes=[[10], [10]]))

    def read_num_replicas(self) -> int:
        return self._replicas

    def load_profiles(self, profile_reader: Optional[ProfileReader] = None) -> bool:
        try:
            if profile_reader is None:
                profile_reader = self._profile_reader
                if profile_reader is None:
                    logger.error("Profile reader not initialized")
                    return False

            profiles = profile_reader.read()
            for profile in profiles:
                profile.cost /= self.gpu_fraction
                self._profiles[profile.gpu] = profile
                self._optimizer.set_profile(profile)
            return True
        except Exception as e:
            logger.error(f"Failed to load profiles: {e}")
            return False

    def start(self):
        self.thread = threading.Thread(target=self._run)
        self.thread.daemon = True
        self.thread.start()        

    def stop(self):
        self.done = True

    def _run(self):
        try:
            next(self._run_yieldable(False))
        except StopIteration:
            pass
        except Exception as e:
            logger.error(f"Unexpected error on monitoring {self.model_name}: {e}")

    def _run_yieldable(self, yieldable: bool, window_scaling: float = 1.0):
        clusterer = MovingDBSCANClusterer(0.5, 10, 4, self.window * window_scaling)
        self._data = DataBuffer(int(self.window) * 10)

        n = 0
        while not self.done:
            start = time.time()
            if clusterer.validate():
                self._data.trim_head(-clusterer.length)
                print(f"Data buffer length: {self._data.len}, clusterer length: {clusterer.length}")

            records, cur_rate = self._load_reader.read(time.time())
            # for record in records:
            #     print(f"Record: {record.ts}, Input: {record.input_tokens}, Output: {record.output_tokens}, Freq: {record.freq}")
            # print(f"current rate: {cur_rate:.2f} records/sec")
            tokens = list(self._expand_records(records))
            if len(tokens) > 0:
                self._data.reconcile(clusterer.length + len(tokens))
                dps = self._data.append(tokens)
                clusterer.insert(dps)
                self._data.commit()

            if self._data.len > 0:
                # for i in range(self._data.len):
                #     dp = self._data.datapoints.datapoint(i)
                #     print(f"[{i}] log2_input={dp.signature[0]:.2f}, log2_output={dp.signature[1]:.2f}, timestamp={dp.age:.2f}")
                uncategorized = None
                self._labels, self._centers = clusterer.get_cluster_labels(self._data.datapoints, uncategorized=uncategorized)

                # for i, label in enumerate(self._labels):
                #     dp = self._data.datapoints.datapoint(i)
                #     print(f"[{i}] cluster={label}, log2_input={dp.signature[0]:.2f}, log2_output={dp.signature[1]:.2f}, timestamp={dp.age:.2f}")
                # for i, center in enumerate(self._centers):
                #     print(f"[cluster {i}] center={center.center}, radius={center.radius:.2f}, rate={center.rate:.2f}, size={center.size}")
            else:
                self._labels, self._centers = Empty_Array, Empty_Array
                
            n += 1
            duration = (datetime.now().timestamp() - start) * 1000
            centers = list(self._centers)
            # logger.info(
            #     "%s batch %d took %d ms: %d centers: %s",
            #     self.model_name,
            #     n,
            #     round(duration),
            #     len(centers),
            #     DelayedLog(lambda: str([str(center) for center in self._centers])),
            # )

            centers = list(self._centers)
            if len(centers) > 0:
                self._optimize(centers, max(self._data.len / clusterer.window, cur_rate))
            elif self._data.len == 0:
                self._replicas = 0
                logger.info(f"{self.model_name} scaled to 0 replicas")
            else:
                logger.info("Skip optimization, insufficient data")

            if yieldable:
                yield
            else:
                wait = self._load_reader.next_available() - time.time()
                if wait > 0:
                    time.sleep(wait)

    def _expand_records(self, records: Iterable[LoadRecord]):
        for r in records:
            for _ in range(r.freq):
                yield DataPoint(r.output_tokens, r.input_tokens, age=r.ts)

    def _optimize(self, centers: Iterable[Centeroid], request_rate: float):
        self.load_profiles()
        if not self._optimizer.set_workload_distribution(centers, request_rate):
            return

        result = self._optimizer.run()
        if result is None or "cost" not in result:
            return

        self._replicas = int(sum(result[k] for k in result if k != "cost"))
        self._cost = result["cost"]
        logger.info(f"{self.model_name} optimized: replicas={self._replicas}, cost={self._cost:.2f}")
