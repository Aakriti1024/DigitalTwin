"""
backend/replay.py
-----------------
Progressive sensor replay manager for NASA C-MAPSS test engines.

Allows progressive step-by-step or timed replay (1 cycle / sec by default)
without bulk-loading test data at server startup.
"""

import asyncio
import logging
import threading
from typing import Dict, Any, Optional

import pandas as pd
from config import SELECTED_SENSORS, SEQUENCE_LENGTH
from ml.preprocess import load_cmapss
from backend.digital_twin import registry

logger = logging.getLogger(__name__)


class EngineReplay:
    """
    Manages progressive streaming replay of NASA C-MAPSS test data
    for a single engine.
    """

    def __init__(self, engine_id: str, subset: str = "FD001", interval_sec: float = 1.0):
        self.engine_id = engine_id
        self.subset = subset
        self.interval_sec = max(0.1, float(interval_sec))
        self.current_index = 0
        self.is_playing = False
        self._lock = threading.Lock()
        self._task: Optional[asyncio.Task] = None

        # Parse numeric engine id (e.g. "E1" -> 1)
        try:
            num_id = int(str(engine_id).upper().replace("E", ""))
        except ValueError:
            num_id = 1

        self.num_id = num_id

        # Load dataset
        _, test_df, rul_series = load_cmapss(subset)
        self.engine_df = (
            test_df[test_df["engine_id"] == num_id]
            .sort_values("cycle")
            .reset_index(drop=True)
        )
        self.total_cycles = len(self.engine_df)
        self.final_test_rul = (
            float(rul_series.loc[num_id]) if num_id in rul_series.index else 0.0
        )
        self.max_test_cycle = (
            int(self.engine_df["cycle"].max()) if not self.engine_df.empty else 0
        )
        # Total operational life until failure = max cycle in test record + final ground truth RUL
        self.total_engine_life = self.max_test_cycle + self.final_test_rul

        logger.info(
            "EngineReplay initialized for %s (%s): %d cycles, final_rul=%.1f, total_life=%.1f",
            engine_id,
            subset,
            self.total_cycles,
            self.final_test_rul,
            self.total_engine_life,
        )

    def get_status(self) -> Dict[str, Any]:
        """Return current replay state."""
        with self._lock:
            status_text = "completed" if self.current_index >= self.total_cycles else (
                "playing" if self.is_playing else ("paused" if self.current_index > 0 else "ready")
            )
            return {
                "engine_id": self.engine_id,
                "current_index": self.current_index,
                "current_cycle": (
                    int(self.engine_df.iloc[self.current_index - 1]["cycle"])
                    if 0 < self.current_index <= self.total_cycles
                    else 0
                ),
                "total_cycles": self.total_cycles,
                "is_playing": self.is_playing,
                "interval_sec": self.interval_sec,
                "status": status_text,
                "is_warmed_up": self.current_index >= SEQUENCE_LENGTH,
                "warmup_progress": f"{min(self.current_index, SEQUENCE_LENGTH)}/{SEQUENCE_LENGTH}",
            }

    def step(self) -> Dict[str, Any]:
        """
        Advance exactly one cycle. Thread-safe.
        Returns the updated twin state.
        """
        with self._lock:
            if self.current_index >= self.total_cycles:
                logger.info("Engine %s reached end of replay data.", self.engine_id)
                self.is_playing = False
                twin = registry.get_or_create(self.engine_id)
                return twin.to_dict()

            row = self.engine_df.iloc[self.current_index]
            cycle = int(row["cycle"])

            sensor_data = {
                sensor: float(row[sensor])
                for sensor in SELECTED_SENSORS
                if sensor in row
            }

            op_settings = {
                "op_set1": float(row.get("op_set1", 0.0)),
                "op_set2": float(row.get("op_set2", 0.0)),
                "op_set3": float(row.get("op_set3", 0.0)),
            }

            # Reference RUL: total_engine_life - current cycle
            reference_rul = max(0.0, self.total_engine_life - cycle)

            twin = registry.get_or_create(self.engine_id)
            updated_state = twin.update_with_sensor_data(
                cycle=cycle,
                sensor_data=sensor_data,
                op_settings=op_settings,
                reference_rul=reference_rul,
            )

            self.current_index += 1
            if self.current_index >= self.total_cycles:
                self.is_playing = False

            return updated_state

    def reset(self) -> Dict[str, Any]:
        """
        Reset replay cursor to 0 and reset twin history.
        """
        with self._lock:
            self.is_playing = False
            if self._task and not self._task.done():
                self._task.cancel()
                self._task = None

            self.current_index = 0
            twin = registry.get_or_create(self.engine_id)
            twin.reset_state()
            return twin.to_dict()

    def pause(self) -> Dict[str, Any]:
        """Pause playback."""
        with self._lock:
            self.is_playing = False
            if self._task and not self._task.done():
                self._task.cancel()
                self._task = None
        return self.get_status()

    def set_speed(self, interval_sec: float) -> Dict[str, Any]:
        """Update interval between playback cycles."""
        with self._lock:
            self.interval_sec = max(0.1, float(interval_sec))
        return self.get_status()

    async def _playback_loop(self):
        """Async playback worker loop."""
        try:
            while self.is_playing and self.current_index < self.total_cycles:
                self.step()
                await asyncio.sleep(self.interval_sec)
        except asyncio.CancelledError:
            pass
        finally:
            with self._lock:
                if self.current_index >= self.total_cycles:
                    self.is_playing = False

    def start(self, interval_sec: Optional[float] = None) -> Dict[str, Any]:
        """
        Start timed playback in an async task.
        """
        with self._lock:
            if interval_sec is not None:
                self.interval_sec = max(0.1, float(interval_sec))

            if self.is_playing:
                return self.get_status()

            if self.current_index >= self.total_cycles:
                # If completed, reset to beginning before replaying
                self.current_index = 0
                twin = registry.get_or_create(self.engine_id)
                twin.reset_state()

            self.is_playing = True

            try:
                loop = asyncio.get_running_loop()
                self._task = loop.create_task(self._playback_loop())
            except RuntimeError:
                # No running asyncio loop in current thread
                logger.warning("No running asyncio event loop for replay playback task.")

        return self.get_status()


class ReplayManager:
    """Singleton manager for engine replays."""

    def __init__(self):
        self._replays: Dict[str, EngineReplay] = {}
        self._lock = threading.Lock()

    def get_or_create(self, engine_id: str, subset: str = "FD001", interval_sec: float = 1.0) -> EngineReplay:
        with self._lock:
            if engine_id not in self._replays:
                self._replays[engine_id] = EngineReplay(
                    engine_id=engine_id, subset=subset, interval_sec=interval_sec
                )
            return self._replays[engine_id]

    def get(self, engine_id: str) -> Optional[EngineReplay]:
        with self._lock:
            return self._replays.get(engine_id)


replay_manager = ReplayManager()
