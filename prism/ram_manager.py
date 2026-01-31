"""
PRISM RAM Manager

Universal RAM manager for batch processing.

Strategy:
1. Batch entities
2. Process batch
3. Write to disk
4. Clear memory
5. Repeat

Memory: O(batch_size), not O(total_entities)

FULL COMPUTE. RAM OPTIMIZED. NO EXCEPTIONS.
"""

import gc
from dataclasses import dataclass
from typing import Callable, Generator, Any, List
from pathlib import Path

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False


@dataclass
class MemoryStats:
    """Current memory statistics."""
    total_gb: float
    available_gb: float
    used_pct: float

    @classmethod
    def current(cls) -> "MemoryStats":
        if not HAS_PSUTIL:
            return cls(total_gb=0, available_gb=0, used_pct=0)
        mem = psutil.virtual_memory()
        return cls(
            total_gb=mem.total / (1024**3),
            available_gb=mem.available / (1024**3),
            used_pct=mem.percent,
        )

    def __str__(self):
        if not HAS_PSUTIL:
            return "RAM: (psutil not available)"
        return f"RAM: {self.used_pct:.1f}% used ({self.available_gb:.1f}GB available)"


@dataclass
class RAMConfig:
    """RAM management configuration."""
    batch_size: int = 100          # Entities per batch
    flush_interval: int = 100      # Same as batch_size typically
    clear_cache: bool = True       # gc.collect() after each batch
    max_memory_pct: float = 0.8    # Use max 80% of available RAM

    @classmethod
    def auto(cls, n_entities: int, n_signals: int, window_size: int = 1024) -> "RAMConfig":
        """Auto-configure based on available RAM and data characteristics."""

        if not HAS_PSUTIL:
            # Conservative default without psutil
            return cls(batch_size=50, flush_interval=50, clear_cache=True)

        available_ram = psutil.virtual_memory().available

        # Estimate memory per entity (rough)
        # Each engine produces ~50 metrics, each metric is 8 bytes
        # Plus intermediate computations ~10x
        bytes_per_window = n_signals * window_size * 8  # Raw data
        bytes_per_entity_estimate = bytes_per_window * 100  # With all engines

        # Ensure minimum estimate
        bytes_per_entity_estimate = max(bytes_per_entity_estimate, 10_000_000)  # 10MB min

        # How many entities can we fit in 80% of RAM?
        target_ram = available_ram * 0.8
        batch_size = max(1, int(target_ram / bytes_per_entity_estimate))

        # Cap at reasonable limits
        batch_size = min(batch_size, 500)   # Never more than 500
        batch_size = max(batch_size, 10)    # Never less than 10

        return cls(
            batch_size=batch_size,
            flush_interval=batch_size,
            clear_cache=True,
        )


class RAMManager:
    """
    Universal RAM manager for batch processing.

    No domain-specific logic. Works for any dataset.

    Usage:
        ram_manager = RAMManager()
        ram_manager.process_in_batches(
            items=unit_ids,
            process_func=run_all_engines,
            write_func=write_parquet,
        )
    """

    def __init__(
        self,
        max_memory_pct: float = 0.8,
        min_batch_size: int = 10,
        max_batch_size: int = 500,
    ):
        self.max_memory_pct = max_memory_pct
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size

    def estimate_batch_size(
        self,
        n_items: int,
        bytes_per_item: int = 10_000_000,  # 10MB default
    ) -> int:
        """
        Estimate optimal batch size based on available RAM.

        Args:
            n_items: Total items to process
            bytes_per_item: Estimated memory per item

        Returns:
            Batch size that fits in available RAM
        """

        if not HAS_PSUTIL:
            return min(self.max_batch_size, n_items)

        available = psutil.virtual_memory().available
        target = available * self.max_memory_pct

        batch_size = max(1, int(target / bytes_per_item))
        batch_size = min(batch_size, self.max_batch_size)
        batch_size = max(batch_size, self.min_batch_size)
        batch_size = min(batch_size, n_items)

        return batch_size

    def batch_generator(
        self,
        items: list,
        batch_size: int,
    ) -> Generator[list, None, None]:
        """Generate batches of items."""
        for i in range(0, len(items), batch_size):
            yield items[i:i + batch_size]

    def clear(self):
        """Clear memory caches."""
        gc.collect()

    def check_memory(self, threshold_pct: float = 90.0) -> bool:
        """
        Check if memory usage is below threshold.

        Returns True if OK, False if memory is critical.
        """

        if not HAS_PSUTIL:
            return True

        stats = MemoryStats.current()

        if stats.used_pct > threshold_pct:
            print(f"  Warning: Memory critical: {stats}")
            self.clear()
            return False

        return True

    def process_in_batches(
        self,
        items: list,
        process_func: Callable[[list], Any],
        write_func: Callable[[Any, int], None],
        bytes_per_item: int = 10_000_000,  # 10MB default estimate
        verbose: bool = True,
    ) -> List[Any]:
        """
        Universal batch processor with RAM management.

        Args:
            items: List of items to process (e.g., entity IDs)
            process_func: Function that processes a batch → results
            write_func: Function that writes results to disk (or None to skip)
            bytes_per_item: Estimated memory per item
            verbose: Print progress

        Returns:
            List of batch file paths (if write_func provided) or results

        This is the ONLY pattern needed. Works for any domain.
        """

        batch_size = self.estimate_batch_size(len(items), bytes_per_item)

        if verbose:
            print(f"  RAMManager: {len(items)} items, batch_size={batch_size}")
            print(f"  {MemoryStats.current()}")

        batch_outputs = []

        for batch_idx, batch in enumerate(self.batch_generator(items, batch_size)):

            # Check memory before processing
            if not self.check_memory():
                if verbose:
                    print(f"  Forcing garbage collection...")
                self.clear()

            # Process batch
            results = process_func(batch)

            # Write immediately (don't accumulate in memory)
            if write_func is not None:
                output = write_func(results, batch_idx)
                batch_outputs.append(output)
            else:
                batch_outputs.append(results)

            # Clear memory
            del results
            self.clear()

            # Report progress
            if verbose:
                processed = min((batch_idx + 1) * batch_size, len(items))
                print(f"  Batch {batch_idx + 1}: {processed}/{len(items)} ({MemoryStats.current()})")

        return batch_outputs


def streaming_parquet_writer(output_dir: Path, prefix: str = "batch"):
    """
    Factory for creating a streaming parquet writer.

    Returns a write function that writes batches to numbered parquet files.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def write(data, batch_idx: int) -> Path:
        if HAS_POLARS:
            if isinstance(data, list):
                data = pl.DataFrame(data)
            batch_path = output_dir / f"{prefix}_{batch_idx:06d}.parquet"
            data.write_parquet(batch_path)
        else:
            import pandas as pd
            if isinstance(data, list):
                data = pd.DataFrame(data)
            batch_path = output_dir / f"{prefix}_{batch_idx:06d}.parquet"
            data.to_parquet(batch_path, index=False)
        return batch_path

    return write


def combine_parquet_batches(
    output_dir: Path,
    final_path: Path,
    prefix: str = "batch",
    cleanup: bool = True
) -> int:
    """
    Combine batch parquet files into single file.

    Uses lazy evaluation - memory efficient.

    Returns number of rows in final file.
    """

    output_dir = Path(output_dir)
    batch_files = sorted(output_dir.glob(f"{prefix}_*.parquet"))

    if not batch_files:
        print(f"  Warning: No batch files with prefix '{prefix}' in {output_dir}")
        return 0

    if HAS_POLARS:
        # Lazy concat - memory efficient
        lazy_frames = [pl.scan_parquet(f) for f in batch_files]
        combined = pl.concat(lazy_frames)

        # Collect and write
        df = combined.collect()
        df.write_parquet(final_path)
        n_rows = len(df)
    else:
        import pandas as pd
        dfs = [pd.read_parquet(f) for f in batch_files]
        combined = pd.concat(dfs, ignore_index=True)
        combined.to_parquet(final_path, index=False)
        n_rows = len(combined)

    # Cleanup batch files
    if cleanup:
        for f in batch_files:
            f.unlink()

    print(f"  Combined {len(batch_files)} batches → {final_path.name} ({n_rows:,} rows)")
    return n_rows


__all__ = [
    'RAMManager',
    'RAMConfig',
    'MemoryStats',
    'streaming_parquet_writer',
    'combine_parquet_batches',
]
