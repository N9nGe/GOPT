"""
Curriculum learning for 3D bin packing.
Implements difficulty scheduling and box-to-bin ratio metrics.

Based on the principle: Larger boxes relative to bin = easier packing
"""

import numpy as np
from typing import List, Tuple, Optional


class CurriculumScheduler:
    """
    Manages curriculum learning schedule.
    Controls difficulty progression from easy (large boxes) to hard (small boxes).
    """

    def __init__(self,
                 initial_difficulty: float = 0.2,
                 final_difficulty: float = 0.8,
                 curriculum_epochs: int = 400,
                 schedule_type: str = 'linear'):
        """
        Args:
            initial_difficulty: Starting difficulty (0.0 = easiest, 1.0 = hardest)
            final_difficulty: Final difficulty
            curriculum_epochs: Number of epochs to ramp up difficulty
            schedule_type: 'linear', 'exponential', or 'step'
        """
        self.initial_difficulty = initial_difficulty
        self.final_difficulty = final_difficulty
        self.curriculum_epochs = curriculum_epochs
        self.schedule_type = schedule_type
        self.current_epoch = 0

        print(f"Curriculum scheduler initialized:")
        print(f"  Difficulty range: {initial_difficulty:.2f} -> {final_difficulty:.2f}")
        print(f"  Ramp-up epochs: {curriculum_epochs}")
        print(f"  Schedule type: {schedule_type}")

    def get_target_difficulty(self, epoch: Optional[int] = None) -> float:
        """
        Get target difficulty for current epoch.

        Args:
            epoch: Current epoch number (if None, uses internal counter)

        Returns:
            difficulty: Target difficulty value [0.0, 1.0]
        """
        if epoch is not None:
            self.current_epoch = epoch

        # After curriculum is complete, use final difficulty
        if self.current_epoch >= self.curriculum_epochs:
            return self.final_difficulty

        # Calculate progress ratio [0, 1]
        progress = self.current_epoch / self.curriculum_epochs

        # Apply schedule
        if self.schedule_type == 'linear':
            difficulty = self.initial_difficulty + \
                        (self.final_difficulty - self.initial_difficulty) * progress

        elif self.schedule_type == 'exponential':
            # Exponential growth: stays easy longer, then rapid increase
            difficulty = self.initial_difficulty * \
                        (self.final_difficulty / self.initial_difficulty) ** progress

        elif self.schedule_type == 'step':
            # Step function: discrete jumps at 25%, 50%, 75%, 100%
            step_progress = int(progress * 4) / 4
            difficulty = self.initial_difficulty + \
                        (self.final_difficulty - self.initial_difficulty) * step_progress

        else:
            # Default to linear
            difficulty = self.initial_difficulty + \
                        (self.final_difficulty - self.initial_difficulty) * progress

        return np.clip(difficulty, self.initial_difficulty, self.final_difficulty)

    def increment_epoch(self):
        """Increment internal epoch counter"""
        self.current_epoch += 1

    def get_info_dict(self) -> dict:
        """Get dictionary of curriculum info for logging"""
        return {
            'curriculum/epoch': self.current_epoch,
            'curriculum/difficulty': self.get_target_difficulty(),
            'curriculum/progress': min(1.0, self.current_epoch / self.curriculum_epochs)
        }


class BinSizeCurriculumScheduler:
    """
    Manages bin size curriculum learning.
    Progressively introduces larger container sizes during training.

    Strategy: Start with smallest bin(s), gradually add larger bins over time.
    This is independent of box difficulty curriculum.
    """

    def __init__(self,
                 bin_sizes: List[List[int]],
                 curriculum_epochs: int = 400,
                 schedule_type: str = 'linear',
                 sort_by: str = 'volume'):
        """
        Args:
            bin_sizes: List of bin sizes (will be sorted small to large)
            curriculum_epochs: Number of epochs to complete curriculum
            schedule_type: 'linear', 'step', or 'all_at_once'
            sort_by: 'volume' or 'max_dim' - how to sort bin sizes
        """
        self.all_bin_sizes = bin_sizes
        self.curriculum_epochs = curriculum_epochs
        self.schedule_type = schedule_type
        self.current_epoch = 0

        # Sort bin sizes from smallest to largest
        if sort_by == 'volume':
            self.sorted_bin_sizes = sorted(bin_sizes, key=lambda b: b[0] * b[1] * b[2])
        else:  # max_dim
            self.sorted_bin_sizes = sorted(bin_sizes, key=lambda b: max(b))

        self.n_bins = len(self.sorted_bin_sizes)

        print(f"Bin size curriculum scheduler initialized:")
        print(f"  Total bin sizes: {self.n_bins}")
        print(f"  Sorted bin sizes: {self.sorted_bin_sizes}")
        print(f"  Curriculum epochs: {curriculum_epochs}")
        print(f"  Schedule type: {schedule_type}")

    def get_active_bin_sizes(self, epoch: Optional[int] = None) -> List[List[int]]:
        """
        Get list of active bin sizes for current epoch.

        Args:
            epoch: Current epoch number (if None, uses internal counter)

        Returns:
            active_bins: List of bin sizes that should be used
        """
        if epoch is not None:
            self.current_epoch = epoch

        # After curriculum is complete, use all bins
        if self.current_epoch >= self.curriculum_epochs:
            return self.sorted_bin_sizes

        if self.schedule_type == 'linear':
            # Gradually add bins linearly over time
            # Start with 1 bin, end with all bins
            progress = self.current_epoch / self.curriculum_epochs
            n_active = max(1, int(np.ceil(progress * self.n_bins)))
            return self.sorted_bin_sizes[:n_active]

        elif self.schedule_type == 'step':
            # Add bins in discrete steps
            # Divide curriculum into n_bins equal phases
            progress = self.current_epoch / self.curriculum_epochs
            n_active = max(1, int(np.ceil(progress * self.n_bins)))
            return self.sorted_bin_sizes[:n_active]

        elif self.schedule_type == 'all_at_once':
            # Use all bins from the start (no curriculum)
            return self.sorted_bin_sizes

        else:
            # Default to linear
            progress = self.current_epoch / self.curriculum_epochs
            n_active = max(1, int(np.ceil(progress * self.n_bins)))
            return self.sorted_bin_sizes[:n_active]

    def get_n_active_bins(self, epoch: Optional[int] = None) -> int:
        """Get number of active bin sizes"""
        return len(self.get_active_bin_sizes(epoch))

    def increment_epoch(self):
        """Increment internal epoch counter"""
        self.current_epoch += 1

    def get_info_dict(self) -> dict:
        """Get dictionary of bin curriculum info for logging"""
        active_bins = self.get_active_bin_sizes()
        return {
            'bin_curriculum/epoch': self.current_epoch,
            'bin_curriculum/n_active': len(active_bins),
            'bin_curriculum/progress': min(1.0, self.current_epoch / self.curriculum_epochs),
            'bin_curriculum/largest_active': max(b[0] * b[1] * b[2] for b in active_bins) if active_bins else 0
        }


class BoxDifficultyMetrics:
    """
    Metrics for assessing bin packing difficulty.

    Key principle: Larger boxes relative to bin size = easier packing
    """

    @staticmethod
    def box_to_bin_volume_ratio(box_size: Tuple[int, int, int],
                                 bin_size: Tuple[int, int, int]) -> float:
        """
        Compute volume ratio of box to bin.
        Higher ratio = larger box relative to bin = easier

        Args:
            box_size: (length, width, height) of box
            bin_size: (length, width, height) of bin

        Returns:
            ratio: box_volume / bin_volume
        """
        box_vol = box_size[0] * box_size[1] * box_size[2]
        bin_vol = bin_size[0] * bin_size[1] * bin_size[2]
        return box_vol / bin_vol if bin_vol > 0 else 0.0

    @staticmethod
    def box_to_bin_dimensional_ratio(box_size: Tuple[int, int, int],
                                      bin_size: Tuple[int, int, int]) -> float:
        """
        Compute dimension-aware ratio considering shape compatibility.

        For non-cubic bins, this is better than pure volume ratio.
        Uses the geometric mean of per-dimension ratios.

        Higher ratio = larger box relative to bin in ALL dimensions = easier

        Args:
            box_size: (length, width, height) of box
            bin_size: (length, width, height) of bin

        Returns:
            ratio: geometric mean of dimensional ratios
        """
        if bin_size[0] <= 0 or bin_size[1] <= 0 or bin_size[2] <= 0:
            return 0.0

        # Per-dimension ratios
        dim_ratios = [
            box_size[i] / bin_size[i] for i in range(3)
        ]

        # Geometric mean (cube root of product)
        geometric_mean = (dim_ratios[0] * dim_ratios[1] * dim_ratios[2]) ** (1/3)

        return geometric_mean

    @staticmethod
    def avg_box_to_bin_ratio(boxes: List[Tuple[int, int, int]],
                             bin_size: Tuple[int, int, int]) -> float:
        """
        Average box-to-bin volume ratio for a set of boxes.

        Args:
            boxes: List of box sizes
            bin_size: Bin dimensions

        Returns:
            avg_ratio: Average volume ratio
        """
        if len(boxes) == 0:
            return 0.0

        ratios = [BoxDifficultyMetrics.box_to_bin_volume_ratio(box, bin_size)
                  for box in boxes]
        return np.mean(ratios)

    @staticmethod
    def box_size_variance(boxes: List[Tuple[int, int, int]]) -> float:
        """
        Measure variance in box volumes.
        Higher variance = more heterogeneous = potentially harder

        Args:
            boxes: List of box sizes

        Returns:
            normalized_variance: Coefficient of variation (std / mean)
        """
        if len(boxes) == 0:
            return 0.0

        volumes = [b[0] * b[1] * b[2] for b in boxes]
        mean_vol = np.mean(volumes)

        if mean_vol == 0:
            return 0.0

        return np.std(volumes) / mean_vol

    @staticmethod
    def compute_difficulty_score(boxes: List[Tuple[int, int, int]],
                                 bin_size: Tuple[int, int, int]) -> float:
        """
        Compute composite difficulty score for a packing scenario.

        Difficulty increases with:
        - Smaller boxes (lower box-to-bin ratio)
        - Higher variance in box sizes

        Args:
            boxes: List of box sizes
            bin_size: Bin dimensions

        Returns:
            difficulty: Score in [0, 1] where 1 is hardest
        """
        if len(boxes) == 0:
            return 0.5  # Neutral difficulty

        # Box-to-bin ratio: larger boxes = easier
        avg_ratio = BoxDifficultyMetrics.avg_box_to_bin_ratio(boxes, bin_size)
        # Invert: smaller ratio = harder
        ratio_difficulty = 1.0 - min(avg_ratio * 10, 1.0)  # Scale and invert

        # Variance: more heterogeneous = slightly harder
        variance = BoxDifficultyMetrics.box_size_variance(boxes)
        variance_difficulty = min(variance, 1.0)

        # Weighted combination
        difficulty = (
            0.7 * ratio_difficulty +      # Primary factor: box size
            0.3 * variance_difficulty     # Secondary factor: heterogeneity
        )

        return np.clip(difficulty, 0.0, 1.0)


def difficulty_to_box_size_bias(difficulty: float,
                                box_size_set: List[Tuple[int, int, int]],
                                bin_size: Optional[Tuple[int, int, int]] = None) -> np.ndarray:
    """
    Convert difficulty level to sampling weights for box sizes.

    Easy (difficulty ~0.0): Bias toward large boxes RELATIVE TO BIN
    Hard (difficulty ~1.0): Bias toward small boxes RELATIVE TO BIN or uniform

    Args:
        difficulty: Target difficulty [0, 1]
        box_size_set: List of available box sizes
        bin_size: Container size (optional). If provided, uses box-to-bin ratio for better difficulty control.

    Returns:
        weights: Probability weights for each box size (sums to 1)
    """
    n_boxes = len(box_size_set)

    # If bin_size provided, use dimensional ratio (accounts for shape compatibility)
    if bin_size is not None:
        # Use dimensional ratio - better for non-cubic bins
        # This considers how well box dimensions match bin dimensions
        ratios = np.array([
            BoxDifficultyMetrics.box_to_bin_dimensional_ratio(box, bin_size)
            for box in box_size_set
        ])
        max_ratio = ratios.max()
        min_ratio = ratios.min()

        # Normalize ratios to [0, 1]
        if max_ratio > min_ratio:
            norm_values = (ratios - min_ratio) / (max_ratio - min_ratio)
        else:
            norm_values = np.ones(n_boxes) * 0.5
    else:
        # Fallback: use absolute volumes (original behavior)
        volumes = np.array([b[0] * b[1] * b[2] for b in box_size_set])
        max_vol = volumes.max()
        min_vol = volumes.min()

        # Normalize volumes to [0, 1]
        if max_vol > min_vol:
            norm_values = (volumes - min_vol) / (max_vol - min_vol)
        else:
            norm_values = np.ones(n_boxes) * 0.5

    if difficulty < 0.4:
        # Easy: Prefer large boxes (relative to bin)
        # Weight proportional to normalized value
        bias_strength = (0.4 - difficulty) / 0.4  # 1.0 at diff=0, 0.0 at diff=0.4
        weights = 1.0 + bias_strength * norm_values

    elif difficulty > 0.6:
        # Hard: Prefer small boxes (relative to bin)
        # Weight inversely proportional to normalized value
        bias_strength = (difficulty - 0.6) / 0.4  # 0.0 at diff=0.6, 1.0 at diff=1.0
        weights = 1.0 + bias_strength * (1.0 - norm_values)

    else:
        # Medium: Uniform distribution
        weights = np.ones(n_boxes)

    # Normalize to probabilities
    weights = weights / weights.sum()

    return weights


if __name__ == "__main__":
    # Test curriculum scheduler
    print("=== Testing Curriculum Scheduler ===\n")

    scheduler = CurriculumScheduler(
        initial_difficulty=0.2,
        final_difficulty=0.8,
        curriculum_epochs=400,
        schedule_type='linear'
    )

    test_epochs = [0, 50, 100, 200, 300, 400, 500]
    print("\nEpoch -> Difficulty:")
    for epoch in test_epochs:
        difficulty = scheduler.get_target_difficulty(epoch)
        print(f"  Epoch {epoch:3d}: {difficulty:.3f}")

    # Test difficulty metrics
    print("\n=== Testing Difficulty Metrics ===\n")

    bin_size = (10, 10, 10)
    easy_boxes = [(5, 5, 5), (6, 6, 6), (4, 5, 6)]  # Large boxes
    hard_boxes = [(1, 1, 1), (2, 1, 1), (1, 2, 1)]  # Small boxes

    easy_diff = BoxDifficultyMetrics.compute_difficulty_score(easy_boxes, bin_size)
    hard_diff = BoxDifficultyMetrics.compute_difficulty_score(hard_boxes, bin_size)

    print(f"Easy boxes {easy_boxes}: difficulty = {easy_diff:.3f}")
    print(f"Hard boxes {hard_boxes}: difficulty = {hard_diff:.3f}")

    # Test box size bias
    print("\n=== Testing Box Size Bias ===\n")

    box_set = [(1, 1, 1), (3, 3, 3), (5, 5, 5), (7, 7, 7), (9, 9, 9)]
    print(f"Box set volumes: {[b[0]*b[1]*b[2] for b in box_set]}\n")

    for diff in [0.0, 0.3, 0.5, 0.7, 1.0]:
        weights = difficulty_to_box_size_bias(diff, box_set)
        print(f"Difficulty {diff:.1f}: weights = {weights.round(3)}")
