"""Pipeline stage definitions and state transition tracker.

Enforces strict lifecycle progression as specified in requirements:
INIT -> INPUTS_LOADED -> TEXT_PREPROCESSED -> MODEL_PROMPTED ->
STRUCTURED_OUTPUT_PARSED -> CONFIDENCE_CHECKED -> ROUTED ->
RESPONSE_GENERATED -> RESULTS_SAVED -> EVALUATION_COMPUTED ->
VALIDATION_COMPLETED
"""

from enum import Enum
import logging

logger = logging.getLogger(__name__)


class PipelineStage(str, Enum):
    INIT = "INIT"
    INPUTS_LOADED = "INPUTS_LOADED"
    TEXT_PREPROCESSED = "TEXT_PREPROCESSED"
    MODEL_PROMPTED = "MODEL_PROMPTED"
    STRUCTURED_OUTPUT_PARSED = "STRUCTURED_OUTPUT_PARSED"
    CONFIDENCE_CHECKED = "CONFIDENCE_CHECKED"
    ROUTED = "ROUTED"
    RESPONSE_GENERATED = "RESPONSE_GENERATED"
    RESULTS_SAVED = "RESULTS_SAVED"
    EVALUATION_COMPUTED = "EVALUATION_COMPUTED"
    VALIDATION_COMPLETED = "VALIDATION_COMPLETED"


STAGE_ORDER = [
    PipelineStage.INIT,
    PipelineStage.INPUTS_LOADED,
    PipelineStage.TEXT_PREPROCESSED,
    PipelineStage.MODEL_PROMPTED,
    PipelineStage.STRUCTURED_OUTPUT_PARSED,
    PipelineStage.CONFIDENCE_CHECKED,
    PipelineStage.ROUTED,
    PipelineStage.RESPONSE_GENERATED,
    PipelineStage.RESULTS_SAVED,
    PipelineStage.EVALUATION_COMPUTED,
    PipelineStage.VALIDATION_COMPLETED,
]


class PipelineStageTracker:
    """Tracks and enforces pipeline stage progression."""

    def __init__(self):
        self._current_stage = PipelineStage.INIT
        self._history: list[tuple[PipelineStage, str]] = [(PipelineStage.INIT, "Pipeline initialized")]

    @property
    def current_stage(self) -> PipelineStage:
        return self._current_stage

    @property
    def history(self) -> list[tuple[PipelineStage, str]]:
        return list(self._history)

    def transition_to(self, next_stage: PipelineStage, message: str = "") -> None:
        """Transitions to the next stage, validating that order is strictly preserved."""
        curr_idx = STAGE_ORDER.index(self._current_stage)
        next_idx = STAGE_ORDER.index(next_stage)

        if next_idx <= curr_idx:
            raise ValueError(
                f"Invalid stage transition: cannot move backwards or repeat from "
                f"{self._current_stage.value} to {next_stage.value}."
            )

        # Allow sequential transition or progressing as expected
        self._current_stage = next_stage
        self._history.append((next_stage, message))
        log_msg = f"[Stage Transition] -> {next_stage.value}"
        if message:
            log_msg += f": {message}"
        logger.info(log_msg)
