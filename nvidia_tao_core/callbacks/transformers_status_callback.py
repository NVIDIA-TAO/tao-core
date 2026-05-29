# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A transformer trainer status callback for publishing training status to status.json file."""
from datetime import timedelta
import os
import time
from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from nvidia_tao_core.loggers.logging import Status, StatusLogger, Verbosity, get_status_logger, set_status_logger

logger = get_status_logger()


class StatusCallback(TrainerCallback):
    """A [`TrainerCallback`] that handles status callbacks.

    Args:
        results_dir (str): The directory where the logs will be saved.
        verbosity (status_logger.verbosity.Verbosity()): Verbosity level.
        append: True: append if file exists (useful for continuing
            training). False: overwrite existing file,
    """

    def __init__(self, results_dir, verbosity=Verbosity.INFO, append=False, logging_step_interval=50):
        """Instantiate the StatusCallback."""
        # Make sure that the status logger obtained is always
        # an instance of iva.common.logging.logging.StatusLogger.
        # Otherwise, this data get's rendered in stdout.
        if isinstance(get_status_logger(), StatusLogger):
            self.logger = get_status_logger()
        else:
            set_status_logger(StatusLogger(
                filename=os.path.join(results_dir, "status.json"),
                verbosity=verbosity,
                is_master=True,
                append=append)
            )
            self.logger = get_status_logger()
        self.logging_step_interval = logging_step_interval
        self.training_start_time = None
        super().__init__()

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        """Event called during KPI logging event"""
        if state.is_world_process_zero and state.global_step % self.logging_step_interval == 0 and logs:
            # make a shallow copy of logs so we can mutate the fields copied
            # but avoid doing any value pickling.
            shallow_logs = {}
            for k, v in logs.items():
                shallow_logs[k] = v
            _ = shallow_logs.pop("total_flos", None)
            # round numbers so that it looks better in console
            logging_dict = {}
            logging_dict["step"] = state.global_step
            logging_dict["max_step"] = state.max_steps
            logging_dict["max_epoch"] = state.num_train_epochs
            if "epoch" in shallow_logs:
                logging_dict["epoch"] = round(shallow_logs["epoch"], 2)
                shallow_logs.pop("epoch")
            if self.training_start_time:
                elapsed_time = time.time() - self.training_start_time
                if state.global_step > 0 and state.max_steps > 0:
                    avg_step_time = elapsed_time / state.global_step
                    remaining_steps = state.max_steps - state.global_step
                    eta_seconds = max(0, avg_step_time * remaining_steps)
                    logging_dict["eta"] = str(timedelta(seconds=eta_seconds))
            self.logger.kpi = shallow_logs
            self.logger.write(logging_dict, message="Training loop in progress")

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Write data beginning of the training."""
        if not state.is_world_process_zero:
            return
        self.training_start_time = time.time()
        self.logger.write(
            status_level=Status.STARTED,
            message="Starting Training Loop."
        )

    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Event called at the beginning of an epoch."""
        if not state.is_world_process_zero:
            return
        self._epoch_start_time = time.time()

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Event called at the end of an epoch."""
        if not state.is_world_process_zero:
            return
        data = {}
        data["epoch"] = state.epoch
        data["step"] = state.global_step
        data["max_epoch"] = args.num_train_epochs
        data["max_step"] = state.max_steps
        epoch_end_time = time.time()
        time_per_epoch = epoch_end_time - self._epoch_start_time
        eta = (args.num_train_epochs - state.epoch) * time_per_epoch
        data["time_per_epoch"] = str(timedelta(seconds=time_per_epoch))
        data["eta"] = str(timedelta(seconds=eta))
        self.logger.write(data=data, message=f"Epoch {state.epoch} of {args.num_train_epochs} complete")

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Callback function run at the end of training."""
        if not state.is_world_process_zero:
            return
        self.logger.write(
            status_level=Status.RUNNING,
            message="Training loop complete."
        )
