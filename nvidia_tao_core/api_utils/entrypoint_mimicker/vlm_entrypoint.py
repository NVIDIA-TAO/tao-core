# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Original source taken from https://github.com/NVIDIA/NeMo

"""VLM entrypoint."""


import os
import re
import sys
import shlex
import subprocess
import threading
from contextlib import contextmanager
from time import time
import logging

try:
    from nvidia_tao_core.telemetry.nvml import get_device_details
    from nvidia_tao_core.telemetry.telemetry import send_telemetry_data
    TELEMETRY_AVAILABLE = True
except ImportError:
    get_device_details = None
    send_telemetry_data = None
    TELEMETRY_AVAILABLE = False

# Configure logging
TAO_LOG_LEVEL = os.getenv('TAO_LOG_LEVEL', 'INFO').upper()
tao_log_level = getattr(logging, TAO_LOG_LEVEL, logging.INFO)
logging.basicConfig(
    level=logging.WARNING,  # Root logger: suppress third-party DEBUG logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logging.getLogger('nvidia_tao_core').setLevel(tao_log_level)
logger = logging.getLogger(__name__)


def convert_dict_to_cli_args(data, parent_key=""):
    """Convert a dictionary to CLI arguments.

    Args:
    - data (dict): The dictionary to convert.
    - parent_key (str, optional): The parent key for nested dictionaries.

    Returns:
    - list: A list of CLI arguments.
    """
    cli_args = []
    for key, value in data.items():
        # Construct the current key path
        if isinstance(value, dict):
            # Recursively process nested dictionaries
            cli_args.extend(convert_dict_to_cli_args(value, key))
        else:
            # Append the CLI argument as --key value
            if str(value):
                cli_args.append(f"--{key}")
                # Handle multi-word strings by adding quotes if needed
                if isinstance(value, str) and (" " in value or "\t" in value):
                    cli_args.append(f'"{value}"')
                else:
                    cli_args.append(str(value))

    return cli_args


def get_default_script_path(specs):
    """Get the default dataloader hook script path based on finetuning mode.

    Args:
    - specs (dict): The specifications dictionary containing train config.

    Returns:
    - str: Path to the default script for the finetuning mode.
    """
    # Determine finetuning mode from train.train_policy.type
    train_config = specs.get("train", {})
    train_policy = train_config.get("train_policy", {})
    policy_type = train_policy.get("type", "sft").lower()

    # Map policy type to script path
    script_paths = {
        "sft": "/opt/cosmos_rl/tao_sft_example.py",
        "grpo": "/opt/cosmos_rl/tao_rl_example.py",
        "rl": "/opt/cosmos_rl/tao_rl_example.py",
    }

    script_path = script_paths.get(policy_type, script_paths["sft"])
    logger.info(f"Detected finetuning mode: {policy_type}, using script: {script_path}")
    return script_path


def handle_custom_script(specs, custom_script_key, target_script_path=None):
    """Handle custom training script provided by user.

    Args:
    - specs (dict): The specifications dictionary that may contain custom_training_script.
    - custom_script_key (str): The key of the custom script in the specifications dictionary.
    - target_script_path (str): The target path where the custom script should be copied.
        If None, uses the default script path based on finetuning mode.

    Returns:
    - str: The script path to use (either user-provided or default).
    """
    # Get user-provided script path if specified
    user_script_path = specs.pop(custom_script_key, None) if custom_script_key in specs else None

    if user_script_path:
        if not os.path.exists(user_script_path):
            logger.error(f"Custom training script not found at: {user_script_path}")
            raise FileNotFoundError(f"Custom training script not found: {user_script_path}")

        logger.info(f"Using user-provided custom script: {user_script_path}")
        return user_script_path

    # Use default script based on finetuning mode
    default_script = get_default_script_path(specs)
    if os.path.exists(default_script):
        logger.info(f"Using default script: {default_script}")
        return default_script

    # Fallback to legacy path if new path doesn't exist
    legacy_script = "/opt/cosmos_rl/custom_sft.py"
    if os.path.exists(legacy_script):
        logger.warning(f"Default script not found, falling back to legacy: {legacy_script}")
        return legacy_script

    logger.warning("No dataloader hook script found, training may fail")
    return None


@contextmanager
def dual_output(log_file=None):
    """Context manager to handle dual output redirection for subprocess.

    Args:
    - log_file (str, optional): Path to the log file. If provided, output will be
      redirected to both sys.stdout and the specified log file. If not provided,
      output will only go to sys.stdout.

    Yields:
    - stdout_target (file object): Target for stdout output (sys.stdout or log file).
    - log_target (file object or None): Target for log file output, or None if log_file
      is not provided.
    """
    if log_file:
        with open(log_file, "a", encoding="utf-8") as f:
            yield sys.stdout, f
    else:
        yield sys.stdout, None


def _build_cosmos_rl_multinode_command(config_path, script_path, train_args):
    """Build command for cosmos-rl multi-node SLURM launch.

    Each node determines its role (controller, policy, rollout) from
    environment variables set by the SLURM sbatch script and runs the
    appropriate cosmos-rl worker.

    Env vars expected (set by slurm_handler sbatch / SLURM runtime):
        NODE_RANK: This node's rank (set from SLURM_NODEID in sbatch)
        SLURM_NODEID: This node's rank (set by Slurm, forwarded via container-env)
        SLURMD_NODENAME: This node's hostname (set by Slurm, forwarded via container-env)
        NUM_POLICY_NODES: Number of policy nodes
        NUM_ROLLOUT_NODES: Number of rollout nodes
        POLICY_NODES: Space-separated hostnames of policy nodes
        ROLLOUT_NODES: Space-separated hostnames of rollout nodes
        COSMOS_CONTROLLER_HOST: 'hostname:port' of the controller
        CONTROLLER_PORT: Port for the controller
        NODE_LAUNCH_METADATA_POLICY: JSON metadata for policy nodes
        NODE_LAUNCH_METADATA_ROLLOUT: JSON metadata for rollout nodes
    """
    node_id = int(os.environ.get("SLURM_NODEID", os.environ.get("NODE_RANK", "0")))
    n_policy_nodes = int(os.environ.get("NUM_POLICY_NODES", "1"))
    controller_port = os.environ.get("CONTROLLER_PORT", "8082")

    # Determine this node's role.
    # With the 3-srun pattern, COSMOS_NODE_ROLE is explicitly set per-srun.
    # Fall back to SLURM_NODEID-based detection for single-srun compat.
    explicit_role = os.environ.get("COSMOS_NODE_ROLE", "")
    if explicit_role in ("controller", "policy", "rollout"):
        role = explicit_role
        if role in ("controller", "policy"):
            local_node_list = os.environ.get("POLICY_NODES", "")
        else:
            local_node_list = os.environ.get("ROLLOUT_NODES", "")
    elif node_id < n_policy_nodes:
        role = "policy"
        local_node_list = os.environ.get("POLICY_NODES", "")
    else:
        role = "rollout"
        local_node_list = os.environ.get("ROLLOUT_NODES", "")

    # Set LOCAL_NODE_LIST for cosmos_rl_slurm_launch.py
    os.environ["LOCAL_NODE_LIST"] = local_node_list

    logger.info(
        f"[COSMOS-RL MULTINODE] Node {node_id}, role={role}, "
        f"controller={os.environ.get('COSMOS_CONTROLLER_HOST', 'unknown')}, "
        f"LOCAL_NODE_LIST={local_node_list}"
    )

    # Discover the cosmos_rl package directory inside the container.
    # The container may have multiple cosmos_rl installations (e.g.
    # /workspace/cosmos_rl and /workspace/cosmos_rl_merged/cosmos_rl).
    # We need the one that actually contains the launcher scripts, so
    # iterate __path__ entries and pick the first that has them.
    # Falls back to known locations if none match.
    cosmos_pkg_cmd = r'''COSMOS_RL_PKG=$(python -c "
import cosmos_rl, os
for p in cosmos_rl.__path__:
    if os.path.isfile(os.path.join(p, 'tools', 'slurm', 'cosmos_rl_slurm_launch.py')):
        print(p); break
else:
    for fallback in ['/workspace/cosmos_rl_merged/cosmos_rl', '/workspace/cosmos_rl']:
        if os.path.isfile(os.path.join(fallback, 'tools', 'slurm', 'cosmos_rl_slurm_launch.py')):
            print(fallback); break
    else:
        print(list(cosmos_rl.__path__)[0])
")'''

    # Build the launcher script and args.
    # When a custom hook script (.py file) is provided, it *replaces* the
    # default launcher module (cosmos_rl.dispatcher.run_web_panel).
    # This matches how cosmos_rl's own dispatch_job.py passes the launcher
    # arg — the custom script becomes the entrypoint for both controller
    # and workers, and internally calls cosmos_rl.launcher.worker_entry.run().
    if script_path:
        launcher_script = script_path
    else:
        launcher_script = "cosmos_rl.dispatcher.run_web_panel"
    launcher_args_str = ""

    # All nodes need this preamble
    # Activate the cosmos_rl venv if it exists — the system Python may only
    # have a namespace-packaged cosmos_rl with limited dependencies, while the
    # venv has the full environment needed by launcher scripts.
    preamble = f"""set -e
export COSMOS_LOG_LEVEL=DEBUG
if [ -f /opt/venv/cosmos_rl/bin/activate ]; then
    source /opt/venv/cosmos_rl/bin/activate
fi
{cosmos_pkg_cmd}
echo "COSMOS_RL_PKG=$COSMOS_RL_PKG"
export LOCAL_NODE_LIST="{local_node_list}"
"""

    if role == "controller":
        # 3-srun pattern: dedicated controller srun (runs ONLY the controller)
        launch_cmd = f"""{preamble}
echo "[Node {node_id}] Starting controller on port {controller_port}"
export COSMOS_LOG_LEVEL=DEBUG
bash $COSMOS_RL_PKG/launcher/launch_controller.sh \\
    --port {controller_port} \\
    --config {config_path} \\
    --script {launcher_script} {launcher_args_str}
"""
    elif explicit_role:
        # 3-srun pattern: dedicated policy or rollout srun
        launch_cmd = f"""{preamble}
echo "[Node {node_id}] Starting {role} worker"
python $COSMOS_RL_PKG/tools/slurm/cosmos_rl_slurm_launch.py \\
    --type {role} \\
    --config {config_path} \\
    {launcher_script} {launcher_args_str}
"""
    elif node_id == 0:
        # Single-srun fallback: Node 0 runs controller + policy together
        launch_cmd = f"""{preamble}
echo "[Node {node_id}] Starting controller on port {controller_port}"
bash $COSMOS_RL_PKG/launcher/launch_controller.sh \\
    --port {controller_port} \\
    --config {config_path} \\
    --script {launcher_script} {launcher_args_str} &
controller_pid=$!

sleep 3

echo "[Node {node_id}] Starting policy worker"
set +e
python $COSMOS_RL_PKG/tools/slurm/cosmos_rl_slurm_launch.py \\
    --type policy \\
    --config {config_path} \\
    {launcher_script} {launcher_args_str}
policy_exit=$?
set -e

echo "[Node {node_id}] Policy worker exited with code $policy_exit"
kill $controller_pid 2>/dev/null || true
wait $controller_pid 2>/dev/null || true
exit $policy_exit
"""
    else:
        # Single-srun fallback: other nodes run their detected role
        launch_cmd = f"""{preamble}
echo "[Node {node_id}] Starting {role} worker"
python $COSMOS_RL_PKG/tools/slurm/cosmos_rl_slurm_launch.py \\
    --type {role} \\
    --config {config_path} \\
    {launcher_script} {launcher_args_str}
"""

    logger.info(f"[COSMOS-RL MULTINODE] Launch command for node {node_id}:\n{launch_cmd}")
    return ["/bin/bash", "-c", launch_cmd]


def vlm_launch(neural_network_name, action, specs, job_id=""):
    """Launch a VLM model.

    Args:
    - neural_network_name (str): The name of the neural network.
    - action (str): The action to perform.
    - specs (dict): The specifications for the action.
    """
    command = []
    if 'lepton_specs' in specs:
        lepton_specs = specs.pop('lepton_specs')
        lepton_args = ['--lepton-mode']
        lepton_args += convert_dict_to_cli_args(lepton_specs)
        lepton_args = " ".join(lepton_args)
    else:
        lepton_args = ""
    if neural_network_name == "cosmos-rl" and action in ["train", "evaluate"]:
        # Handle custom training script or use default based on finetuning mode
        script_path = None
        if action == "train":
            script_path = handle_custom_script(specs, "custom_script")

        train_args = ""
        if action == "train" and script_path:
            train_args = f"{lepton_args} {script_path}"
        # Use TAO_API_RESULTS_DIR for SLURM compatibility, fallback to /results
        results_base = os.getenv('TAO_API_RESULTS_DIR', '/results')
        logger.info(f"results_base: {results_base}")
        config_path = f"{results_base}/{job_id}/spec.toml"

        # Check for multi-node SLURM mode
        is_multinode = os.environ.get("COSMOS_RL_MULTINODE") == "1"

        if is_multinode:
            command = _build_cosmos_rl_multinode_command(
                config_path, script_path, train_args
            )
        else:
            suffix = f"-{action}" if action != "train" else ""
            launch_cmd = (
                f"{neural_network_name}{suffix} --config {config_path} {train_args}"
            )
            logger.info(f"launch_cmd: {launch_cmd}")
            command = ["/bin/bash", "-c", launch_cmd]
    else:
        cli_args = convert_dict_to_cli_args(specs)
        cli_args = " ".join(cli_args)
        call = f"{neural_network_name}-{action} {cli_args}"
        command = shlex.split(call)
    process_passed = False
    try:
        # Run the script.
        log_file = ""
        if os.getenv("JOB_ID"):
            # Use TAO_API_RESULTS_DIR for SLURM compatibility, fallback to /results
            logs_dir = os.getenv('TAO_MICROSERVICES_TTY_LOG') or os.getenv('TAO_API_RESULTS_DIR', '/results')
            log_file = f"{logs_dir}/{os.getenv('JOB_ID')}/microservices_log.txt"

        progress_bar_pattern = re.compile(r"Epoch \d+: \s*\d+%|\[.*\]")
        start = time()
        logger.info(f"command: {command}")
        with dual_output(log_file) as (stdout_target, log_target):
            proc = subprocess.Popen(  # pylint: disable=R1732
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,  # Capture stderr separately
                bufsize=1,  # Line-buffered
                universal_newlines=True  # Text mode
            )
            last_progress_bar_line = None

            def handle_stderr():
                """Thread to handle stderr output separately to ensure errors are captured."""
                try:
                    for line in proc.stderr:
                        # Prefix stderr with [STDERR] to distinguish error output
                        error_line = f"[STDERR] {line}" if not line.startswith('[STDERR]') else line
                        stdout_target.write(error_line)
                        stdout_target.flush()
                        if log_target:
                            log_target.write(error_line)
                            log_target.flush()
                except Exception as e:
                    logger.error(f"Error in stderr handler thread: {e}")

            # Start stderr handler thread
            stderr_thread = threading.Thread(target=handle_stderr, daemon=True)
            stderr_thread.start()

            # Handle stdout in main thread
            for line in proc.stdout:
                # Check if the line contains \r or matches the progress bar pattern
                if '\r' in line or progress_bar_pattern.search(line):
                    last_progress_bar_line = line.strip()
                    # Print the progress bar line to the terminal
                    stdout_target.write('\r' + last_progress_bar_line)
                    stdout_target.flush()
                else:
                    # Write the final progress bar line to the log file before a new log line
                    if last_progress_bar_line:
                        if log_target:
                            log_target.write(last_progress_bar_line + '\n')
                            log_target.flush()
                        last_progress_bar_line = None
                    stdout_target.write(line)
                    stdout_target.flush()
                    if log_target:
                        log_target.write(line)
                        log_target.flush()

            proc.wait()  # Wait for the process to complete
            stderr_thread.join(timeout=5)  # Wait for stderr thread to finish processing

            # Write the final progress bar line after process completion
            if last_progress_bar_line and log_target:
                log_target.write(last_progress_bar_line + '\n')
                log_target.flush()

            # Log the return code for debugging
            if proc.returncode != 0:
                logger.error(f"Process exited with return code: {proc.returncode}")

            if proc.returncode == 0:
                process_passed = True

    except (KeyboardInterrupt, SystemExit):
        logger.warning("Command was interrupted")
        process_passed = True
    except subprocess.CalledProcessError as e:
        if e.output is not None:
            logger.error(e.output)
        process_passed = False
    except Exception as e:
        logger.error(f"Error: {e}")
        process_passed = False

    end = time()
    time_lapsed = int(end - start)

    if not TELEMETRY_AVAILABLE:
        logging.info("Telemetry module not available; skipping telemetry reporting.")
    else:
        try:
            gpu_data = []
            for device in get_device_details():
                gpu_data.append(device.get_config())
            logging.info("Sending telemetry data.")
            send_telemetry_data(
                neural_network_name,
                action,
                gpu_data,
                time_lapsed=time_lapsed,
                pass_status=process_passed
            )
        except Exception as e:
            logging.warning("Telemetry data couldn't be sent, but the command ran successfully.")
            logging.warning(f"[Error]: {e}")

    if not process_passed:
        logger.error("Execution status: FAIL")
        return False

    logger.info("Execution status: PASS")
    return True
