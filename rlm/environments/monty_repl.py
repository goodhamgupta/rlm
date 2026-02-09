import copy
import io
import os
import re
import sys
import time
from typing import Any

from rlm.core.comms_utils import LMRequest, send_lm_request, send_lm_request_batched
from rlm.core.types import REPLResult, RLMChatCompletion
from rlm.environments.base_env import NonIsolatedEnv

Monty: Any | None
try:
    from pydantic_monty import Monty
except ImportError:
    Monty = None


class FDCapture:
    """Capture stdout/stderr at the file descriptor level."""

    def __init__(self, capture_stdout=True, capture_stderr=True):
        self.capture_stdout = capture_stdout
        self.capture_stderr = capture_stderr
        self.stdout_fd = sys.stdout.fileno() if sys.stdout else 1
        self.stderr_fd = sys.stderr.fileno() if sys.stderr else 2
        self.saved_stdout_fd = None
        self.saved_stderr_fd = None
        self.stdout_capture = io.BytesIO()
        self.stderr_capture = io.BytesIO()

    def __enter__(self):
        # Flush buffers
        sys.stdout.flush()
        sys.stderr.flush()

        if self.capture_stdout:
            self.saved_stdout_fd = os.dup(self.stdout_fd)
            self.pipe_out_r, self.pipe_out_w = os.pipe()
            os.dup2(self.pipe_out_w, self.stdout_fd)
            os.close(self.pipe_out_w)

        if self.capture_stderr:
            self.saved_stderr_fd = os.dup(self.stderr_fd)
            self.pipe_err_r, self.pipe_err_w = os.pipe()
            os.dup2(self.pipe_err_w, self.stderr_fd)
            os.close(self.pipe_err_w)

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # Flush C level buffers?
        # sys.stdout.flush()

        if self.capture_stdout:
            # Restore stdout
            os.dup2(self.saved_stdout_fd, self.stdout_fd)
            os.close(self.saved_stdout_fd)
            # Read from pipe
            data = os.read(self.pipe_out_r, 102400)  # Read up to 100KB?
            # If output is larger, this blocks?
            # Correct implementation needs thread/poll to drain pipe.
            # Simplified: assuming small output for RLM.
            # Or loop until empty?
            # Since write end is closed (by us closing pipe_out_w AND dup2 closed the original fd copy?),
            # wait, os.dup2 closes target? Yes.
            # But the process might still have it open?
            # No, we closed pipe_out_w in parent.
            # So read should return EOF when drained.
            while data:
                self.stdout_capture.write(data)
                try:
                    data = os.read(self.pipe_out_r, 4096)
                except OSError:
                    break
            os.close(self.pipe_out_r)

        if self.capture_stderr:
            os.dup2(self.saved_stderr_fd, self.stderr_fd)
            os.close(self.saved_stderr_fd)
            data = os.read(self.pipe_err_r, 102400)
            while data:
                self.stderr_capture.write(data)
                try:
                    data = os.read(self.pipe_err_r, 4096)
                except OSError:
                    break
            os.close(self.pipe_err_r)

    def get_stdout(self):
        return self.stdout_capture.getvalue().decode("utf-8", errors="replace")

    def get_stderr(self):
        return self.stderr_capture.getvalue().decode("utf-8", errors="replace")


class MontyREPL(NonIsolatedEnv):
    """
    Monty-backed REPL environment powered by pydantic-monty.
    Provides a secure, sandboxed Python environment.
    """

    def __init__(
        self,
        lm_handler_address: tuple[str, int] | None = None,
        context_payload: dict | list | str | None = None,
        setup_code: str | None = None,
        persistent: bool = False,
        depth: int = 1,
        **kwargs,
    ):
        if Monty is None:
            raise ImportError(
                'pydantic-monty is required for MontyREPL. Install with `uv pip install -e ".[monty]"`.'
            )
        super().__init__(persistent=persistent, depth=depth, **kwargs)

        self.lm_handler_address = lm_handler_address
        self._context_count: int = 0
        self._history_count: int = 0

        # Monty State Management
        self.code_history: list[str] = []
        self.inputs: dict[str, Any] = {}
        # History of external function calls: (func_name, args_repr, result)
        self.call_history: list[tuple[str, Any, Any]] = []
        # Track length of stdout from previous replays to slice off old output
        self.last_stdout_len: int = 0

        # Setup environment
        self.setup()

        # Load context if provided
        if context_payload is not None:
            self.load_context(context_payload)

        # Run setup code if provided
        if setup_code:
            self.execute_code(setup_code)

    def setup(self):
        """Setup the environment."""
        self._pending_llm_calls: list[RLMChatCompletion] = []

    def _llm_query(self, prompt: str, model: str | None = None) -> str:
        """Query the LM via socket connection to the handler."""
        if not self.lm_handler_address:
            return "Error: No LM handler configured"

        try:
            request = LMRequest(prompt=prompt, model=model, depth=self.depth)
            response = send_lm_request(self.lm_handler_address, request)

            if not response.success:
                return f"Error: {response.error}"

            # Track this LLM call
            self._pending_llm_calls.append(
                response.chat_completion,
            )

            return response.chat_completion.response
        except Exception as e:
            return f"Error: LM query failed - {e}"

    def _llm_query_batched(self, prompts: list[str], model: str | None = None) -> list[str]:
        """Query the LM with multiple prompts concurrently."""
        if not self.lm_handler_address:
            return ["Error: No LM handler configured"] * len(prompts)

        try:
            responses = send_lm_request_batched(
                self.lm_handler_address, prompts, model=model, depth=self.depth
            )

            results = []
            for response in responses:
                if not response.success:
                    results.append(f"Error: {response.error}")
                else:
                    self._pending_llm_calls.append(response.chat_completion)
                    results.append(response.chat_completion.response)

            return results
        except Exception as e:
            return [f"Error: LM query failed - {e}"] * len(prompts)

    def _print(self, *args, **kwargs):
        pass

    def load_context(self, context_payload: dict | list | str):
        """Load context into the environment as context_0 (and 'context' alias)."""
        self.add_context(context_payload, 0)

    def add_context(
        self, context_payload: dict | list | str, context_index: int | None = None
    ) -> int:
        """
        Add a context variable to the inputs.
        """
        if context_index is None:
            context_index = self._context_count

        var_name = f"context_{context_index}"

        # In Monty, we inject variables directly via inputs
        self.inputs[var_name] = context_payload

        # Alias context_0 as 'context'
        if context_index == 0:
            self.inputs["context"] = context_payload

        self._context_count = max(self._context_count, context_index + 1)
        return context_index

    def update_handler_address(self, address: tuple[str, int]) -> None:
        """Update the LM handler address."""
        self.lm_handler_address = address

    def get_context_count(self) -> int:
        """Return the number of contexts loaded."""
        return self._context_count

    def add_history(
        self, message_history: list[dict[str, Any]], history_index: int | None = None
    ) -> int:
        """
        Store a conversation's message history as a versioned variable in inputs.
        """
        if history_index is None:
            history_index = self._history_count

        var_name = f"history_{history_index}"

        history_copy = copy.deepcopy(message_history)
        self.inputs[var_name] = history_copy

        # Alias history_0 as 'history'
        if history_index == 0:
            self.inputs["history"] = history_copy

        self._history_count = max(self._history_count, history_index + 1)
        return history_index

    def get_history_count(self) -> int:
        """Return the number of conversation histories stored."""
        return self._history_count

    def execute_code(self, code: str) -> REPLResult:
        """
        Execute code using Monty.
        Strategy: Concatenate history + new code, and Replay execution.
        We cache external function calls to ensure idempotency and avoid re-billing LLM tokens.
        """
        start_time = time.perf_counter()
        self._pending_llm_calls = []

        # 1. Rewriting for FINAL_VAR
        # We rewrite 'print(FINAL_VAR("x"))' -> 'print(x)' to allow Monty to access the variable directly.
        # We also handle variations with single/double quotes and whitespace.
        code = re.sub(r"print\(FINAL_VAR\(\s*['\"](.*?)['\"]\s*\)\)", r"print(\1)", code)

        # 2. Add to history
        self.code_history.append(code)

        # 3. Construct full code
        full_code = "\n".join(self.code_history)

        # 4. Prepare inputs and external functions
        # We declare all helper functions that might be called by the model.
        external_funcs = ["llm_query", "llm_query_batched", "SHOW_VARS", "FINAL_VAR"]

        # 5. Execution State for Replay
        replay_cursor = 0

        def handle_external_function(func_name: str, args: tuple) -> Any:
            nonlocal replay_cursor

            if replay_cursor < len(self.call_history):
                # Verify match
                cached_name, cached_args, cached_result = self.call_history[replay_cursor]
                replay_cursor += 1
                return cached_result
            else:
                # New Call
                result = None
                if func_name == "llm_query":
                    result = self._llm_query(*args)
                elif func_name == "llm_query_batched":
                    result = self._llm_query_batched(*args)
                elif func_name == "SHOW_VARS":
                    # We can't see internal Monty variables, but we can show inputs.
                    input_vars = list(self.inputs.keys())
                    result = (
                        f"Available input variables: {input_vars}. "
                        "Note: Variables defined within REPL blocks are not listed here but are available in scope."
                    )
                elif func_name == "FINAL_VAR":
                    # If rewriting failed or was missed, we try to look in inputs.
                    var_name = args[0] if args else "unknown"
                    if var_name in self.inputs:
                        result = str(self.inputs[var_name])
                    else:
                        result = (
                            f"Error: Variable '{var_name}' not found in inputs. "
                            "If it was defined in a previous block, use print() instead of FINAL_VAR() "
                            "or ensure it is correctly assigned."
                        )

                self.call_history.append((func_name, args, result))
                replay_cursor += 1
                return result

        # 6. Run Monty with FDCapture
        full_stdout = ""
        full_stderr = ""

        try:
            with FDCapture() as captured:
                input_names = list(self.inputs.keys())

                m = Monty(
                    full_code,
                    inputs=input_names,
                    external_functions=external_funcs,
                )

                if self.inputs:
                    res = m.start(inputs=self.inputs)
                else:
                    res = m.start()

                while hasattr(res, "function_name"):
                    func_name = res.function_name
                    args = res.args
                    ret_val = handle_external_function(func_name, tuple(args))
                    res = res.resume(return_value=ret_val)

            full_stdout = captured.get_stdout()
            full_stderr = captured.get_stderr()

        except Exception as e:
            full_stderr = str(e)

        # 7. Slice Output
        new_stdout = full_stdout[self.last_stdout_len :]

        # Update cursor
        self.last_stdout_len = len(full_stdout)

        return REPLResult(
            stdout=new_stdout,
            stderr=full_stderr,
            locals={},
            execution_time=time.perf_counter() - start_time,
            rlm_calls=self._pending_llm_calls.copy(),
        )

    def cleanup(self):
        self.code_history = []
        self.inputs = {}
        self.call_history = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
