import { CodeBlock } from "@/components/CodeBlock";
import { Table } from "@/components/Table";

export default function MontyPage() {
  return (
    <div>
      <h1 className="text-3xl font-bold mb-4">MontyREPL</h1>

      <p className="text-xl text-muted-foreground mb-6 leading-relaxed">
        <strong className="text-foreground">MontyREPL</strong> executes code using{" "}
        <strong className="text-foreground">pydantic-monty</strong>, a Rust-backed interpreter with
        stricter sandboxing than the default LocalREPL. It is a non-isolated environment (runs on the
        host machine), but the interpreter model blocks many unsafe operations by default. To preserve
        state across blocks, MontyREPL replays accumulated code and caches external function calls such as{" "}
        <code className="px-1.5 py-0.5 rounded bg-muted text-foreground text-sm font-semibold">
          llm_query()
        </code>{" "}
        to avoid duplicate network calls and token usage.
      </p>

      <p className="text-muted-foreground mb-2">
        <strong>Prerequisite:</strong>
      </p>
      <CodeBlock language="bash" code={`uv pip install -e ".[monty]"`} />

      <CodeBlock
        code={`rlm = RLM(
    backend="openai",
    backend_kwargs={"model_name": "gpt-5-mini"},
    environment="monty",
    environment_kwargs={
        "setup_code": "x = 1",  # Optional
    },
)`}
      />

      <hr className="my-8 border-border" />

      <h2 className="text-2xl font-semibold mb-4">Arguments</h2>
      <Table
        headers={["Argument", "Type", "Default", "Description"]}
        rows={[
          [<code key="1">setup_code</code>, <code key="2">str</code>, <code key="3">None</code>, "Code to run at initialization"],
          [<code key="4">context_payload</code>, <code key="5">str | dict | list</code>, "Auto", "Initial context (set by RLM)"],
          [<code key="6">lm_handler_address</code>, <code key="7">tuple</code>, "Auto", "Socket address (set by RLM)"],
        ]}
      />

      <hr className="my-8 border-border" />

      <h2 className="text-2xl font-semibold mb-4">How It Works</h2>
      <ol className="list-decimal list-inside text-muted-foreground space-y-1">
        <li>Stores each executed code block in an internal history</li>
        <li>Builds a full replay script by concatenating prior blocks and the new block</li>
        <li>Executes through Monty with declared external functions</li>
        <li>Caches external function results during replay to keep execution idempotent</li>
        <li>Captures stdout/stderr and returns only newly produced output for the current block</li>
      </ol>

      <hr className="my-8 border-border" />

      <h2 className="text-2xl font-semibold mb-4">Behavior Notes</h2>
      <ul className="list-disc list-inside text-muted-foreground space-y-1">
        <li>More constrained interpreter behavior than LocalREPL</li>
        <li>Context and history are injected as Monty inputs</li>
        <li>Supports persistent multi-turn mode in RLM</li>
      </ul>
    </div>
  );
}
