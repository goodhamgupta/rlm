import pytest

pytest.importorskip("pydantic_monty")

from rlm.environments.monty_repl import MontyREPL


class TestMontyREPL:
    def test_simple_execution(self, capsys):
        with capsys.disabled():
            repl = MontyREPL()
            result = repl.execute_code("print('Hello Monty')")
            assert "Hello Monty" in result.stdout
            assert result.stderr == ""

    def test_persistence(self, capsys):
        with capsys.disabled():
            repl = MontyREPL()
            repl.execute_code("x = 42")
            result = repl.execute_code("print(x)")
            assert "42" in result.stdout

    def test_context_injection(self, capsys):
        with capsys.disabled():
            repl = MontyREPL(context_payload={"foo": "bar"})
            result = repl.execute_code("print(context['foo'])")
            assert "bar" in result.stdout

    def test_history_injection(self, capsys):
        with capsys.disabled():
            repl = MontyREPL()
            repl.add_history([{"role": "user", "content": "hi"}])
            result = repl.execute_code("print(history[0]['content'])")
            assert "hi" in result.stdout

    def test_final_var_rewriting(self, capsys):
        with capsys.disabled():
            repl = MontyREPL()
            repl.execute_code("answer = 100")
            # Simulate what find_final_answer generates
            code = "print(FINAL_VAR('answer'))"
            result = repl.execute_code(code)
            assert "100" in result.stdout

    def test_context_manager(self):
        with MontyREPL() as repl:
            repl.execute_code("print('ok')")

    def test_syntax_error(self, capsys):
        with capsys.disabled():
            repl = MontyREPL()
            result = repl.execute_code("def broken(")
            # Monty returns "unexpected EOF..." which is a syntax error description
            assert "unexpected EOF" in result.stderr or "SyntaxError" in result.stderr

    def test_import_failure_expected(self, capsys):
        # Monty blocks imports by default
        with capsys.disabled():
            repl = MontyREPL()
            result = repl.execute_code("import math")
            assert "ModuleNotFoundError" in result.stderr

    def test_show_vars(self, capsys):
        with capsys.disabled():
            repl = MontyREPL(context_payload={"a": 1})
            result = repl.execute_code("print(SHOW_VARS())")
            assert "Available input variables" in result.stdout
            assert "'context'" in result.stdout
