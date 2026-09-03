"""
Symbolic Calculator Tool: Solves mathematical expressions using SymPy.
Handles equations, algebraic simplification, and symbolic math.

SECURITY: Uses safe expression parsing — sympify() is NEVER called on raw user input.
All input passes through a regex pre-filter and is parsed via parse_expr() with a
restricted locals dict containing only known-safe sympy symbols/functions.
"""

import logging
import platform
import re
import signal
import threading
from typing import Optional

from sympy import (
    Symbol,
    Eq,
    solve,
    simplify,
    factor,
    expand,
    pi,
    E,
    I,
    oo,
    zoo,
    nan,
    sqrt,
    log,
    ln,
    sin,
    cos,
    tan,
    cot,
    sec,
    csc,
    asin,
    acos,
    atan,
    sinh,
    cosh,
    tanh,
)
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
)

logger = logging.getLogger(__name__)

_IS_WINDOWS = platform.system() == "Windows"

# Seconds before a single expression solve is killed (DoS protection)
_SOLVE_TIMEOUT_SECONDS = 5

# ---- Input pre-filter (defense-in-depth) ----
# Blocks obvious code-injection patterns before any parsing occurs.
_BLOCKED_PATTERNS = re.compile(
    r"(__\w+__|"        # dunder attributes: __import__, __class__, etc.
    r"\bimport\b|"      # import keyword
    r"\bexec\b|"        # exec keyword
    r"\beval\b|"        # eval keyword
    r"\bos\.\w|"        # os.system, os.popen, etc.
    r"\bsys\.\w|"       # sys.exit, sys.path, etc.
    r"\bopen\(|"        # file open()
    r"\bcompile\(|"     # compile()
    r"\bgetattr\(|"     # getattr()
    r"\bsetattr\(|"     # setattr()
    r"\bdelattr\(|"     # delattr()
    r"\bglobals\(|"     # globals()
    r"\blocals\(|"      # locals()
    r"\b__import__\b|"  # explicit __import__
    r"[;`]|"           # semicolons, backticks
    r"\bclass\b|"       # class keyword
    r"\blambda\b|"      # lambda keyword
    r"\bdef\b)"         # def keyword
)

# ---- Safe locals: ONLY known-safe sympy symbols/functions ----
_SAFE_MATH_LOCALS = {
    # Constants
    "pi": pi,
    "e": E,
    "E": E,
    "I": I,
    "oo": oo,
    "zoo": zoo,
    "nan": nan,
    # Common functions
    "sqrt": sqrt,
    "log": log,
    "ln": ln,
    "sin": sin,
    "cos": cos,
    "tan": tan,
    "cot": cot,
    "sec": sec,
    "csc": csc,
    "asin": asin,
    "acos": acos,
    "atan": atan,
    "sinh": sinh,
    "cosh": cosh,
    "tanh": tanh,
}

# ---- Transformations for parse_expr (safe subset) ----
_SAFE_TRANSFORMATIONS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,  # Allow ^ as exponent (e.g., x^2)
)


def _pre_filter(expr: str) -> None:
    """
    Defense-in-depth regex check. Raises ValueError if the expression
    contains patterns that suggest code injection rather than math.
    """
    if _BLOCKED_PATTERNS.search(expr):
        raise ValueError(
            f"Rejected: expression contains disallowed pattern. "
            f"Only mathematical expressions are accepted."
        )


def _safe_parse(expr: str):
    """
    Parse a math expression safely: pre-filter, then parse_expr() with
    restricted locals dict. NO access to Python builtins.
    """
    _pre_filter(expr)
    return parse_expr(
        expr,
        local_dict=_SAFE_MATH_LOCALS,
        transformations=_SAFE_TRANSFORMATIONS,
    )


def _with_timeout(func, *args, timeout: int = _SOLVE_TIMEOUT_SECONDS):
    """
    Run func(*args) with a wall-clock timeout.
    Uses signal.alarm on Linux (main thread only).
    Uses threading.Event on Windows (signal.alarm not available).
    Returns (result, None) on success, (None, error_msg) on timeout/error.
    """
    if _IS_WINDOWS:
        # Windows: use threading-based timeout (signal.SIGALRM not available)
        result_container = [None]
        exception_container = [None]
        done_event = threading.Event()

        def _worker():
            try:
                result_container[0] = func(*args)
            except Exception as e:
                exception_container[0] = e
            finally:
                done_event.set()

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        t.join(timeout=timeout)

        if t.is_alive():
            # Thread is still running — timeout occurred
            return None, "Error: expression too complex to solve within time limit (timeout)."

        if exception_container[0] is not None:
            return None, f"Error solving expression: {exception_container[0]}"
        return result_container[0], None
    else:
        # Linux/macOS: use signal.alarm (main thread only)
        def _handler(signum, frame):
            raise TimeoutError("Expression too complex to solve within time limit.")

        old_handler = signal.signal(signal.SIGALRM, _handler)
        signal.alarm(timeout)
        try:
            result = func(*args)
            return result, None
        except TimeoutError:
            return None, "Error: expression too complex to solve within time limit (timeout)."
        except Exception as e:
            return None, f"Error solving expression: {e}"
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)


def solve_expression(expression: str) -> str:
    """
    Parse and solve a mathematical expression.

    Supports:
      - Equations with '=' (e.g. "2*x + 5 = 15")
      - Expressions to simplify (e.g. "(x+1)**2")
      - Factoring (e.g. "factor x**2 - 1")

    Args:
        expression: The mathematical expression string.

    Returns:
        A string showing the steps and the final answer.
    """
    expr = expression.strip()
    if not expr:
        return "Error: Empty expression provided."

    try:
        # Detect factor / expand / simplify commands
        lower = expr.lower()
        if lower.startswith("factor "):
            return _factor_expr(expr[len("factor "):])
        if lower.startswith("expand "):
            return _expand_expr(expr[len("expand "):])
        if lower.startswith("simplify "):
            return _simplify_expr(expr[len("simplify "):])

        # Check if the expression contains '=' (equation)
        if "=" in expr:
            return _solve_equation(expr)

        # Otherwise, try to evaluate / simplify the expression
        return _eval_expression(expr)

    except ValueError as e:
        # Pre-filter rejection or parse error
        logger.warning(f"Calculator rejected input: {e}")
        return f"Error: {e}"
    except Exception as e:
        logger.error(f"Calculator error: {e}")
        return f"Error solving expression: {e}"


def _solve_equation(expr: str) -> str:
    """Solve an equation of the form LHS = RHS."""
    parts = expr.split("=", 1)
    if len(parts) != 2:
        return f"Error: Invalid equation format '{expr}'."

    lhs_str = parts[0].strip()
    rhs_str = parts[1].strip()

    # Parse both sides safely
    lhs, err = _with_timeout(_safe_parse, lhs_str)
    if err:
        return err
    rhs, err = _with_timeout(_safe_parse, rhs_str)
    if err:
        return err

    # Create equation: lhs - rhs = 0
    equation = Eq(lhs, rhs)

    # Find variables
    free_symbols = list(equation.free_symbols)
    if not free_symbols:
        # No variables to solve for
        result = "True" if lhs == rhs else "False"
        return f"{lhs} = {rhs}  ->  {result}"

    # Use the first free symbol as the primary variable
    var = free_symbols[0]

    solutions, err = _with_timeout(solve, equation, var)
    if err:
        return err

    # Format output
    steps = [
        f"Equation: {lhs} = {rhs}",
        f"Rearranging: {lhs} - {rhs} = 0",
    ]

    if len(solutions) == 1:
        steps.append(f"{var} = {solutions[0]}")
    elif len(solutions) > 1:
        steps.append(f"{var} ∈ {{{', '.join(str(s) for s in solutions)}}}")
    else:
        steps.append("No solution found.")

    return " -> ".join(steps)


def _eval_expression(expr: str) -> str:
    """Evaluate/simplify an expression without '='."""
    parsed, err = _with_timeout(_safe_parse, expr)
    if err:
        return err
    result, err = _with_timeout(simplify, parsed)
    if err:
        return err
    return f"{expr}  ->  {result}"


def _factor_expr(expr: str) -> str:
    """Factor an expression."""
    parsed, err = _with_timeout(_safe_parse, expr)
    if err:
        return err
    result, err = _with_timeout(factor, parsed)
    if err:
        return err
    return f"factor({expr})  ->  {result}"


def _expand_expr(expr: str) -> str:
    """Expand an expression."""
    parsed, err = _with_timeout(_safe_parse, expr)
    if err:
        return err
    result, err = _with_timeout(expand, parsed)
    if err:
        return err
    return f"expand({expr})  ->  {result}"


def _simplify_expr(expr: str) -> str:
    """Simplify an expression."""
    parsed, err = _with_timeout(_safe_parse, expr)
    if err:
        return err
    result, err = _with_timeout(simplify, parsed)
    if err:
        return err
    return f"simplify({expr})  ->  {result}"
