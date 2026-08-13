import ast
import re
import textwrap
from typing import Dict, Any, List

from staticfg import CFGBuilder


_INITIAL_PROGRAM_SYSTEM_PROMPT = """\
    You are a precise reasoning assistant. Given a problem, write a self-contained \
    Python program that solves it step by step.

    Rules:
    - Wrap your program in a <code> </code> block
    - Use only the standard library (no external packages).
    - Every logical step must appear as its own statement or block — do NOT \
    collapse multiple reasoning steps into a single expression.
    - Use clear, descriptive variable names that reflect the meaning of each value.
    - Where the logic branches (if/elif/else) or iterates (for/while), write it \
    explicitly as a separate block — this is critical for later analysis.
    - End the program with a print() call that outputs ONLY the final answer, \
    in the format:  print("Answer:", <result>)
    - Do NOT include explanatory comments or docstrings — the code structure \
    itself must carry the reasoning.
    """

_ALIGNMENT_SYSTEM_PROMPT = """\
    you will receive a question along with a series of logic units describing a control flow.
    Analyze each individual unit and determine if it adheres to the question.
    For each unit there will be instruction signs wrapped in two `#`, such as `#ENTER FUNCTION#`, \
    which should not be regarded as wrong.
    Rules for each unit:
    1. focus only on the current unit. Do not attempt to solve or address issues beyond what \
    is presented in the current unit.
    2. pay attention to the logic only. The unit only reflects the underlying logic, so you \
    ignore syntax issues.
    3. error identification: If there are any logic errors or deviations from the question \
    within this unit - explain what went wrong and provide corrections. Otherwise, simply respond \
    with 'ok'.
    4. Explain why you think this unit is **logically** correct or wrong. Consider \
    the objective of this unit and whether it meets a specific part in the question to support \
    your judgement.
    5. format requirements:
        - First - judge whether the provided unit is **logically** correct. If it is correct, \
    begin your response with 'OK', otherwise begin with 'WRONG'.
        - Second - only when a correction is needed, wrap your fixed unit in `<Fix></Fix>`
    """

_SYNTHESIS_SYSTEM_PROMPT = """\
        You are an expert in writing python code that solves math questions. Your task is to write \
        a correct python program based on the given reasoning path to solve the question \
        by returning `ans`.

        Format your response as:
        <code>
        # Python code, return ans
        ...
        </code>

        Analysis: <brief explanation of how the code follows the reasoning path>
    """

_BRANCH_TYPES = (ast.If, ast.For, ast.While, ast.Try)

_CODE_TAG_RE = re.compile(r"<\s*code\s*>(.*?)<\s*/\s*code\s*>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)


def _extract_program(raw_output: str) -> str:
    if not raw_output:
        return ""

    match = _CODE_TAG_RE.search(raw_output) or _FENCE_RE.search(raw_output)
    if match:
        raw_output = match.group(1)

    return textwrap.dedent(raw_output).strip()


def _walk_blocks(cfg) -> List[Any]:
    roots = [cfg.entryblock]
    roots += [fn.entryblock for fn in getattr(cfg, "functioncfgs", {}).values()]

    blocks, seen, stack = [], set(), list(roots)
    while stack:
        block = stack.pop()
        if block is None or block.id in seen:
            continue
        seen.add(block.id)
        blocks.append(block)
        for link in block.exits:
            stack.append(link.target)

    return sorted(blocks, key=lambda b: b.id)


def _block_source(statements: List[ast.stmt]) -> str:
    parts = []
    for stmt in statements:
        source = ast.unparse(stmt)
        if isinstance(stmt, _BRANCH_TYPES):
            source = source.splitlines()[0]
        parts.append(source)

    return "\n".join(parts)


def _build_cfg(program: str) -> List[Dict[str, Any]]:
    program = _extract_program(program)
    if not program:
        return []

    try:
        cfg = CFGBuilder().build_from_src(name="ralu_path", src=program)
    except (SyntaxError, ValueError, RecursionError):
        return []

    nodes = []
    for block in _walk_blocks(cfg):
        statements = block.statements
        if not statements:
            continue

        last = statements[-1]
        nodes.append({
            'id': block.id,
            'lines': (statements[0].lineno, getattr(last, 'end_lineno', last.lineno)),
            'source': _block_source(statements),
            'ast_node': statements[0],
            'is_branch': any(isinstance(s, _BRANCH_TYPES) for s in statements),
            'successors': [link.target.id for link in block.exits],
        })

    return nodes

def _flush(slices: List[str], units: List[Dict[str, Any]]) -> None:
    if not slices:
        return
    units.append({
        'unit_id': len(units),
        'code': "\n".join(slices),
        'nl_description': "",
    })
    slices.clear()

def _extract_logic_units(cfg: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not cfg:
        return []

    # Topological sort with kahn's algorithm
    node_by_id = {node['id']: node for node in cfg}

    in_deg = {node['id']: 0 for node in cfg}
    for node in cfg:
        for succ_id in node['successors']:
            if succ_id in in_deg:
                in_deg[succ_id] += 1

    queue = [n_id for n_id, deg in in_deg.items() if deg == 0]
    queue.sort()
    topo_order = []

    while queue:
        nid = queue.pop(0)
        topo_order.append(node_by_id[nid])
        for succ_id in sorted(node_by_id[nid]['successors']):
            if succ_id not in in_deg:
                continue
            in_deg[succ_id] -= 1
            if in_deg[succ_id] == 0:
                queue.append(succ_id)

    # fall back to original CFG order if graph is cyclical
    if len(topo_order) < len(cfg):
        topo_order = list(cfg)

    units= []
    code_chunks_left = []

    for node in topo_order:
        if node['is_branch']:
            _flush(code_chunks_left, units)
            units.append({
                'unit_id': len(units),
                'code': node['source'],
                'nl_description': "",
            })
        else:
            code_chunks_left.append(node['source'])

    _flush(code_chunks_left, units)

    return units

def _build_alignment_prompt(unit: Dict[str, Any], prompt: str, verified_units: List[Dict[str, Any]], ) -> str:
    lines = [f"Question: {prompt}", ""]

    if verified_units or unit["unit_id"] == 0:
        lines.append("## Process")

    for v in verified_units:
        lines.append(f"Unit {v['unit_id'] + 1}: {v['code']}")
        if v.get("nl_description"):
            lines.append(v["nl_description"])
        lines.append("")

    lines.append(f"Unit {unit['unit_id'] + 1}: {unit['code']}")

    return "\n".join(lines)