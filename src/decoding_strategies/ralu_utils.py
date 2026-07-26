import ast
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


def _build_cfg(program: str) -> List[Dict[str, Any]]:
    if not program:
        return []

    try:
        cfg = CFGBuilder().build_from_src(name="ralu_path", src=program)
    except SyntaxError:
        return []

    branch_types = (ast.If, ast.For, ast.While, ast.Try)

    nodes = []
    for block in cfg.blocks:
        stmts = block.statements
        if not stmts:
            continue
        nodes.append({
            'id': block.id,
            'lines': (stmts[0].lineno, stmts[-1].end_lineno
            if hasattr(stmts[-1], 'end_lineno')
            else stmts[-1].lineno),
            'source': ast.unparse(stmts[0].parent
                                  if hasattr(stmts[0], 'parent')
                                  else stmts[0]),
            'ast_node': stmts[0],
            'is_branch': any(isinstance(s, branch_types) for s in stmts),
            'successors': [e.target.id for e in block.exits],
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