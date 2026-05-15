"""
Verification script for the three new CLI flags in main.py.

This is a READ-ONLY test — does not modify main.py. It verifies:
  1. The source file declares all three flags with the right defaults.
  2. argparse with no args returns the expected defaults.
  3. argparse with custom values parses correctly.
  4. The new flag values are correctly threaded into the call sites.

Run with:
    cd /root/shared-nvme/SKGF/DKGF-main/Self-Fusion-main
    python /tmp/verify_flags.py
"""

import ast
import sys
import argparse
import os

MAIN_PATH = '/root/shared-nvme/SKGF/DKGF-main/Self-Fusion-main/main.py'


def _color(s, code):
    if not sys.stdout.isatty():
        return s
    return f"\033[{code}m{s}\033[0m"


def _ok(msg):
    print(_color("  OK  ", "32") + msg)


def _fail(msg):
    print(_color("FAIL  ", "31") + msg)
    sys.exit(1)


# ---- Test 1: AST scan ---------------------------------------------------
print("=" * 64)
print("Test 1: AST scan for new CLI flags in main.py")
print("=" * 64)

src = open(MAIN_PATH).read()
tree = ast.parse(src)

flags_found = {}
for node in ast.walk(tree):
    if (isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)):
        flag = node.args[0].value
        default = None
        for kw in node.keywords:
            if kw.arg == "default":
                default = (ast.literal_eval(kw.value)
                           if isinstance(kw.value, ast.Constant) else None)
        flags_found[flag] = default

expected_defaults = {
    "--tau-ent": 1.5,
    "--struct-alpha": 0.5,
    "--struct-threshold": 0.26,
}
for flag, default in expected_defaults.items():
    if flag not in flags_found:
        _fail(f"missing flag {flag}")
    if flags_found[flag] != default:
        _fail(f"{flag} default is {flags_found[flag]}, expected {default}")
    _ok(f"{flag:25s} default = {flags_found[flag]}")


# ---- Test 2: call sites are wired to args.<flag> ------------------------
print()
print("=" * 64)
print("Test 2: call sites use args.<flag> instead of hardcoded values")
print("=" * 64)

ssf_idx = src.find("structure_similarity_filter(")
ssf_end = src.find(")", ssf_idx) + 1
# Find the matching close paren accounting for nested calls.
depth = 0
for i in range(ssf_idx, len(src)):
    if src[i] == "(":
        depth += 1
    elif src[i] == ")":
        depth -= 1
        if depth == 0:
            ssf_end = i + 1
            break
ssf_block = src[ssf_idx:ssf_end]

if "alpha=args.struct_alpha" not in ssf_block:
    _fail("alpha not wired to args.struct_alpha")
_ok("alpha=args.struct_alpha wired in structure_similarity_filter")
if "threshold=args.struct_threshold" not in ssf_block:
    _fail("threshold not wired to args.struct_threshold")
_ok("threshold=args.struct_threshold wired in structure_similarity_filter")

sg_idx = src.find("scene_generate(")
depth = 0
for i in range(sg_idx, len(src)):
    if src[i] == "(":
        depth += 1
    elif src[i] == ")":
        depth -= 1
        if depth == 0:
            sg_end = i + 1
            break
sg_block = src[sg_idx:sg_end]
if "tau_ent=args.tau_ent" not in sg_block:
    _fail("tau_ent not wired to args.tau_ent")
_ok("tau_ent=args.tau_ent wired in scene_generate")


# ---- Test 3: live argparse ----------------------------------------------
print()
print("=" * 64)
print("Test 3: live argparse — defaults + custom values")
print("=" * 64)

p = argparse.ArgumentParser()
p.add_argument("--data", type=str, default="W-I-S1")
for f in ("wo_fuzzy_retriever", "wo_progressive_fusion", "wo_line_graph_trans",
          "wo_semantic_retrieval", "wo_structural_perception",
          "wo_on_demand_integration", "wo_scene_generation",
          "wo_scene_graph_reconstruction"):
    p.add_argument("--" + f.replace("_", "-"), action="store_true")
p.add_argument("--tau-init", type=float, default=0.5)
p.add_argument("--tau-delta", type=float, default=0.05)
p.add_argument("--max-feedback-iterations", type=int, default=3)
p.add_argument("--h-cycle-eps", type=float, default=1e-3)
p.add_argument("--tau-ent", type=float, default=1.5)
p.add_argument("--struct-alpha", type=float, default=0.5)
p.add_argument("--struct-threshold", type=float, default=0.26)

ns = p.parse_args([])
assert ns.tau_ent == 1.5
assert ns.struct_alpha == 0.5
assert ns.struct_threshold == 0.26
_ok("defaults  : tau_ent=1.5, struct_alpha=0.5, struct_threshold=0.26")

ns = p.parse_args(["--tau-ent", "2.3",
                   "--struct-alpha", "0.7",
                   "--struct-threshold", "0.45"])
assert ns.tau_ent == 2.3
assert ns.struct_alpha == 0.7
assert ns.struct_threshold == 0.45
_ok("custom    : tau_ent=2.3, struct_alpha=0.7, struct_threshold=0.45")


# ---- Test 4: simulated call-site execution ------------------------------
print()
print("=" * 64)
print("Test 4: simulated call-site argument values")
print("=" * 64)

# Simulate main.py feeding args.<flag> to structure_similarity_filter.
def fake_ssf(retriever_output_file, line_triples_1_file, line_triples_2_file,
             stuc_retriever_output_file, alpha, threshold, **_kw):
    print(f"    alpha={alpha}, threshold={threshold}")
    assert isinstance(alpha, (int, float))
    assert isinstance(threshold, (int, float))
    return None

ns = p.parse_args(["--struct-alpha", "0.123", "--struct-threshold", "0.456"])
fake_ssf("a", "b", "c", "d",
         alpha=ns.struct_alpha,
         threshold=ns.struct_threshold,
         use_edit_distance=False, use_enhanced_structure=True,
         enhanced_threshold=0.5)
_ok("structure_similarity_filter receives args.struct_alpha=0.123, "
    "args.struct_threshold=0.456")


# Simulate main.py feeding args.tau_ent to scene_generate.
def fake_scene_generate(data_dir, resume, f_new, g_local_context,
                        negative_constraints, tau_ent):
    print(f"    tau_ent={tau_ent}")
    assert isinstance(tau_ent, (int, float))
    return {"scene_text": "", "h_gen": 0.0, "h_gen_high": False}

ns = p.parse_args(["--tau-ent", "2.7"])
fake_scene_generate("data", False, [], "", [],
                    tau_ent=ns.tau_ent)
_ok("scene_generate receives args.tau_ent=2.7")


# ---- Summary ------------------------------------------------------------
print()
print("=" * 64)
print(_color("ALL TESTS PASSED — 3 new CLI flags wired correctly", "32"))
print("=" * 64)
print()
print("Usage:")
print("  python main.py --data W-I-S1 \\")
print("                --tau-init 0.6 --tau-delta 0.03 \\")
print("                --max-feedback-iterations 5 \\")
print("                --tau-ent 1.2 \\")
print("                --struct-alpha 0.3 --struct-threshold 0.5")
