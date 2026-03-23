from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent.parent
DATA_DIR = REPO_ROOT / "Data"
TEST_EXECUTIONS_DIR = REPO_ROOT / "TestExecutions"
EXECUTIONS31_DIR = REPO_ROOT / "31Executions"
ANALYSIS_OUTPUT_DIR = REPO_ROOT / "AlgorithmComparison"
PARETO_OUTPUT_DIR = REPO_ROOT / "pareto_outputs"
PARETO_OUTPUT_TESTS_DIR = REPO_ROOT / "pareto_outputs_tests"
