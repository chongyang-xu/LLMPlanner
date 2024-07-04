pip wheel . -w wheels

pip uninstall -y llm_planner
pip install wheels/llm_planner-0.0.0-py3-none-any.whl

