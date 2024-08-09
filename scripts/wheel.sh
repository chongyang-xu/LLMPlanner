pip wheel . -w wheels

pip uninstall -y llm_planner
pip install dist/llm_planner-0.0.0-py3-none-any.whl

