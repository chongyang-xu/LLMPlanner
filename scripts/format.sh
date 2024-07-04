set -e

yapf -vv --style google -i setup.py
yapf -vv --style google -i llm_planner/*.py
yapf -vv --style google -i llm_planner/*/*.py
yapf -vv --style google -i llm_planner/*/*/*.py
yapf -vv --style google -i tests/*.py
yapf -vv --style google -i experiments/*.py
