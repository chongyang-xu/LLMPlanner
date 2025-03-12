set -e

yapf -vv --style google -i setup.py
yapf -vv --style google -i llm_planner/*.py
yapf -vv --style google -i llm_planner/*/*.py
yapf -vv --style google -i llm_planner/*/*/*.py

yapf -vv --style google -i apps/agents/*.py
yapf -vv --style google -i apps/templates/*.py

yapf -vv --style google -i tests/agents/*.py
yapf -vv --style google -i tests/service/*.py
yapf -vv --style google -i tests/templates/*.py
yapf -vv --style google -i experiments/*/*.py

yapf -vv --style google -i apps/pz_demo/*.py
