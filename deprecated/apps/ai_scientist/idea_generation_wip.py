from llm_planner.agents.miniLLM_wip import MiniLLM
from llm_planner.agents.SemanticScholar import SemanticScholar

from llm_planner.message import Message
from llm_planner.agents.agent_wip import Agent, start_agents

import os
import json
from os import path as osp

import logging
import signal

import pykka.debug

logging.basicConfig(level=logging.DEBUG)
signal.signal(signal.SIGUSR1, pykka.debug.log_thread_tracebacks)


def extract_json(txt):
    json_start_marker = "```json"
    json_end_marker = "```"

    # Find the start and end indices of the JSON string
    start_index = txt.find(json_start_marker)
    if start_index != -1:
        start_index += len(json_start_marker)  # Move past the marker
        end_index = txt.find(json_end_marker, start_index)
    else:
        return None  # JSON markers not found

    if end_index == -1:
        return None  # End marker not found

    # Extract the JSON string
    json_string = txt[start_index:end_index].strip()
    try:
        parsed_json = json.loads(json_string)
        return parsed_json
    except json.JSONDecodeError:
        return None  # Invalid JSON format


class NoveltyChecker(Agent):

    def __init__(self, agent_name, llm_agent, semantic_search):
        super().__init__(agent_name)
        self.MAX_NOVEL_ITER = 3

        self.llm_agent = llm_agent
        self.semantic_search = semantic_search

        self.add_dependency(llm_agent, semantic_search)

        self.counter = 0
        self.novel_ideas = []

        self.NOVELTY_SYSTEM_PROMPT = """
You are an ambitious AI PhD student who is looking to publish a paper that will contribute significantly to the field.
You have an idea and you want to check if it is novel or not. I.e., not overlapping significantly with existing literature or already well explored.
Be a harsh critic for novelty, ensure there is a sufficient contribution in the idea for a new conference or workshop paper.
You will be given access to the Semantic Scholar API, which you may use to survey the literature and find relevant papers to help you make your decision.
The top 10 results for any search query will be presented to you with the abstracts.

You will be given {num_rounds} to decide on the paper, but you do not need to use them all.
At any round, you may exit early and decide on the novelty of the idea.
Decide a paper idea is novel if after sufficient searching, you have not found a paper that significantly overlaps with your idea.
Decide a paper idea is not novel, if you have found a paper that significantly overlaps with your idea.

{task_description}
<experiment.py>
{code}
</experiment.py>
"""

        self.NOVELTY_USER_PROMPT = '''
Round {current_round}/{num_rounds}.
You have this idea:

"""
{idea}
"""

The results of the last query are (empty on first round):
"""
{last_query_results}
"""

Respond in the following format:

THOUGHT:
<THOUGHT>

RESPONSE:
```json
<JSON>
```

In <THOUGHT>, first briefly reason over the idea and identify any query that could help you make your decision.
If you have made your decision, add "Decision made: novel." or "Decision made: not novel." to your thoughts.

In <JSON>, respond in JSON format with ONLY the following field:
- "Query": An optional search query to search the literature (e.g. attention is all you need). You must make a query if you have not decided this round.

A query will work best if you are able to recall the exact name of the paper you are looking for, or the authors.
This JSON will be automatically parsed, so ensure the format is precise.'''

    async def process(self, message: Message):
        print(f"NoveltyChecker:on_receive: {message}")

        self.counter += 0

        idx = message["idx"]
        total = message["total"]
        idea = message["idea"]
        base_dir = message["base_dir"]

        if "novel" in idea:
            print(f"Skipping idea {idx}, already checked.")
            self.futures[message.id] = None
            return

        task_description_prompt = message["task_description"]
        code = message["code"]

        novel = False
        msg_history = []
        papers_str = ""

        for j in range(self.MAX_NOVEL_ITER):
            print(
                f"NoveltyChecker:on_receive: idea@{idx}/{total} novel_iter@{j}")

            try:
                system_prompt = self.NOVELTY_SYSTEM_PROMPT.format(
                    num_rounds=self.MAX_NOVEL_ITER,
                    task_description=task_description_prompt,
                    code=code,
                )
                user_prompt = self.NOVELTY_USER_PROMPT.format(
                    current_round=j + 1,
                    num_rounds=self.MAX_NOVEL_ITER,
                    idea=idea,
                    last_query_results=papers_str,
                )

                msg_history = [f"User:\n{user_prompt}"]

                msg_hist_str = "\n\n".join(msg_history)
                msg = Message(
                    prompt=f"User:\n{system_prompt}\n\n{msg_hist_str}")
                msg_future = self.llm_agent.ask(msg)

                ret = await msg_future.value()

                text = ret["ret"][0]
                msg_history = msg_history + [f"Assistant:\n{text}"]

                if "decision made: novel" in text.lower():
                    print("Decision made: novel after round", j)
                    novel = True
                    break
                if "decision made: not novel" in text.lower():
                    print("Decision made: not novel after round", j)
                    break

                json_output = extract_json(text)
                assert json_output is not None, "Failed to extract JSON from LLM output"

                query = json_output["Query"]
                qmsg = Message()
                qmsg["query"] = query
                msg_future = self.semantic_search.ask(qmsg)
                ret = await msg_future.value()

                papers = ret["ret"]

                if papers is None:
                    papers_str = "No papers found."

                paper_strings = []
                for i, paper in enumerate(papers):
                    paper_strings.append(
                        """{i}: {title}. {authors}. {venue}, {year}.\nNumber of citations: {cites}\nAbstract: {abstract}"""
                        .format(
                            i=i,
                            title=paper["title"],
                            authors=paper["authors"],
                            venue=paper["venue"],
                            year=paper["year"],
                            cites=paper["citationCount"],
                            abstract=paper["abstract"],
                        ))
                papers_str = "\n\n".join(paper_strings)

            except Exception as e:
                print(f"Error: {e}")
                continue
        # end for j in range(MAX_NOVEL_ITER):

        if novel:
            self.novel_ideas.append(idea)

        if self.counter == total:
            results_file = osp.join(base_dir, "novel_ideas.json")
            with open(results_file, "w") as f:
                json.dump(ideas, f, indent=4)

        self.futures[message.id] = None


class IdeaGenerator(Agent):

    def __init__(self, agent_name, llm_agent):
        super().__init__(agent_name)
        self.NUM_REFLECTIONS = 5
        self.MAX_GEN_IDEAS = 1
        self.llm_agent = llm_agent
        self.semantic_search = None

        self.add_dependency(llm_agent)

        self.IDEA_PROMPT_FIRST = """
{task_description}
<experiment.py>
{code}
</experiment.py>

Here are the ideas that you have already generated:

'''
{idea_examples}
'''

Come up with the next impactful and creative idea for research experiments and directions you can feasibly investigate with the code provided.
Note that you will not have access to any additional resources or datasets.
Make sure any idea is not overfit the specific training dataset or model, and has wider significance.

Respond in the following format:

THOUGHT:
<THOUGHT>

NEW IDEA JSON:
```json
<JSON>
```

In <THOUGHT>, first briefly discuss your intuitions and motivations for the idea. Detail your high-level plan, necessary design choices and ideal outcomes of the experiments. Justify how the idea is different from the existing ones.

In <JSON>, provide the new idea in JSON format with the following fields:
- "Name": A shortened descriptor of the idea. Lowercase, no spaces, underscores allowed.
- "Title": A title for the idea, will be used for the report writing.
- "Experiment": An outline of the implementation. E.g. which functions need to be added or modified, how results will be obtained, ...
- "Interestingness": A rating from 1 to 10 (lowest to highest).
- "Feasibility": A rating from 1 to 10 (lowest to highest).
- "Novelty": A rating from 1 to 10 (lowest to highest).

Be cautious and realistic on your ratings.
This JSON will be automatically parsed, so ensure the format is precise.
You will have {num_reflections} rounds to iterate on the idea, but do not need to use them all.
"""

        self.IDEA_PROMPT_REFLECTION = """
Round {current_round}/{num_reflections}.
In your thoughts, first carefully consider the quality, novelty, and feasibility of the idea you just created.
Include any other factors that you think are important in evaluating the idea.
Ensure the idea is clear and concise, and the JSON is the correct format.
Do not make things overly complicated.
In the next attempt, try and refine and improve your idea.
Stick to the spirit of the original idea unless there are glaring issues.

Respond in the same format as before:
THOUGHT:
<THOUGHT>

NEW IDEA JSON:
```json
<JSON>
```

If there is nothing to improve, simply repeat the previous JSON EXACTLY after the thought and include "I am done" at the end of the thoughts but before the JSON.
ONLY INCLUDE "I am done" IF YOU ARE MAKING NO MORE CHANGES."""

    async def process(self, message: Message):
        print(f"IdeaGenerator:on_receive: {message}")

        BASE_DIR = message["base_dir"]
        EXPERIMENT = message["experiment"]
        assert EXPERIMENT in ["grokking", "nanoGPT"]

        base_dir = osp.join(BASE_DIR, "templates", EXPERIMENT)

        idea_str_archive = []
        with open(osp.join(base_dir, "seed_ideas.json"), "r") as f:
            seed_ideas = json.load(f)
        for seed_idea in seed_ideas:
            idea_str_archive.append(json.dumps(seed_idea))

        # read code
        with open(osp.join(base_dir, "experiment.py"), "r") as f:
            code = f.read()

        # read prompt
        with open(osp.join(base_dir, "prompt.json"), "r") as f:
            prompt = json.load(f)

        idea_system_prompt = prompt["system"]
        task_description_prompt = prompt["task_description"]

        idea_examples = "\n\n".join(idea_str_archive)

        ## idea generation
        ###############################
        for iidx in range(self.MAX_GEN_IDEAS):
            ###############################
            print(f"Generating idea {iidx + 1}/{self.MAX_GEN_IDEAS}")
            try:
                user_prompt = self.IDEA_PROMPT_FIRST.format(
                    task_description=task_description_prompt,
                    code=code,
                    idea_examples=idea_examples,
                    num_reflections=self.NUM_REFLECTIONS)
                system_prompt = idea_system_prompt

                msg_history = [f"User:\n{user_prompt}"]

                msg_hist_str = "\n\n".join(msg_history)
                msg = Message(
                    prompt=f"User:\n{system_prompt}\n\n{msg_hist_str}")

                msg_future = self.llm_agent.ask(msg)

                print("await msg_future.value()")
                ret = await msg_future.value()

                text = ret["ret"][0]
                msg_history = msg_history + [f"Assistant:\n{text}"]

                # parse json output
                json_output = extract_json(text)
                assert json_output is not None, f"Failed to extract JSON from LLM output"

                if self.NUM_REFLECTIONS < 2:
                    continue

                #####################################
                for j in range(self.NUM_REFLECTIONS - 1):
                    print(f"Reflecting idea {iidx + 1} @ {j}th reflection")

                    #####################################
                    # sequential
                    user_prompt = self.IDEA_PROMPT_REFLECTION.format(
                        current_round=j + 2,
                        num_reflections=self.NUM_REFLECTIONS)

                    msg_history = msg_history + [f"User:\n{user_prompt}"]
                    msg_hist_str = "\n\n".join(msg_history)
                    msg = Message(
                        prompt=f"User:\n{system_prompt}\n\n{msg_hist_str}")

                    msg_future = self.llm_agent.ask(msg)

                    ret = await msg_future.value()

                    text = ret["ret"]
                    msg_history = msg_history + [f"Assistant:\n{text}"]

                    print(msg["prompt"])
                    print("-" * 50)
                    print("-" * 50)
                    print(text)
                    json_output = extract_json(text)
                    assert json_output is not None, f"Failed to extract JSON from LLM output"
                    if "I am done" in text:
                        print(
                            f"Idea generation converged after {j + 2} iterations."
                        )
                        break

                idea_str_archive.append(json.dumps(json_output))
            except Exception as e:
                print(f"Failed to generate idea: {e}")
                continue

        ## SAVE IDEAS
        ideas = []
        for idea_str in idea_str_archive:
            ideas.append(json.loads(idea_str))

        with open(osp.join(base_dir, "save_ideas.json"), "w") as f:
            json.dump(ideas, f, indent=4)

        ## CHECK
        self.semantic_search = SemanticScholar()

        print(f"IdeaGenerator:on_receive: start semantic_search")

        futs = []
        for idx, idea in enumerate(ideas):
            msg = Message()
            msg["idx"] = idx
            msg["total"] = len(ideas)
            msg["idea"] = idea

            msg["base_dir"] = base_dir
            msg["code"] = code
            msg["task_description"] = task_description_prompt

            checker = NoveltyChecker(self.llm_agent, self.semantic_search)
            print(f"IdeaGenerator:on_receive: start NoveltyChecker")

            fut = checker.ask(msg)
            futs.append(fut)

        for fut in futs:
            await fut.value()

        self.futures[message.id] = None


###################
# run test
###################
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


async def main():
    minillm = MiniLLM("minillm", max_token=4096)
    idea_generator = IdeaGenerator("minillm", minillm)

    begin = Message(uidx=1)
    begin["base_dir"] = "/home/cxu/ws/llm/LLMPlanner/apps/ai_scientist"
    begin["experiment"] = "grokking"

    fut = idea_generator.ask(begin)
    await fut.value()


start_agents(main)
