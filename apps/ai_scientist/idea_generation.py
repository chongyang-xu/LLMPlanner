import os
import json
from os import path as osp

from llm_planner.message import Message
from llm_planner.actor.system import System
from llm_planner.actor.agent import Agent

from llm_planner.agents.miniLLM import MiniLLM
from llm_planner.agents.Llama3_8B import Llama3_8B
from llm_planner.agents.SemanticScholar import SemanticScholar


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

    def __init__(self, llm_agent, semantic_search):
        super().__init__()
        self.MAX_NOVEL_ITER = 3

        self.llm_agent = llm_agent
        self.semantic_search = semantic_search

        # self.add_dependency(llm_agent, semantic_search)

        self.total_ideas = 0
        self.novel_ideas = []
        self.all_codes = {}
        self.all_ideas = {}
        # book keeping if idx is novel
        self.idx_novel = {}
        # book keeping idx -> history
        self.msg_history = {}
        self.papers_str = {}

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

    async def process(self, sender_id, message: Message):
        # print(f"NoveltyChecker:on_receive: {message}")

        if message["response"] is not None:
            rmsg = message["request_message"]
            idx = rmsg["idx"]
            total = rmsg["total"]
            novel_iter = rmsg["novel_iter"]

            idea = self.all_ideas[idx]
            code = self.all_codes[idx]

            if rmsg["to_llm_agent"] is not None:
                j = novel_iter

                text = message["response"]

                self.msg_history[idx] = self.msg_history[idx] + [{
                    "role": "assistant",
                    "content": f"{text}"
                }]

                if "decision made: novel" in text.lower():
                    print("Decision made: novel after round", j)
                    self.idx_novel[idx] = True
                    self.save_novel_ideas()
                    return

                if "decision made: not novel" in text.lower():
                    print("Decision made: not novel after round", j)
                    self.idx_novel[idx] = False
                    self.save_novel_ideas()
                    return

                json_output = extract_json(text)
                # print(json_output)
                assert json_output is not None, "Failed to extract JSON from LLM output"

                query = json_output["Query"]
                qmsg = Message()
                qmsg["query"] = query
                qmsg["idx"] = idx
                qmsg["total"] = total
                qmsg["novel_iter"] = j + 1
                qmsg["to_semantic_search"] = True

                self.send(self.semantic_search.id, qmsg)

            elif rmsg["to_semantic_search"] is not None:
                papers = message["response"]

                papers_str = ""
                if papers is None:
                    papers_str = "No papers found."
                else:
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

                j = rmsg["novel_iter"] + 1
                if j >= self.MAX_NOVEL_ITER:
                    print("Decision made: not novel after round", j)
                    self.idx_novel[idx] = False
                    self.save_novel_ideas()
                    return

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

                    self.msg_history[idx] = self.msg_history[idx] + [{
                        "role": "user",
                        "content": f"{user_prompt}"
                    }]

                    msg = Message()
                    msg["content"] = [{
                        "role": "system",
                        "content": f"{system_prompt}"
                    }] + self.msg_history[idx]
                    msg["idx"] = idx
                    msg["total"] = total
                    msg["novel_iter"] = j
                    msg["to_llm_agent"] = True

                    self.send(self.llm_agent.id, msg)
                except Exception as e:
                    print(f"Error: {e} {idx}@{j}")

        elif message["novel_checker"] is not None:
            # self.counter += 1

            idx = message["idx"]
            total = message["total"]
            idea = message["idea"]
            self.base_dir = message["base_dir"]

            self.all_ideas[idx] = idea
            self.total_ideas = total

            if idx in self.idx_novel:
                print(f"Skipping idea {idx}, already checked.")
                return

            self.task_description_prompt = message["task_description"]
            code = message["code"]
            self.all_codes[idx] = code

            self.msg_history[idx] = []
            self.papers_str[idx] = ""

            j = 0
            print(
                f"NoveltyChecker:on_receive: idea@{idx}/{total} novel_iter@{j}")

            try:
                system_prompt = self.NOVELTY_SYSTEM_PROMPT.format(
                    num_rounds=self.MAX_NOVEL_ITER,
                    task_description=self.task_description_prompt,
                    code=code,
                )
                user_prompt = self.NOVELTY_USER_PROMPT.format(
                    current_round=j + 1,
                    num_rounds=self.MAX_NOVEL_ITER,
                    idea=idea,
                    last_query_results=self.papers_str[idx],
                )

                self.msg_history[idx] = self.msg_history[idx] + [{
                    "role": "user",
                    "content": f"{user_prompt}"
                }]

                msg = Message()
                msg["content"] = [{
                    "role": "system",
                    "content": f"{system_prompt}"
                }] + self.msg_history[idx]
                msg["idx"] = idx
                msg["total"] = total
                msg["novel_iter"] = 0
                msg["to_llm_agent"] = True

                self.send(self.llm_agent.id, msg)
            except Exception as e:
                print(f"Error: {e} {idx}@{j}")

    def save_novel_ideas(self):
        novel_ideas = []
        if self.total_ideas == len(self.all_ideas):
            if self.total_ideas == len(self.idx_novel):
                for idx, novel in self.idx_novel.items():
                    if novel:
                        novel_ideas.append(self.all_ideas[idx])

                results_file = osp.join(self.base_dir, "novel_ideas.json")
                with open(results_file, "w") as f:
                    json.dump(novel_ideas, f, indent=4)


class IdeaGenerator(Agent):

    def __init__(self, llm_agent):
        super().__init__()
        self.NUM_REFLECTIONS = 5
        self.MAX_GEN_IDEAS = 1
        self.llm_agent = llm_agent
        self.semantic_search = None

        # dictionary generating idx to history
        self.msg_history = {}
        # dictionary generating idx to reflection idx
        self.reflection_idx = {}

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

    async def process(self, sender_id, message: Message):
        # print(f"IdeaGenerator:on_receive: {message}")

        if message["response"] is not None:
            rmsg = message["request_message"]
            gen_iidx = rmsg["gen_iidx"]
            reflection_idx = rmsg["reflection_idx"]
            system_prompt = rmsg["system_prompt"]

            if rmsg["reflection_idx"] is None:
                # print("resp: ", rmsg["content"][-100:])
                text = message["response"]
                # print("resp: text: ", text)

                self.msg_history[gen_iidx] = self.msg_history[gen_iidx] + [{
                    "role": "assistant",
                    "content": f"{text}"
                }]

                # parse json output
                json_output = extract_json(text)
                # print(text)
                assert json_output is not None, f"Failed to extract JSON from LLM output"

                if self.NUM_REFLECTIONS < 2:
                    return

                # start reflecting
                j = 0
                print(f"Reflecting idea {gen_iidx + 1} @ {j}th reflection")

                #####################################
                # sequential
                user_prompt = self.IDEA_PROMPT_REFLECTION.format(
                    current_round=j + 2, num_reflections=self.NUM_REFLECTIONS)

                self.msg_history[gen_iidx] = self.msg_history[gen_iidx] + [{
                    "role": "user",
                    "content": f"{user_prompt}"
                }]

                msg = Message()
                msg["content"] = [{
                    "role": "system",
                    "content": f"{system_prompt}"
                }] + self.msg_history[gen_iidx]
                msg["gen_iidx"] = gen_iidx
                msg["reflection_idx"] = j

                self.send(self.llm_agent.id, msg)

            elif rmsg["reflection_idx"] is not None:
                j = reflection_idx
                text = message["response"]
                self.msg_history[gen_iidx] = self.msg_history[gen_iidx] + [{
                    "role": "assistant",
                    "content": f"{text}"
                }]

                # print(text)
                json_output = extract_json(text)
                assert json_output is not None, f"Failed to extract JSON from LLM output"
                if "I am done" in text:
                    print(
                        f"Idea generation converged after {j + 2} iterations.")
                    self.idea_str_archive.append(json.dumps(json_output))
                    self.save_and_check_ideas()
                    return

                if reflection_idx + 1 >= self.NUM_REFLECTIONS:
                    # finished after NUM_REFLECTIONS attempts
                    self.idea_str_archive.append(json.dumps(json_output))
                    self.save_and_check_ideas()
                    return
                else:
                    j = reflection_idx + 1
                    print(f"Reflecting idea {gen_iidx + 1} @ {j}th reflection")

                    #####################################
                    # sequential
                    user_prompt = self.IDEA_PROMPT_REFLECTION.format(
                        current_round=j + 2,
                        num_reflections=self.NUM_REFLECTIONS)

                    self.msg_history[gen_iidx] = self.msg_history[gen_iidx] + [{
                        "role": "user",
                        "content": f"{user_prompt}"
                    }]

                    msg = Message()
                    msg["content"] = [{
                        "role": "system",
                        "content": f"{system_prompt}"
                    }] + self.msg_history[gen_iidx]
                    msg["gen_iidx"] = gen_iidx
                    msg["reflection_idx"] = j

                    self.send(self.llm_agent.id, msg)

        elif message["experiment"] is not None:

            BASE_DIR = message["base_dir"]
            EXPERIMENT = message["experiment"]
            assert EXPERIMENT in ["grokking", "nanoGPT"]

            self.base_dir = osp.join(BASE_DIR, "templates", EXPERIMENT)

            self.idea_str_archive = []
            with open(osp.join(self.base_dir, "seed_ideas.json"), "r") as f:
                seed_ideas = json.load(f)
            for seed_idea in seed_ideas:
                self.idea_str_archive.append(json.dumps(seed_idea))
            self.num_seed_ideas = len(seed_ideas)

            # read code
            with open(osp.join(self.base_dir, "experiment.py"), "r") as f:
                self.code = f.read()

            # read prompt
            with open(osp.join(self.base_dir, "prompt.json"), "r") as f:
                prompt = json.load(f)

            idea_system_prompt = prompt["system"]
            self.task_description_prompt = prompt["task_description"]

            idea_examples = "\n\n".join(self.idea_str_archive)

            ## idea generation
            ###############################
            for iidx in range(self.MAX_GEN_IDEAS):
                ###############################
                print(f"Generating idea {iidx + 1}/{self.MAX_GEN_IDEAS}")
                try:
                    user_prompt = self.IDEA_PROMPT_FIRST.format(
                        task_description=self.task_description_prompt,
                        code=self.code,
                        idea_examples=idea_examples,
                        num_reflections=self.NUM_REFLECTIONS)
                    system_prompt = idea_system_prompt

                    self.msg_history[iidx] = [{
                        "role": "user",
                        "content": f"{user_prompt}"
                    }]

                    msg = Message()
                    msg["content"] = [{
                        "role": "system",
                        "content": f"{system_prompt}"
                    }] + self.msg_history[iidx]
                    # print('-'*50)
                    # print(msg["content"])
                    # print('-'*50)
                    msg["gen_iidx"] = iidx
                    msg["system_prompt"] = system_prompt
                    self.send(self.llm_agent.id, msg)
                except Exception as e:
                    print(f"Failed to generate idea: {e}")
                    print(self.msg_history[iidx])
                    import traceback
                    traceback.print_exc()
                    continue

    def save_and_check_ideas(self):
        if len(self.idea_str_archive
              ) < self.MAX_GEN_IDEAS + self.num_seed_ideas:
            return

        ## SAVE IDEAS
        self.ideas = []
        for idea_str in self.idea_str_archive:
            self.ideas.append(json.loads(idea_str))

        with open(osp.join(self.base_dir, "save_ideas.json"), "w") as f:
            json.dump(self.ideas, f, indent=4)

        ## CHECK
        self.semantic_search = SemanticScholar()
        # print(f"IdeaGenerator:on_receive: start semantic_search")
        for idx, idea in enumerate(self.ideas):
            msg = Message()
            msg["idx"] = idx
            msg["total"] = len(self.ideas)
            msg["idea"] = idea

            msg["base_dir"] = self.base_dir
            msg["code"] = self.code
            msg["task_description"] = self.task_description_prompt
            msg["novel_checker"] = True

            checker = NoveltyChecker(self.llm_agent, self.semantic_search)
            # print(f"IdeaGenerator:on_receive: start NoveltyChecker")

            self.send(checker.id, msg)


###################
# run test
###################
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#minillm = MiniLLM(max_token=1024)
llama = Llama3_8B(max_token=1024)
idea_generator = IdeaGenerator(llama)

begin = Message(uidx=1)
begin["base_dir"] = "/home/cxu/ws/llm/LLMPlanner/apps/ai_scientist"
begin["experiment"] = "grokking"

idea_generator.send(idea_generator.id, begin)

System.start()
