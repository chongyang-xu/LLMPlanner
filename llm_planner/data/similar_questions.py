from llm_planner.data.dataset import MiscDataset

import numpy as np
from pandas import json_normalize

import json

from llm_planner.planner.queue import Ingress
from llm_planner.query import Query, Stop


class SimilarQuestions(MiscDataset):

    def __init__(self) -> None:
        super().__init__()
        self.questions_dict = {
            "q1": [
                "What time is it?", "Do you know the time?",
                "Could you tell me the time?", "What’s the current time?",
                "Can you give me the time?", "What’s the hour now?",
                "Could you check the time for me?", "Do you have the time?",
                "What’s the clock say?", "What’s the time right now?",
                "How late is it?"
            ],
            "q2": [
                "How are you doing today?", "How’s it going today?",
                "How do you feel today?", "How are you today?",
                "What’s up with you today?", "How are things today?",
                "How are you feeling today?", "How’s your day so far?",
                "What’s going on today?", "How have you been today?",
                "How’s everything today?"
            ],
            "q3": [
                "Where do you live?", "What’s your address?",
                "Where’s your home?", "Where do you reside?",
                "Where’s your place?", "Where are you located?",
                "Where’s your house?", "What’s your living situation?",
                "Where’s your residence?", "What’s your location?",
                "Where do you stay?"
            ],
            "q4": [
                "What is your favorite book?",
                "Which book do you like the most?", "What’s your top book?",
                "Do you have a favorite book?", "Which book is your favorite?",
                "What book do you love the most?",
                "What’s your most-loved book?", "Which book do you prefer?",
                "What book do you enjoy the most?",
                "Which book stands out to you?", "What’s your preferred book?"
            ],
            "q5": [
                "Can you help me with this problem?",
                "Could you assist me with this issue?",
                "Would you help me with this?",
                "Can you give me a hand with this problem?",
                "Do you mind helping me with this?",
                "Could you give me some help here?",
                "Can I get your assistance with this?",
                "Would you mind helping me out?",
                "Could you offer some help with this?",
                "Can you lend me a hand with this issue?",
                "Could you help me figure this out?"
            ],
            "q6": [
                "What did you have for breakfast?",
                "What was your breakfast today?",
                "What did you eat this morning?",
                "Can you tell me your breakfast meal?",
                "What was on your breakfast menu?",
                "What did you consume for breakfast?",
                "What was your morning meal?",
                "What did you have to eat this morning?",
                "What did you enjoy for breakfast today?",
                "What was your first meal today?",
                "What did you start your day with?"
            ],
            "q7": [
                "Do you have any pets?", "Are there any pets in your home?",
                "Do you own any pets?", "Do you keep any animals?",
                "Are you a pet owner?", "Do you have pets at home?",
                "Do you have any furry friends?", "Do you live with any pets?",
                "Are there any animals in your house?",
                "Do you have any domestic animals?", "Do you care for any pets?"
            ],
            "q8": [
                "What are your plans for the weekend?",
                "What are you doing this weekend?",
                "Any plans for the weekend?",
                "How are you spending your weekend?",
                "What’s on your weekend agenda?",
                "Do you have plans this weekend?",
                "What are you up to this weekend?",
                "Any activities planned for the weekend?",
                "How will you spend your weekend?",
                "What’s your weekend schedule like?",
                "What’s your weekend looking like?"
            ],
            "q9": [
                "Have you seen any good movies lately?",
                "Watched any good films recently?",
                "Seen any good movies recently?",
                "Any good films you’ve watched lately?",
                "Watched any interesting movies recently?",
                "Seen any great movies lately?",
                "Any good movies you’ve seen recently?",
                "Watched any good films lately?",
                "Any recent movies you’ve enjoyed?",
                "Seen any good films lately?",
                "Any movies you’ve liked recently?"
            ],
            "q10": [
                "What kind of music do you like?",
                "What’s your favorite genre of music?",
                "What type of music do you enjoy?", "What music do you prefer?",
                "Which music genre do you like?",
                "What’s your favorite kind of music?",
                "What music do you listen to?",
                "What’s your preferred music genre?",
                "Which type of music do you enjoy?",
                "What’s your favorite type of music?",
                "What genre of music do you like?"
            ],
            "q11": [
                "Where are you from?", "What’s your hometown?",
                "Where did you grow up?", "Where were you born?",
                "Where do you hail from?", "What’s your place of origin?",
                "Where’s your native place?", "Which city are you from?",
                "Which country are you from?", "Where are your roots?",
                "What’s your birthplace?"
            ],
            "q12": [
                "What do you do for a living?", "What’s your job?",
                "What’s your occupation?", "What line of work are you in?",
                "What’s your profession?", "What do you do?",
                "Where do you work?", "What’s your career?",
                "What’s your employment?", "What’s your role at work?",
                "What’s your vocation?"
            ],
            "q13": [
                "What are your hobbies?", "What do you do in your free time?",
                "What are your interests?",
                "How do you spend your leisure time?",
                "What activities do you enjoy?",
                "What do you like to do for fun?", "What are your pastimes?",
                "What do you do for recreation?", "What do you do to relax?",
                "What are your favorite hobbies?", "How do you unwind?"
            ]
        }

    def to_ingress(self, n_query=128, shuffle=True):
        qid = 0
        q_list = []
        for _, v in self.questions_dict.items():
            for question in v:
                q = Query(qid=qid, query=question)
                q_list.append(q)

        if shuffle:
            np.random.seed(42)
            np.random.shuffle(q_list)

        ing = Ingress()
        for q in q_list:
            ing.enq(q)

        ing.enq(Stop())

        return ing

    def train(self):
        return super().train()

    def val(self):
        return super().val()

    def test(self):
        return super().test()

    def get_data_collator(self):
        return super().get_data_collator()
