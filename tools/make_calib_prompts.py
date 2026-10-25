#!/usr/bin/env python3
"""Writes a getp input file of generic chat prompts, for calibrating the
engine (for example the expert split, tools/ep_split.py) on data that is not
the evaluation set.

    tools/make_calib_prompts.py 2048 > calib_2048.txt
    tools/make_calib_prompts.py 1024 --kind prose > calib_prose_1024.txt

The prompts are built from the lists below (coding, math, writing, everyday
advice, science and history questions, explanations and essays on
general-knowledge subjects, a few in other languages), so the
same arguments always give the same file. Every prompt is different.
"""

import argparse
import random
import sys

LANGS = ["Python", "C++", "JavaScript", "Rust", "Go", "Java", "C", "TypeScript", "Kotlin", "SQL"]
CODE_TASKS = [
    "reverses a singly linked list", "checks whether a string is a palindrome",
    "merges two sorted arrays into one sorted array", "splits a CSV line into fields, honoring quotes",
    "returns the n-th Fibonacci number using memoization", "counts how often each word appears in a text",
    "finds the shortest path between two cells of a grid with walls", "rotates a square matrix by 90 degrees",
    "implements a small LRU cache", "removes duplicate entries from a list while keeping the order",
    "converts a Roman numeral to an integer", "checks whether parentheses in a string are balanced",
    "computes the greatest common divisor of two numbers", "flattens a nested list of arbitrary depth",
    "returns all permutations of a short string", "parses a date like 2024-03-15 and prints the weekday",
    "implements binary search on a sorted array", "finds the longest common prefix of a list of strings",
    "computes a moving average over a stream of numbers", "serializes a binary tree to a string and back",
    "detects a cycle in a directed graph", "implements a thread-safe counter",
    "converts a temperature table from Celsius to Fahrenheit", "groups a list of people by age",
    "validates a Sudoku board", "finds the k most frequent items in a list",
    "pretty-prints a JSON object with indentation", "simulates a bank account with deposits and withdrawals",
    "reads a file line by line and prints the longest line", "computes the edit distance of two words",
]
CODE_TEMPLATES = [
    "Write a {lang} function that {task}.",
    "Show me how to write a {lang} program that {task}, and explain the time complexity.",
    "I need {lang} code that {task}. Please add a few unit tests.",
    "What is the cleanest way in {lang} to write code that {task}?",
]
CODE_QUESTIONS = [
    "What is the difference between a process and a thread?",
    "Why is my recursive function causing a stack overflow, and how can I avoid it?",
    "Explain how a hash map handles collisions.",
    "When should I use a queue instead of a stack?",
    "How does garbage collection work in a managed language?",
    "What does it mean for an algorithm to be O(n log n)?",
    "How do I undo the last commit in git but keep my changes?",
    "What is a race condition? Give a small example.",
    "Explain the difference between TCP and UDP for a web developer.",
    "What are the pros and cons of microservices compared to a monolith?",
    "How does an index speed up a database query?",
    "What is dependency injection and why do people use it?",
]

NAMES = ["Anna", "Minh", "Carlos", "Priya", "Kenji", "Fatima", "Lukas", "Sofia", "Omar", "Mei", "Jonas", "Amara"]
ITEMS = ["apples", "pencils", "books", "marbles", "stickers", "cookies", "tickets", "stamps"]


def math_prompts(rng, n):
    out = []
    for _ in range(n):
        kind = rng.randrange(8)
        a, b, c = rng.randint(2, 90), rng.randint(2, 40), rng.randint(2, 12)
        name, item = rng.choice(NAMES), rng.choice(ITEMS)
        if kind == 0:
            out.append(f"A train travels {a * 10} km in {c} hours. What is its average speed? Show your steps.")
        elif kind == 1:
            out.append(f"{name} has {a} {item} and gives away {b} of them, then buys {c} boxes of {c} {item} each. "
                       f"How many {item} does {name} have now?")
        elif kind == 2:
            out.append(f"Solve for x: {c}x + {a} = {b * c + a}. Explain each step.")
        elif kind == 3:
            out.append(f"What is {a}% of {b * 25}? Explain how to compute it without a calculator.")
        elif kind == 4:
            out.append(f"A rectangle is {a} cm long and {b} cm wide. What are its area and its perimeter?")
        elif kind == 5:
            out.append(f"If I save {a * 5} dollars every month at {c}% yearly interest, about how much will I have "
                       f"after {b // 4 + 2} years?")
        elif kind == 6:
            out.append(f"Is {a * b + 1} a prime number? Explain how you checked.")
        else:
            out.append(f"Two dice are rolled. What is the probability that the sum is {c}? Explain.")
    return out


STORY_WHO = ["a lighthouse keeper", "a robot learning to paint", "a girl who can talk to birds",
             "an old fisherman", "a detective in a rainy city", "a dragon afraid of fire",
             "two friends lost in a forest", "a baker who bakes one magic loaf a year",
             "an astronaut alone on Mars", "a cat that runs a bookshop", "a young chess player",
             "a time traveler stuck in a Tuesday", "a mail carrier in a mountain village",
             "a shy giant", "a grandmother who becomes a pilot", "a ghost who is afraid of people",
             "a street musician in Paris", "a lost penguin in the desert", "a boy who builds a submarine",
             "twin sisters who swap lives for a week"]
STORY_TEMPLATES = [
    "Write a short story about {who}.",
    "Write a bedtime story for a child about {who}.",
    "Write the opening paragraph of a mystery novel about {who}.",
    "Write a funny dialogue between {who} and a talking teapot.",
    "Write a story in exactly five sentences about {who}.",
    "Describe a single day in the life of {who}.",
]
POEM_SUBJECTS = ["the ocean at night", "autumn leaves", "a city waking up", "an old bicycle", "the first snow",
                 "a cup of coffee", "the moon over the mountains", "a train journey", "a garden after rain",
                 "friendship", "a lost umbrella", "the sound of a violin", "a desert at noon",
                 "an empty playground", "a grandfather's watch", "fireflies in summer", "a crowded market",
                 "the last day of school", "a foggy harbor", "a kite in the wind"]
POEM_TEMPLATES = ["Write a haiku about {s}.", "Write a rhyming poem about {s}.",
                  "Write a free-verse poem about {s}.", "Write a limerick about {s}.",
                  "Write song lyrics about {s}.", "Write a sonnet about {s}."]
EMAILS = [
    "ask my manager for a day off next Friday", "apologize to a customer for a late delivery",
    "invite my team to a farewell lunch", "follow up on a job interview",
    "ask my landlord to fix the heating", "thank a teacher at the end of the school year",
    "remind a client about an unpaid invoice", "decline a meeting invitation politely",
    "ask a professor for a recommendation letter", "announce a new office policy on remote work",
]
EMAIL_TEMPLATES = ["Write a polite email to {e}.", "Draft a short, friendly message to {e}.",
                   "Help me write a formal letter to {e}."]

CITIES = ["Kyoto", "Lisbon", "Hanoi", "Cape Town", "Vancouver", "Istanbul", "Buenos Aires", "Prague",
          "Seoul", "Marrakesh", "Reykjavik", "Singapore"]
TRAVEL_TEMPLATES = ["Plan a {d}-day trip to {c} on a small budget.",
                    "What should a first-time visitor to {c} not miss?",
                    "Suggest a {d}-day itinerary in {c} for a family with young children."]
INGREDIENTS = ["rice, eggs and spinach", "chickpeas, tomatoes and garlic", "chicken, lemon and potatoes",
               "pasta, mushrooms and cream", "tofu, ginger and broccoli", "lentils, carrots and onions",
               "salmon, dill and cucumber", "oats, bananas and peanut butter", "beef, peppers and noodles",
               "beans, corn and avocado"]
FOOD_TEMPLATES = ["Give me a simple recipe using {i}.", "What can I cook for dinner with {i}?",
                  "Suggest a healthy lunch made with {i}, with steps."]
ADVICE = [
    "How can I fall asleep faster?", "How do I start running if I have never exercised?",
    "What are good ways to save money on groceries?", "How do I prepare for a job interview?",
    "How can I stop procrastinating on my studies?", "How do I keep houseplants alive?",
    "What should I consider when buying a used car?", "How do I make a monthly budget?",
    "How can I improve my public speaking?", "How do I learn a new language as an adult?",
    "How can I be more productive when working from home?", "How do I talk to a friend who seems sad?",
    "What is a good morning routine?", "How do I choose a laptop for programming?",
    "How do I remove a coffee stain from a white shirt?", "How do I train a puppy to sit?",
    "How can I reduce stress before exams?", "What are tips for taking better photos with a phone?",
]

SCIENCE = [
    "Why is the sky blue?", "How do vaccines train the immune system?", "What causes the seasons on Earth?",
    "How does a refrigerator keep food cold?", "Why do we have tides?", "How do airplanes stay in the air?",
    "What is the difference between weather and climate?", "How does photosynthesis work?",
    "Why does ice float on water?", "How do black holes form?", "What is DNA and what does it do?",
    "How does a battery store energy?", "Why do cats purr?", "How do earthquakes happen?",
    "What is the greenhouse effect?", "How does the internet send data across the world?",
    "Why do leaves change color in autumn?", "How does a compass work?", "What are antibiotics and why do they stop working?",
    "How do solar panels make electricity?", "What is inflation in economics?", "How does compound interest work?",
    "Why do some materials conduct electricity and others do not?", "How do bees make honey?",
]
SCIENCE_TEMPLATES = ["{q}", "{q} Explain it simply.", "{q} Give a short answer and then a detailed one.",
                     "{q} Use an everyday comparison.", "Answer like a science teacher: {q}"]
HISTORY = [
    "the invention of the printing press", "the building of the Great Wall of China", "the Apollo 11 moon landing",
    "the fall of the Roman Empire", "the Industrial Revolution in Britain", "the Silk Road trade routes",
    "the French Revolution", "the history of the bicycle", "the first transatlantic telegraph cable",
    "the discovery of penicillin", "the Meiji Restoration in Japan", "the construction of the Panama Canal",
    "the invention of the telephone", "the first Olympic Games of the modern era", "the age of sail",
    "the history of coffee", "the Wright brothers' first flight", "the spread of the printing of newspapers",
    "the Hanseatic League", "the building of the first railways",
]
HISTORY_TEMPLATES = ["Tell me about {h}.", "Why was {h} important?", "Summarize {h} in a few paragraphs.",
                     "What were the consequences of {h}?", "Write a timeline of {h}.",
                     "Who were the key people in {h}?"]
MISC = [
    "Translate into French: 'The meeting has been moved to Thursday afternoon.'",
    "Translate into Spanish: 'Could you tell me where the nearest train station is?'",
    "Translate into German: 'I would like to book a table for four people tonight.'",
    "Summarize in one sentence: The city council voted to extend the bike lane network by 20 km over the next "
    "two years, funded by a new parking fee in the downtown area.",
    "Correct the grammar: 'Me and him goes to the store yesterday and buyed three apple.'",
    "Give me five creative names for a coffee shop near a university.",
    "Write a product description for a waterproof hiking backpack.",
    "Explain the rules of chess to a beginner.", "What is the plot of a typical hero's journey story?",
    "Write a short speech for my best friend's wedding.", "Give me ten questions to ask on a first date.",
    "Describe the taste of a mango to someone who has never had one.",
]
CONCEPTS = [
    "how a credit score is calculated", "how a bill becomes a law", "how noise-cancelling headphones work",
    "how a search engine ranks pages", "how a microwave oven heats food", "how GPS finds your position",
    "how a mortgage works", "how the heart pumps blood", "how a rainbow forms", "how yeast makes bread rise",
    "how an elevator knows where to stop", "how a camera lens focuses", "how a vaccine trial is run",
    "how a nuclear reactor makes electricity", "how a touch screen detects a finger",
    "how a 3D printer builds an object", "how a submarine dives and surfaces", "how a QR code stores data",
    "how a hurricane forms", "how an orchestra stays in time", "how a library organizes its books",
    "how bridges carry heavy loads", "how a thermostat controls a room", "how coffee is decaffeinated",
    "how a bank decides on a loan", "how a cell phone network hands off calls", "how glaciers move",
    "how memory works in the human brain", "how an auction sets a price", "how a sailboat moves against the wind",
    "how a jet engine produces thrust", "how a smoke detector senses smoke", "how a heat wave forms",
    "how a cargo ship is loaded", "how wine is made", "how a hospital triage system works",
    "how a power grid balances supply and demand", "how a password manager keeps secrets safe",
    "how a coral reef grows", "how a newspaper decides its front page", "how a violin makes sound",
    "how a city collects rainwater", "how the stock of a supermarket is restocked", "how a fever fights infection",
    "how a telescope gathers light", "how a wind turbine turns wind into power", "how a road is built",
    "how an airport schedules flights", "how a dam controls a river", "how a podcast is produced",
]
CONCEPT_TEMPLATES = ["Explain {c}.", "Can you describe {c} in plain words?", "I always wondered about {c}. Can you explain?",
                     "Write a short article on {c}.", "Teach me {c} with a simple example."]
PAIRS = [
    ("a virus", "a bacterium"), ("stocks", "bonds"), ("weather", "climate"), ("a lake", "a sea"),
    ("RAM", "a hard drive"), ("a crocodile", "an alligator"), ("baking soda", "baking powder"),
    ("a comet", "an asteroid"), ("renting", "buying a home"), ("a credit card", "a debit card"),
    ("speed", "velocity"), ("empathy", "sympathy"), ("an electric car", "a hybrid car"),
    ("a republic", "a monarchy"), ("a symphony", "a concerto"), ("tea", "coffee"),
    ("a laptop", "a tablet"), ("yoga", "pilates"), ("a novel", "a novella"), ("a frog", "a toad"),
]
PAIR_TEMPLATES = ["What is the difference between {a} and {b}?", "Compare {a} and {b} in a short table.",
                  "Which is better for a beginner to understand first, {a} or {b}? Explain."]
DEBATES = [
    "a four-day work week", "banning cars from city centers", "school uniforms", "a universal basic income",
    "homework for young children", "nuclear power", "remote work", "space exploration funding",
    "learning cursive handwriting", "daylight saving time", "self-driving taxis", "a tax on sugary drinks",
]
DEBATE_TEMPLATES = ["Give me the main arguments for and against {d}.", "Write a short opinion piece supporting {d}.",
                    "What would change in a typical town if it adopted {d}?"]
# Explanations and short essays on general-knowledge subjects.
SUBJECTS = [
    "the history of jazz", "how volcanoes shape islands", "the basics of personal budgeting",
    "the rules and strategy of basketball", "the life cycle of stars", "the causes of traffic jams",
    "the role of bees in farming", "how the Roman roads were built", "the science of baking bread",
    "the history of the internet", "how languages change over time", "the basics of music theory",
    "how maps are made", "the physics of roller coasters", "the history of chess",
    "how the human eye sees color", "the ethics of animal testing", "how elections are organized",
    "the history of photography", "how deserts form", "the basics of first aid",
    "the economics of a small restaurant", "the art of the Renaissance", "how glass is made",
    "the history of vaccines", "the psychology of habits", "how a city subway system is planned",
    "the basics of investing in index funds", "the history of the Olympic Games",
    "the role of the moon in human culture", "how sound travels", "the history of writing systems",
    "the basics of nutrition", "how football tactics have changed", "the geography of the Amazon basin",
    "the life of Marie Curie", "the history of tea in Asia", "how earthquakes are measured",
    "the basics of copyright law", "how weather forecasts are made", "the history of comic books",
    "the science of sleep in animals", "how ancient Egyptians built pyramids", "the basics of probability",
    "the history of aviation", "how forests recover after fires", "the basics of public speaking",
    "how online shopping changed retail", "the history of the bicycle in cities", "the chemistry of cooking",
    "the history of clocks and timekeeping", "how rivers carve valleys", "the basics of graphic design",
    "the story of the Silk Road's cities",
]
ESSAY_TEMPLATES = [
    "Write an essay about {s}.", "Give me an overview of {s}.", "Explain {s} to a high school student.",
    "What are the most important things to know about {s}?", "Describe {s} in five paragraphs.",
    "Write a short report on {s}, with an introduction and a conclusion.",
    "What are common myths about {s}, and what is actually true?",
    "Write a blog post about {s} for beginners.", "Summarize the key ideas of {s}.",
    "Explain {s}, and list three questions for further reading.",
    "What are the pros and cons of the usual views on {s}?", "Teach me {s} in ten short points.",
    "Write a lesson plan about {s}.", "Explain {s} and give two real-world examples.",
    "How would you explain {s} to someone from the 1800s?", "Write an encyclopedia-style entry on {s}.",
    "What problems remain unsolved in {s}?", "Write a short speech about {s}.",
    "Create a study guide for {s}.", "Write a dialogue between a teacher and a student about {s}.",
]
OTHER_LANG = [
    "Hãy giải thích vì sao bầu trời có màu xanh.", "Cho tôi một công thức nấu phở bò đơn giản.",
    "Viết một bài thơ ngắn về mùa thu Hà Nội.", "Làm thế nào để học tiếng Anh hiệu quả?",
    "하늘은 왜 파란색인가요?", "간단한 김치찌개 레시피를 알려주세요.", "서울에서 가볼 만한 곳을 추천해 주세요.",
    "¿Cuáles son los beneficios de hacer ejercicio todos los días?", "Escribe un cuento corto sobre un gato curioso.",
    "Explique-moi comment fonctionne un moteur électrique.", "Écris un poème sur la mer.",
    "Wie funktioniert eine Wärmepumpe?", "Schreibe eine kurze Geschichte über einen Roboter.",
    "日本の四季について説明してください。", "简单介绍一下长城的历史。", "如何提高自己的写作能力？",
    "Explique como funciona a fotossíntese.", "Spiegami come si prepara una pizza margherita.",
    "Расскажи, как устроена Солнечная система.", "ช่วยแนะนำวิธีลดความเครียดหน่อย",
]


def pools(rng):
    code = [t.format(lang=l, task=k) for t in CODE_TEMPLATES for l in LANGS for k in CODE_TASKS]
    code += CODE_QUESTIONS
    prose = [t.format(who=w) for t in STORY_TEMPLATES for w in STORY_WHO]
    prose += [t.format(s=x) for t in POEM_TEMPLATES for x in POEM_SUBJECTS]
    prose += [t.format(e=x) for t in EMAIL_TEMPLATES for x in EMAILS]
    prose += [t.format(c=c, d=d) for t in TRAVEL_TEMPLATES for c in CITIES for d in (2, 3, 5)]
    prose += [t.format(i=x) for t in FOOD_TEMPLATES for x in INGREDIENTS]
    prose += [t.format(q=q) for t in ("{q}", "{q} Give me a step-by-step plan.", "{q} Keep it short.")
              for q in ADVICE]
    prose += [t.format(q=q) for t in SCIENCE_TEMPLATES for q in SCIENCE]
    prose += [t.format(h=h) for t in HISTORY_TEMPLATES for h in HISTORY]
    prose += [t.format(a=a, b=b) for t in PAIR_TEMPLATES for a, b in PAIRS]
    prose += [t.format(c=c) for t in CONCEPT_TEMPLATES for c in CONCEPTS]
    prose += [t.format(d=d) for t in DEBATE_TEMPLATES for d in DEBATES]
    prose += MISC
    essay = [t.format(s=x) for t in ESSAY_TEMPLATES for x in SUBJECTS]
    out = dict(code=code, math=math_prompts(rng, 3000), prose=prose, essay=essay, other=OTHER_LANG)
    return {k: list(dict.fromkeys(v)) for k, v in out.items()}  # unique, first occurrence kept


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("count", type=int)
    ap.add_argument("--kind", default="mixed", choices=["mixed", "prose", "essay", "code", "math"],
                    help="mixed (default): every kind except essay, coding prompts only to fill "
                         "up; prose / essay / code / math: that kind only")
    args = ap.parse_args()
    rng = random.Random(20261025)
    p = pools(rng)
    if args.kind == "mixed":
        # Coding prompts would dominate; they only fill up what the others leave.
        rest = p["prose"] + p["math"][:500] + p["other"]
        rng.shuffle(p["code"])
        pool = rest + p["code"][: max(0, args.count - len(rest))]
    else:
        pool = p[args.kind]
    if len(pool) < args.count:
        sys.exit(f"only {len(pool)} different prompts available, asked for {args.count}")
    rng.shuffle(pool)
    print(args.count)
    for line in pool[: args.count]:
        print(line)


if __name__ == "__main__":
    main()
