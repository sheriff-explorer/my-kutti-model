"""
data/generate_flat_files.py

Generates flat file corpora for:
1. Stage 2: Mid-training (Domain Adaptation) - 2,500 samples
   - data/midtrain_corpus.txt   (plain text, 1 sample per line)
   - data/midtrain_corpus.jsonl (JSON Lines, with metadata)

2. Stage 3: Post-training (Supervised Fine-Tuning / SFT) - 100 Q&A pairs
   - data/posttrain_sft.jsonl   (JSON Lines, instruction / response format)
   - data/posttrain_sft.txt     (Formatted text Q&A pairs)
   - data/posttrain_sft.csv     (CSV table with instruction, response)
"""

import os
import json
import csv
import random

DATA_DIR = os.path.dirname(os.path.abspath(__file__))


def generate_midtrain_data():
    facts = [
        "The sky is blue during the day and orange at sunset.",
        "Water is made of hydrogen and oxygen.",
        "The sun is a star at the center of our solar system.",
        "Dogs are friendly animals that like to play and run.",
        "Cats sleep a lot and like to be comfortable and warm.",
        "Trees grow tall and have leaves that turn colors in autumn.",
        "Birds have wings and can fly high in the sky.",
        "Fish live in water and breathe through their gills.",
        "The moon goes around the Earth and shines at night.",
        "Rain comes from clouds and helps plants grow.",
        "Apples grow on trees and are red, green, or yellow.",
        "Children learn new things every day at school.",
        "The ocean is very deep and full of many creatures.",
        "Snow falls in winter when it is very cold outside.",
        "Flowers bloom in spring when the weather gets warm.",
        "Bread is made from flour, water, and baked in an oven.",
        "Fire is hot and gives light but must be handled carefully.",
        "Books contain stories and information that people enjoy.",
        "A family lives together and takes care of each other.",
        "Friends make life happy and fun by playing together.",
    ]

    items = []
    # 20 factual statements x 100 = 2,000 samples
    for i in range(100):
        for f in facts:
            items.append({
                "id": len(items),
                "domain": "factual",
                "text": f
            })

    # 500 synthetic narrative samples
    story_templates = [
        "Once there was a little {animal} who lived in the {place}. "
        "Every day it would {action}. One day, something special happened.",

        "A young {character} wanted to learn how to {skill}. "
        "At first it was hard, but with practice, they got better and better.",

        "In a small village, there lived a kind {character}. "
        "Everyone loved them because they always helped their neighbors.",

        "The little {animal} found a shiny {object} in the forest. "
        "It did not know what it was, so it asked its friends for help.",

        "One morning, {name} woke up and saw that the sky was very bright. "
        "It was going to be a wonderful day full of adventures.",
    ]

    animals     = ["rabbit", "fox", "bear", "bird", "cat", "dog", "turtle", "owl"]
    places      = ["forest", "meadow", "pond", "garden", "hill", "valley"]
    actions     = ["search for food", "explore the woods", "sing a song", "play with friends"]
    characters  = ["child", "girl", "boy", "farmer", "baker"]
    skills      = ["read", "draw", "cook", "swim", "ride a bicycle"]
    objects     = ["stone", "leaf", "acorn", "flower", "feather"]
    names       = ["Lily", "Tom", "Emma", "Jack", "Sophie", "Ben", "Anna", "Max"]

    random.seed(42)
    for _ in range(500):
        template = random.choice(story_templates)
        story = template.format(
            animal    = random.choice(animals),
            place     = random.choice(places),
            action    = random.choice(actions),
            character = random.choice(characters),
            skill     = random.choice(skills),
            object    = random.choice(objects),
            name      = random.choice(names),
        )
        items.append({
            "id": len(items),
            "domain": "narrative",
            "text": story
        })

    # Save to data/midtrain_corpus.txt (1 sample per line)
    txt_path = os.path.join(DATA_DIR, "midtrain_corpus.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(item["text"] + "\n")

    # Save to data/midtrain_corpus.jsonl
    jsonl_path = os.path.join(DATA_DIR, "midtrain_corpus.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Mid-training flat files created:")
    print(f"  - {txt_path} ({len(items)} samples)")
    print(f"  - {jsonl_path} ({len(items)} records)")


def generate_posttrain_data():
    factual_pairs = [
        ("What color is the sky?",          "The sky is blue during the day."),
        ("What do fish do in water?",        "Fish swim in water and breathe through their gills."),
        ("What do birds use to fly?",        "Birds use their wings to fly through the sky."),
        ("Where does rain come from?",       "Rain comes from clouds in the sky."),
        ("What do trees have on them?",      "Trees have leaves, branches, and roots."),
        ("What is the sun?",                 "The sun is a big star that gives us light and warmth."),
        ("What do dogs like to do?",         "Dogs like to play, run, and be with their owners."),
        ("Where do fish live?",              "Fish live in water, in rivers, lakes, and the ocean."),
        ("What falls from the sky in winter?","Snow falls from the sky when it is very cold."),
        ("What helps plants grow?",          "Sunlight, water, and soil help plants grow."),
        ("What is the moon?",                "The moon is a large rock that goes around the Earth."),
        ("What do cats like?",               "Cats like to sleep, play, and be warm and comfortable."),
        ("What season do flowers bloom?",    "Flowers bloom in spring when it gets warm."),
        ("What is bread made from?",         "Bread is made from flour and water and is baked in an oven."),
        ("What do eyes do?",                 "Eyes help us see the world around us."),
        ("What is water made of?",           "Water is made of tiny parts called hydrogen and oxygen."),
        ("What do ears do?",                 "Ears help us hear sounds and music."),
        ("What makes a rainbow?",            "A rainbow appears when sunlight shines through rain drops."),
        ("Where do apples grow?",            "Apples grow on trees in orchards."),
        ("What is fire?",                    "Fire is hot and bright and gives light and warmth."),
        ("What do bees make?",               "Bees make honey in their hives."),
        ("What is a library?",               "A library is a place with many books that people can read."),
        ("What do children do at school?",   "Children learn to read, write, and discover new things at school."),
        ("What is a nest?",                  "A nest is a home that birds build to lay their eggs."),
        ("What do frogs eat?",               "Frogs eat insects like flies and beetles."),
        ("What is a forest?",                "A forest is a large area filled with many trees."),
        ("What happens in autumn?",          "In autumn, leaves change colors and fall from the trees."),
        ("What does a farmer do?",           "A farmer grows food and takes care of animals."),
        ("What is a river?",                 "A river is a large stream of water that flows to the sea."),
        ("What do rabbits eat?",             "Rabbits eat carrots, grass, and other vegetables."),
        ("What is the ocean?",               "The ocean is a vast body of salt water covering much of Earth."),
        ("What does a baker make?",          "A baker makes bread, cakes, and other delicious food."),
        ("What is wind?",                    "Wind is moving air that we can feel on our face."),
        ("What do butterflies eat?",         "Butterflies drink nectar from flowers."),
        ("What is a mountain?",              "A mountain is a very tall piece of land that rises high."),
        ("Why do we sleep?",                 "We sleep so our bodies and minds can rest and grow stronger."),
        ("What is a bridge?",                "A bridge is a structure that helps people cross over water or roads."),
        ("What do bears eat?",               "Bears eat fish, berries, and honey."),
        ("What is the color of grass?",      "Grass is green and grows in fields and gardens."),
        ("What is a cloud?",                 "A cloud is made of tiny drops of water floating in the sky."),
    ]

    story_pairs = [
        (
            "Begin a story about a rabbit in the forest.",
            "Once there was a little rabbit who lived in a big green forest. "
            "Every morning, it hopped through the tall grass looking for carrots."
        ),
        (
            "Start a story about a child who finds a treasure.",
            "One sunny afternoon, a young girl named Lily was playing in her garden. "
            "She dug a hole and found a shiny box buried deep in the ground."
        ),
        (
            "Begin a story about a kind dragon.",
            "In a land far away, there lived a dragon who was different from all the others. "
            "Instead of breathing fire to scare people, he used it to bake bread for the village."
        ),
        (
            "Start a story about a talking fish.",
            "Deep in a calm blue lake lived a small orange fish who could talk. "
            "Nobody knew about this gift until one day a little boy sat by the water."
        ),
        (
            "Begin a story about a lost puppy.",
            "A tiny brown puppy wandered away from its home one rainy afternoon. "
            "It ran through the wet streets, looking for a friendly face."
        ),
        (
            "Start a story about a brave little bird.",
            "High up in a tall oak tree, a tiny sparrow named Pip wanted to fly. "
            "All the other birds could soar above the clouds, but Pip was afraid."
        ),
        (
            "Begin a story about a wizard's apprentice.",
            "Every morning, a young boy named Finn would sweep the wizard's tower. "
            "One day, he accidentally knocked over a glowing blue bottle."
        ),
        (
            "Start a story about a snowman who comes to life.",
            "On the coldest night of winter, three children built a snowman in the yard. "
            "In the morning, they were amazed to find it had moved closer to the house."
        ),
        (
            "Begin a story about a kind farmer.",
            "Old Farmer Green lived on a small hill with lots of animals. "
            "He fed them every morning and sang songs to make them happy."
        ),
        (
            "Start a story about a child who learns to read.",
            "Maya sat under the big apple tree with her first real book. "
            "The words were hard at first, but she kept trying, one page at a time."
        ),
    ]

    all_pairs = []
    for q, a in factual_pairs:
        all_pairs.append(("factual", q, a))
        all_pairs.append(("factual_augmented", "Tell me: " + q, a))

    for q, a in story_pairs:
        all_pairs.append(("narrative", q, a))
        all_pairs.append(("narrative_augmented", "Tell me: " + q, a))

    # Save to data/posttrain_sft.jsonl
    jsonl_path = os.path.join(DATA_DIR, "posttrain_sft.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for idx, (cat, q, a) in enumerate(all_pairs):
            rec = {
                "id": idx,
                "category": cat,
                "instruction": q,
                "response": a,
                "text": f"Q: {q}\nA: {a}"
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Save to data/posttrain_sft.txt
    txt_path = os.path.join(DATA_DIR, "posttrain_sft.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        for idx, (cat, q, a) in enumerate(all_pairs):
            f.write(f"Q: {q}\nA: {a}\n\n")

    # Save to data/posttrain_sft.csv
    csv_path = os.path.join(DATA_DIR, "posttrain_sft.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "category", "instruction", "response"])
        for idx, (cat, q, a) in enumerate(all_pairs):
            writer.writerow([idx, cat, q, a])

    print(f"Post-training flat files created:")
    print(f"  - {jsonl_path} ({len(all_pairs)} records)")
    print(f"  - {txt_path} ({len(all_pairs)} Q&A pairs)")
    print(f"  - {csv_path} ({len(all_pairs)} rows)")


if __name__ == "__main__":
    generate_midtrain_data()
    print("-" * 50)
    generate_posttrain_data()
