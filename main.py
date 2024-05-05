import jsonlines
from scipy.stats import ttest_ind
from scipy.spatial.distance import jensenshannon
import numpy as np
import random


class Datapoint:

    def __init__(self, prompt_or_gen, datapoint):

        if prompt_or_gen == "prompt":
            self.init_prompt(datapoint)
        elif prompt_or_gen == "gen":
            self.init_gen(datapoint)

    def init_prompt(self, datapoint):

        self.text = datapoint["text"]
        
        self.scores = [datapoint["sexually_explicit"],
                       datapoint["profanity"],
                       datapoint["threat"],
                       datapoint["flirtation"],
                       datapoint["identity_attack"],
                       datapoint["toxicity"],
                       datapoint["insult"],
                       datapoint["severe_toxicity"]]
        
    def init_gen(self, datapoint):

        self.text = " | ".join([g["text"] for g in datapoint])
        
        all_scores = []
        for generation in datapoint:
            maybe_scores = [generation["sexually_explicit"],
                            generation["profanity"],
                            generation["threat"],
                            generation["flirtation"],
                            generation["identity_attack"],
                            generation["toxicity"],
                            generation["insult"],
                            generation["severe_toxicity"]]
            if None not in maybe_scores:
                all_scores.append(maybe_scores)
            
        self.scores = []

        for i in range(8):
            only_scores = [row[i] for row in all_scores]
            avg_score = sum(only_scores) / len(only_scores)
            self.scores.append(avg_score)

    def __repr__(self):

        s = ""
        s += self.text + "\n"
        s += "--------------------------------\n"
        s += str(self.scores) + "\n"
        s += "================================\n"
        return s
        

filename = "/uufs/chpc.utah.edu/common/home/u1419632/data/prompted_gens_gpt3_davinci.jsonl"


def to_distribution(matrix):
    """Given a matrix of (num_samples, num_variables) size,
      return the soft-probability over all variables for that dataset
    """

    return np.sum(matrix, axis=0)

    
def get_js(distA, distB):

    return jensenshannon(distA, distB)

def random_partition(samples, sizes):
    """Randomly partition samples into n different parts, each of size sizes[i]"""

    random.shuffle(samples)

    if sum(sizes) > len(samples):
        raise Exception("random_partition: Partition sizes greater than samples list!")

    partitions = {}
    ptr = 0
    for i, size in enumerate(sizes):
        partitions[i] = samples[ptr : ptr + size]
        ptr += size

    return partitions 


reader = jsonlines.open(filename)


prompts = []
generations = []
for n, obj in enumerate(reader):
    prompts.append(Datapoint("prompt", obj["prompt"]))
    generations.append(Datapoint("gen", obj["generations"]))
    
    if n > 5:
        break

distA = np.array([p.scores for p in prompts])
distB = np.array([p.scores for p in generations])


js1 = get_js(distA, distB)

random_js = []
for i in range(100):
    random_parts = random_partition(prompts + generations, [len(prompts), len(generations)])

    distP = [p.scores for p in random_parts[0]]
    distQ = [p.scores for p in random_parts[1]]

    # print(get_js(distP, distQ))
    random_js.append(get_js(distP, distQ))

js2 = np.average(np.array(random_js))

print(js1)
print(js2)
dist = np.linalg.norm(js1 - js2)
print(dist)
