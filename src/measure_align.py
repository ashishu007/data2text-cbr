"""
Use alignment for feature weighting
"""
import json
import pickle
import numpy as np
import pyswarms as ps
import pandas as pd
from sklearn.metrics import ndcg_score
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
from sentence_transformers import SentenceTransformer

roberta_model = SentenceTransformer('paraphrase-distilroberta-base-v1')

class CaseAlignMeasure:
    def __init__(self):
        self.top_k = 5
        self.num_player_ftrs = 29
        self.num_team_ftrs = 23
        self.problem_side_csv = pd.read_csv(f'./data/case_base/player_stats_problem.csv')
        self.solution_side_csv = pd.read_csv(f'./data/case_base/player_stats_solution.csv')
        self.case_base = {
            "problem_side": [list(json.loads(i).values()) for i in list(self.problem_side_csv['sim_features'])],
            "solution_side": list(self.solution_side_csv['templates'])
        }
        self.scaler_model = pickle.load(open('./data/imp_players/imp_player_data_scaler.pkl', 'rb'))
        self.embedded_solutions = roberta_model.encode(self.case_base['solution_side'])

    def get_vector_similarity(self, vec1, vec2):
        v1 = np.reshape(vec1, (1, -1))
        v2 = np.reshape(vec2, (1, -1))
        return cosine_similarity(v1, v2)[0][0]

    def get_align_score(self, problem_list, solution_list):
        """
        input:
            problem/solution lists
        return:
            align score
        """
        prob_rank = np.copy(problem_list)
        sol_rank = np.copy(solution_list)

        prob_rank[self.top_k:] = 1
        for i in range(self.top_k):
            prob_rank[i] = (self.top_k + 1) - i

        sol_rank[:] = 1
        for i in range(self.top_k):
            sol_ind = solution_list.index(problem_list[i])
            sol_rank[sol_ind] = (self.top_k + 1) - i

        ndcg = ndcg_score([prob_rank], [sol_rank])    
        return ndcg

def problem_lists_function(p):
    """ Calculate roll-back the weights and biases

    Inputs
    ------
    p: np.ndarray
        The dimensions should include an unrolled version of the
        weights and biases.

    Returns
    -------

    This should give the list of problem side lists for all examples in the train set
    """
    
    problem_side_cb = cam_obj.scaler_model.transform(np.array(cam_obj.case_base['problem_side'])).tolist()
    # print(len(problem_side_cb))

    problem_lists, solutions = [], []
    for idx, case in enumerate(problem_side_cb):
        target_problem_arr = np.multiply(np.array(case), p)
        case_base_arr = np.multiply(np.array([i for idx1, i in enumerate(problem_side_cb) if idx1 != idx]), p)
        dists = euclidean_distances(case_base_arr, [target_problem_arr])
        dists_1d = dists.ravel()
        dists_arg = np.argsort(dists_1d)
        problem_lists.append(dists_arg)
        solutions.append(cam_obj.embedded_solutions[dists_arg[1]])

    # print("len(problem_lists), len(problem_lists[0])", len(problem_lists), len(problem_lists[0]))

    return problem_lists, solutions

# Forward propagation
def forward_prop(params):
    """Forward propagation as objective function

    This computes for the forward propagation of the neural network, as
    well as the loss.

    Inputs
    ------
    params: np.ndarray
        The dimensions should include an unrolled version of the
        weights and biases.

    Returns
    -------
    float
        The computed case-alignment given the parameters

    1. get the problem_side lists and their corresponding best solution index for all the cases from previous function 
    2. get the solution_side lists for all the cases
    3. calculate the align score for whole case-base - this should be our loss function
    """

    problem_lists, generated_solutions = problem_lists_function(params)
    solution_side_cb = cam_obj.embedded_solutions

    solution_lists = []
    for idx, case in enumerate(solution_side_cb):
        target_solution_arr = generated_solutions[idx]
        case_base_solution_arr = np.array([i for idx1, i in enumerate(solution_side_cb) if idx1 != idx])
        dists = euclidean_distances(case_base_solution_arr, [target_solution_arr])
        dists_1d = dists.ravel()
        dists_arg = np.argsort(dists_1d)
        solution_lists.append(dists_arg)

    # print("len(solution_lists), len(solution_lists[0])", len(solution_lists), len(solution_lists[0]))

    # print("len(problem_lists), len(solution_lists)", len(problem_lists), len(solution_lists))

    all_align = []
    for pl, sl in zip(problem_lists, solution_lists):
        all_align.append(cam_obj.get_align_score(pl.tolist(), sl.tolist()))

    loss = 1 - np.mean(np.array(all_align))
    # print("loss", loss)

    return loss

def f(x):
    """Higher-level method to do forward_prop in the
    whole swarm.

    Inputs
    ------
    x: numpy.ndarray of shape (n_particles, dimensions)
        The swarm that will perform the search

    Returns
    -------
    numpy.ndarray of shape (n_particles, )
        The computed loss for each particle
    """
    n_particles = x.shape[0]
    j = [forward_prop(x[i]) for i in range(n_particles)]
    return np.array(j)

print("Constructing main...")
cam_obj = CaseAlignMeasure()
print("Constructed!!\n\n")
# params = np.random.rand((29))
# # problem_lists_function(params)
# print(forward_prop(params))


# Initialize swarm
options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}

# Call instance of PSO
dimensions = cam_obj.num_player_ftrs
optimizer = ps.single.GlobalBestPSO(n_particles=100, dimensions=dimensions, options=options)

# Perform optimization
cost, pos = optimizer.optimize(f, iters=10)

print(pos)

