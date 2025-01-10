import numpy as np
from time import sleep

class Ant:
    def __init__(self, min_window_length, max_window_length, num_starts, signal_length):
        self.min_window_length = min_window_length
        self.max_window_length = max_window_length
        self.num_starts = num_starts
        self.window_length = None
        self.start_idx = None

        # Calculate number of windows dynamically based on min and max window length
        self.num_windows = max_window_length - min_window_length + 1
        self.signal_length = signal_length

    def choose_parameters(self, pheromone, heuristic, alpha, beta, iteration):
        while True: 
            try:
                # Joint probability matrix calcuulation
                joint_probs = (pheromone ** alpha) * (heuristic ** beta)
                joint_probs /= joint_probs.sum()

                # Flatten the 2D matrix into 1D for random choice
                flat_probs = joint_probs.flatten()
                idx = np.random.choice(len(flat_probs), p=flat_probs)

                # Map the index back to the start time and window length
                self.start_idx = idx // self.num_windows

                # Adjust max_window_length dynamically to avoid going out of bounds
                max_valid_window_length = min(self.max_window_length, self.signal_length - self.start_idx)

                if max_valid_window_length < self.min_window_length:
                    # If the start index is too close to the end, raise a ValueError
                    raise ValueError(f"Start index {self.start_idx} too close to the end of signal, unable to fit minimum window length.")

                # Adjust window_length selection to be within the valid range
                valid_window_lengths = np.arange(self.min_window_length, max_valid_window_length + 1)
                window_probs = joint_probs[self.start_idx, :len(valid_window_lengths)]
                window_probs /= window_probs.sum()

                self.window_length = np.random.choice(valid_window_lengths, p=window_probs)

                # Output the chosen start and window length
                print(f"(start, length, end) index = {self.start_idx, self.window_length, self.start_idx + self.window_length}")
                # Convert to ms, 1000 ms/250 Hz = 4
                print(f"(start, length, end) ms = {self.start_idx*4, self.window_length*4, (self.start_idx*4) + (self.window_length*4)}")
                sleep(0.01)
                break

            except ValueError as e:
                # Handle the case where the start index is invalid, continue loop for new selection
                print(e)
                print("Retrying with a new random start index...")

class ACO:
    def __init__(self, num_ants, num_iterations, min_window_length, max_window_length, alpha, beta, evaporation_rate, signal_length, num_starts):
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.min_window_length = min_window_length
        self.max_window_length = max_window_length
        self.num_starts = num_starts
        self.signal_length = signal_length
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate

        # Joint pheromone and heuristic matrices
        self.pheromone = np.ones((num_starts, max_window_length - min_window_length + 1))
        self.heuristic = np.ones((num_starts, max_window_length - min_window_length + 1))  # Uniform heuristic

        self.training_result = {}

    def update_pheromone(self, ants, scores):
        # Evaporate pheromone
        self.pheromone = (1 - self.evaporation_rate) * self.pheromone

        # Update pheromone based on the scores for each ant
        for ant, score in zip(ants, scores):
            if score > max(scores):
                delta_tau = score / max(scores)
            else:
                delta_tau = 0    
            self.pheromone[ant.start_idx, ant.window_length] += delta_tau 

    def optimize(self, evaluate_function):
        def composite_score(scores):
            mean_sc = round(np.mean(np.array(scores)),2)
            sd_sc = round(np.std(np.array(scores)),2)
            comp_score = mean_sc-(sd_sc*0.1)
            return mean_sc, sd_sc, comp_score

        best_score = -np.inf
        best_window_length = None
        best_start_idx = None
        temp_result_iter = {}

        for iteration in range(self.num_iterations):
            ants = [Ant(self.min_window_length, self.max_window_length, 
                    self.num_starts, self.signal_length) for _ in range(self.num_ants)]
            scores = []
            temp_result_ant = {}
            for i, ant in enumerate(ants):
                print(f"ant {i}")
                # Ant chooses its parameters based on joint pheromone and heuristic
                ant.choose_parameters(self.pheromone, self.heuristic, self.alpha, self.beta, iteration)
                train_scores, test_acc = evaluate_function(ant.window_length, ant.start_idx)
                
                ### calculate composite score ####
                mean_sc, sd_sc, score = composite_score(train_scores)
                temp_result_ant[f"ant_{i+1}"] = {"window_params":{
                                                "start_idx":int(ant.start_idx),
                                                "length_idx":int(ant.window_length),
                                                "end_idx":int(ant.start_idx+ant.window_length)},
                                                "accuracy":[float(i) for i in train_scores],
                                                "mean":float(mean_sc),
                                                "sd":float(sd_sc),
                                                "composite_score":float(score)
                                                }
                scores.append(score)

                if score > best_score:
                    best_score = score
                    best_window_length = ant.window_length
                    best_start_idx = ant.start_idx

            temp_result_ant["best_solution"] = {"best_score":float(np.round(best_score,2)),
                                                 "best_start_idx":int(best_start_idx),
                                                 "best_window_length":int(best_window_length),
                                                 "best_end_idx":int(best_start_idx+best_window_length)
                                                }
            temp_result_iter[f"iter_{iteration+1}"] = temp_result_ant

            self.update_pheromone(ants, scores)
            # self.update_heuristic(ants, scores)

            print(f'Iteration {iteration + 1}: Best score = {best_score}')

        return best_window_length, best_start_idx, best_score, temp_result_iter