#!/usr/bin/python
# encoding: utf-8

import Queue
import multiprocessing
import random
import time

from RappCloud import RappPlatformAPI
from automated_denoising_and_evaluation import wdwfs_and_objective_metrics, wdwfs_and_speech_recognition
from dataset_configuration import dataset_configuration_for_objective_metrics, dataset_configuration_for_speech_recognition

# Global Variables -------------------------------------------------------------------------------------------------------------------
sampling_rate = 16000 # A different value is going to give a result but it will most probably be wrong.
# ------------------------------------------------------------------------------------------------------------------------------------

def simple_dataset_reformation(simple_dataset):

	"""A utility function called from optimization_algorithm() when the evaluation procedure is based on a speech recognition system.

Reforms the structure of the simple_dataset in a way effective for usage by the optimization procedure.

Inputs:
  simple_dataset: A list with so many nested lists as the number of languages of the dataset. Each nested list contains so many nested
  lists as the number of language-specific words of the dataset. Each nested nested list contains so many nested lists as the number
  of noise levels of the dataset. Each nested nested nested list contains so many strings as the number of wav files that correspond
  to a specific language, word and noise level. Each string represents the name of a wav file.

Outputs:
  reformed_simple_dataset: A list with so many strings as the total number of wav files of the dataset.
  Each string represents the name of the corresponding wav file."""

	reformed_simple_dataset = []
	number_of_languages = len(simple_dataset)

	for language_id in range(number_of_languages):
		number_of_words = len(simple_dataset[language_id])

		for word_id in range(number_of_words):
			number_of_noise_levels = len(simple_dataset[language_id][word_id])

			for nl_id in range(number_of_noise_levels):
				number_of_wav_files = len(simple_dataset[language_id][word_id][nl_id])

				for wav_id in range(number_of_wav_files):
					reformed_simple_dataset.append(simple_dataset[language_id][word_id][nl_id][wav_id])

	return reformed_simple_dataset

def training_dataset_formation(evaluation_id, simple_dataset, reformed_simple_dataset, optimization_dataset, transcription_list):

	"""Randomly forms the dataset that will be used in the optimization procedure.

If the optimization procedure is based on an objective metric, only 1 sentence is selected to take part in the optimization
procedure. The selection is made randomly with equal chances for all the sentences.

If the optimization procedure is based on a speech recognition system, words from all the available languages are selected. The number
of selected words is the same for all the available languages. Namely, if there are 2 languages and 3 words are going to be selected
from the first language, 3 words are going to be selected from the second language as well. The words are selected randomly with equal
chances for every word. Three words per language are selected. Eventually, noisy wav files, corresponding to each selected word, are
randomly selected. Specifically, 1 noisy wav file is selected for every available noise level from the word-specific recordings.

Inputs:
  evaluation_id: If the optimization procedure is based on a speech recognition system, evaluation_id must be 7.
  Otherwise, evaluation_id may be any integer.

  simple_dataset:
    If evaluation_id = 7, it must be the first output of the dataset_configuration.dataset_configuration_for_speech_recognition().
    Otherwise, it must be the first output of the dataset_configuration.dataset_configuration_for_objective_metrics().

  reformed_simple_dataset:
    If evaluation_id = 7, it must be the output of the simple_dataset_reformation(). Otherwise, it is redundant.

  optimization_dataset:
    If evaluation_id = 7, it must be the second output of the dataset_configuration.dataset_configuration_for_speech_recognition().
    Otherwise, it must be the second output of the dataset_configuration.dataset_configuration_for_objective_metrics().

  transcription_list:
    If evaluation_id = 7, it must be the third output of the dataset_configuration.dataset_configuration_for_speech_recognition().
    Otherwise, it is redundant.

Outputs:
  recognition_content:
    If evaluation_id = 7, it is a list with so many strings as the number of wav files that will participate in the optimization
    procedure. Each string represents the word that is recorded in the corresponding wav file.

    Otherwise, it is a list that contains a string which represents the clean wav file of the sentence that will participate in the
    optimization procedure.

  training_dataset:
    If evaluation_id = 7, it is a list with so many nested 2-D numpy arrays as the number of wav files that will participate in the
    optimization procedure.

    Otherwise, it is a list with 1 nested list. The nested list contains so many nested 1-D numpy arrays as the number of wav files
    that correspond to the sentence that was selected to participate in the optimization procedure. The first wav file represents the
    clean version of the sentence."""

	random.seed()
	recognition_content = []

	if evaluation_id != 7: # Speech intelligibility metrics.
		sentence_id = random.randint(0, len(optimization_dataset) - 1)
		training_dataset = [optimization_dataset[sentence_id]]
		recognition_content.append(simple_dataset[sentence_id][0])

	else: # Speech recognition system.
		number_of_selected_words_per_language = 3
		number_of_languages = len(optimization_dataset)
		training_dataset = []

		for language_id in range(number_of_languages):
			number_of_words = len(optimization_dataset[language_id])
			selected_words_ids, words_counter = [], 0

			while words_counter < number_of_selected_words_per_language:
				word_id = random.randint(0, number_of_words - 1)

				if selected_words_ids.count(word_id) == 0:
					selected_words_ids.append(word_id)
					words_counter = words_counter + 1
					number_of_noise_levels = len(optimization_dataset[language_id][word_id])

					for nl_id in range(number_of_noise_levels):
						number_of_wav_files = len(optimization_dataset[language_id][word_id][nl_id])
						wav_id = random.randint(0, number_of_wav_files - 1)
						wav_name = simple_dataset[language_id][word_id][nl_id][wav_id]
						transcription_id = reformed_simple_dataset.index(wav_name)

						training_dataset.append(optimization_dataset[language_id][word_id][nl_id][wav_id])
						recognition_content.append(transcription_list[transcription_id])

				else:
					random.seed()

	return [recognition_content, training_dataset]

def optimization_parameters_space(noise_estimation_process_id):

	"""Defines the ranges and the resolutions of the optimization parameters and initializes them as well.

Inputs:
  noise_estimation_process_id: 1 if the noise_footprint_creation.best_esnr_channel_selection() will be used.
  Different from 1 if the noise_footprint_creation.esnr_and_common_zero_areas() will be used. More in optimization_algorithm().

The order of the optimization parameters is specific: [db_cutoff, a, b, c, d, dps]. More in optimization_algorithm().
Even if not all of the optimization parameters are going to participate in the optimization procedure, for instance d and dps are
redundant if the WDWF_0 will be used, the length of the optimization parameters vector is remain the same and this function operates
normally, without taking extra care.

Outputs:
  low_boundaries: A list which contains the low boundaries of the ranges of the optimization parameters.
  high_boundaries: A list which contains the high boundaries of the ranges of the optimization parameters.
  step_sizes: A list which contains the resolutions of the optimization parameters.
  initial_parameters: A list which contains the initial values of the optimization parameters."""

	db_cutoff_high = 0.0
	a_low, a_high, initial_a = 0.5, 4.0, 2.0
	b_low, b_high, initial_b = 0.5, 4.0, 2.0
	c_low, c_high, initial_c = 0.01, 2.01, 0.1

	if noise_estimation_process_id == 1: # best_esnr_channel_selection().
		db_cutoff_low, initial_db_cutoff = -10.0, -7.0

	else: # esnr_and_common_zero_areas().
		db_cutoff_low, initial_db_cutoff = -5.0, -4.0

	d_low, d_high, initial_d = 0.05, 0.4, 0.22 # Parameter d depends on the global variable sampling_rate.
	dps_low, dps_high, initial_dps = 0.6, 1.0, 0.9

	low_boundaries = [db_cutoff_low, a_low, b_low, c_low, d_low, dps_low]
	high_boundaries = [db_cutoff_high, a_high, b_high, c_high, d_high, dps_high]
	step_sizes = [(high_boundaries[p] - low_boundaries[p]) / 10 for p in range(len(low_boundaries))]
	initial_parameters = [initial_db_cutoff, initial_a, initial_b, initial_c, initial_d, initial_dps]

	return [low_boundaries, high_boundaries, step_sizes, initial_parameters]

def before_initialization(parameters_vector, parameters, metrics_vector, metric_value, local_maximums_states, state):

	"""A utility function called from optimization_algorithm() when the time for initialization comes.

Inputs:
  parameters_vector: A list with so many nested lists as the number of iterations of the optimization algorithm.
  Each nested list contains the optimization parameters that were used at the corresponding iteration.
  The order of the optimization parameters is specific and defined in the optimization_parameters_space().

  parameters: A list that contains the last found optimization parameters.

  metrics_vector: A list with so many elements as the number of iterations of the optimization algorithm.
  Each element represents the value of the evaluation metric that was computed at the corresponding iteration.

  metric_value: The last computed value of the evaluation metric.

  local_maximums_states: A list that contains so many tuples as the number of local maximums found by the optimization algorithm.
  Each tuple contains a list and a value. The list represents the optimization parameters that led to the corresponding local maximum.
  The value represents the value of the evaluation metric at the corresponding local maximum.

  state: A tuple that contains a list and a value. The list represents the last found optimization parameters that led to a local
  maximum. The value represents the value of the evaluation metric at this local maximum.

Outputs:
  True: Whenever the current function is called, it returns True.
  parameters_vector: Indirectly, it is an output.
  metrics_vector: Indirectly, it is an output.
  local_maximums_states: Indirectly, it is an output."""

	parameters_vector.append(parameters[:])
	metrics_vector.append(metric_value)
	local_maximums_states.append(state)

	return True

def initial_parameters_random_generator(parameters_vector, metrics_vector, low_boundaries, high_boundaries, step_sizes):

	"""Random calculation of the initialized parameters for the Random Restart and Stochastic Hill Climbing (RRSHC) algorithm.

There is 25% probability the initialized optimization parameters to be computed in an area near the one specified by the best
parameters found from the RRSHC.

Inputs:
  parameters_vector: A list with so many nested lists as the number of iterations of the RRSHC.
  Each nested list contains the optimization parameters that were used at the corresponding iteration.
  The order of the optimization parameters is specific and defined in the optimization_parameters_space().

  metrics_vector: A list with so many elements as the number of iterations of the RRSHC.
  Each element represents the value of the evaluation metric that was computed at the corresponding iteration.

  low_boundaries: A list which contains the low boundaries of the ranges of the optimization parameters.
  high_boundaries: A list which contains the high boundaries of the ranges of the optimization parameters.
  step_sizes: A list which contains the resolutions of the optimization parameters.

Outputs:
  initial_parameters: A list that contains the initialized parameters."""

	random.seed()
	initial_parameters = []
	probability_population = [0] * 75 + [1] * 25 # 25% probability to choose 1.
	prob = random.sample(probability_population, 1)[0]

	if prob == 1:
		best_parameters_id = metrics_vector.index(max(metrics_vector))
		best_parameters = parameters_vector[best_parameters_id]

		for par_id in range(len(best_parameters)):
			current_parameter = best_parameters[par_id]
			area_low_bound = current_parameter - step_sizes[par_id]
			area_high_bound = current_parameter + step_sizes[par_id]

			if area_low_bound < low_boundaries[par_id]:
				area_low_bound = low_boundaries[par_id]
				area_high_bound = current_parameter + (1.5 * step_sizes[par_id])

			elif area_high_bound > high_boundaries[par_id]:
				area_high_bound = high_boundaries[par_id]
				area_low_bound = current_parameter - (1.5 * step_sizes[par_id])

			param_value = round(random.uniform(area_low_bound, area_high_bound), 2)
			initial_parameters.append(param_value)

	else:
		for par_id in range(len(parameters_vector[0])):
			param_value = round(random.uniform(low_boundaries[par_id], high_boundaries[par_id]), 2)
			initial_parameters.append(param_value)

	return initial_parameters

def probability_weights_computation(probability_weights, metric_id, weight_id, change_constant, number_of_active_weights):

	"""Calculates the probability of every optimization parameter to be selected for the next computational step of the
Random Restart and Stochastic Hill Climbing (RRSHC) algorithm.

At each iteration of the RRSHC, it is possible to take place more than 1 computational steps and this depends on the value of the
number_of_attempts_before_initialization parameter of the optimization_algorithm().

The order of the optimization parameters is specific and defined in the optimization_parameters_space().

Inputs:
  probability_weights: A list that contains the selection probabilities of the optimization parameters.
  Each valid probability is between 0.001 and 1.0 and its resolution is 0.001. It may contains invalid probabilities as well, which
  have a 0.0 value and thus, the corresponding optimization parameters cannot be selected.

  metric_id: Determines if an objective metric or a speech recognition system is being used. More in optimization_algorithm().
  If metric_id = 7, a speech recognition is being used. Otherwise, an objective metric is being used.

  weight_id: Determines the optimization parameter which was selected in the previous computational step of the RRSHC.
  change_constant: Determines the change that took place in the evaluation metric as a result of the selection of the weight_id.

  number_of_active_weights: Determines the number of active optimization parameters.
  The length of the probability_weights may be bigger than the number of active optimization parameters for practical reasons.
  In this case, the first number_of_active_weights elements of the probability_weights have valid selection probability values,
  whereas the other elements have 0.0 selection probability value, which is invalid.

Outputs:
  If weight_id < 0, initialized_probability_weights: A list that contains the initial selection probabilities of the optimization
  parameters.

  If weight_id >= 0, None. In this case, indirectly, the output is the probability_weights parameter."""

	if weight_id < 0: # Initialization mode.
		if metric_id == 7: # Speech recognition system.
			if number_of_active_weights == 4:
				initialized_probability_weights = [0.2, 0.25, 0.25, 0.3, 0.0, 0.0] # [w_db_cutoff, w_a, w_b, w_c, w_d, w_dps].

			elif number_of_active_weights == 5:
				initialized_probability_weights = [0.2, 0.2, 0.2, 0.25, 0.15, 0.0] # [w_db_cutoff, w_a, w_b, w_c, w_d, w_dps].

			else:
				initialized_probability_weights = [0.15, 0.2, 0.2, 0.25, 0.15, 0.05] # [w_db_cutoff, w_a, w_b, w_c, w_d, w_dps].

		else: # Speech intelligibility metrics.
			if number_of_active_weights == 4:
				initialized_probability_weights = [0.35, 0.2, 0.15, 0.3, 0.0, 0.0] # [w_db_cutoff, w_a, w_b, w_c, w_d, w_dps].

			elif number_of_active_weights == 5:
				initialized_probability_weights = [0.3, 0.2, 0.1, 0.25, 0.15, 0.0] # [w_db_cutoff, w_a, w_b, w_c, w_d, w_dps].

			else:
				initialized_probability_weights = [0.28, 0.18, 0.12, 0.23, 0.12, 0.07] # [w_db_cutoff, w_a, w_b, w_c, w_d, w_dps].

		return initialized_probability_weights

	else:
		probability_weights[weight_id] = round(probability_weights[weight_id] + change_constant, 3) # change_constant can be negative.
		uniform_modification = change_constant / (number_of_active_weights - 1)

		if probability_weights[weight_id] < 0.001:
			probability_weights[weight_id] = 0.001 # This is the resolution of the probability weights.

		elif probability_weights[weight_id] > 1.0:
			probability_weights[weight_id] = 1.0

		for w_id in range(len(probability_weights)):
			if w_id == weight_id:
				continue

			elif w_id >= number_of_active_weights:
				break

			else:
				probability_weights[w_id] = round(probability_weights[w_id] - uniform_modification, 3)

				if probability_weights[w_id] < 0.001:
					probability_weights[w_id] = 0.001 # This is the resolution of the probability weights.

				if probability_weights[w_id] > 1.0:
					probability_weights[w_id] = 1.0

	# In the end, it is possible that the sum of the probability weights will be a bit over than 1.0. It's ok.
	return

def probability_weights_redistribution(probability_weights, excluded_parameters, number_of_active_weights):

	"""Redistributes the selection probabilities of the optimization parameters to make sure that some of them will not be selected
for the next computational step of the Random Restart and Stochastic Hill Climbing (RRSHC) algorithm.

At each iteration of the RRSHC, it is possible to take place more than 1 computational steps and this depends on the value of the
number_of_attempts_before_initialization parameter of the optimization_algorithm().

The order of the optimization parameters is specific and defined in the optimization_parameters_space().

Inputs:
  probability_weights: A list that contains the selection probabilities of the optimization parameters.
  Each valid probability is between 0.001 and 1.0 and its resolution is 0.001. It may contains invalid probabilities as well, which
  have a 0.0 value and thus, the corresponding optimization parameters cannot be selected.

  excluded_parameters: A list that contains the ids of the optimization parameters that normally have valid selection probabilities
  but they cannot be selected for the next computational step of the RRSHC.

  number_of_active_weights: Determines the number of active optimization parameters.
  The length of the probability_weights may be bigger than the number of active optimization parameters for practical reasons.
  In this case, the first number_of_active_weights elements of the probability_weights have valid selection probability values,
  whereas the other elements have 0.0 selection probability value, which is invalid.

Outputs:
  new_weights: A list that contains the selection probabilities of the optimization parameters after the exclusion of the
  excluded_parameters. The exclusion is taking place by equalizing the selection probabilities of the excluded parameters with 0.0.
  Thus, len(new_weights) = len(probability_weights)."""

	new_weights = []
	flag = False

	for w_id in range(len(probability_weights)):
		for exclud_param in excluded_parameters:
			if w_id == exclud_param:
				flag = True
				break

		if flag:
			new_weights.append(0.0)
			flag = False

		else:
			new_weights.append(probability_weights[w_id])

	sum_difference = sum(probability_weights) - sum(new_weights)
	uniform_modification = sum_difference / (number_of_active_weights - len(excluded_parameters)) # Always positive.

	for w_id in range(len(new_weights)):
		if new_weights[w_id] > 0.0:
			new_weights[w_id] = round(new_weights[w_id] + uniform_modification, 3)

	return new_weights

def plus_or_minus_weights_computation(plus_or_minus_weights, weight_id, sign_id, change_constant, number_of_active_weights):

	"""Computes the increment (sign: +) and decrement (sign: -) probabilities of the optimization parameters.

It is assumed that before the previous computational step of the Random Restart and Stochastic Hill Climbing (RRSHC) algorithm, an
optimization parameter was selected to be changed either by increasing or by decreasing its value. Afterwards, the computational
step took place. According to the result of the computational step, the increment and decrement probabilities of the selected
optimization parameter change.

At each iteration of the RRSHC, it is possible to take place more than 1 computational steps and this depends on the value of the
number_of_attempts_before_initialization parameter of the optimization_algorithm().

The order of the optimization parameters is specific and defined in the optimization_parameters_space().

Inputs:
  plus_or_minus_weights: A list that contains 6 nested lists because the maximum number of optimization parameters that can take part
  in the optimization procedure is 6. Each nested list contains 2 elements. The first represents the decrement probability of the
  corresponding parameter while the second its increment probability. Each valid probability is between 0.01 and 1.0 and its
  resolution is 0.01. It is possible to exist invalid probabilities as well, which have 0.0 value.

  weight_id: Determines the optimization parameter which was selected in the previous computational step of the RRSHC.
  If weight_id < 0, the plus_or_minus_weights takes the initial increment and decrement probabilities of the optimization parameters.

  sign_id: 0 if the value of weight_id optimization parameter was decreased. Different from 0 otherwise.

  change_constant: Determines the change that took place in the evaluation metric as a result of the selection of the weight_id
  optimization parameter and the increment or decrement of its value.

  number_of_active_weights: Determines the number of active optimization parameters.
  The length of the plus_or_minus_weights may be bigger than the number of active optimization parameters for practical reasons.
  In this case, the first number_of_active_weights lists of the plus_or_minus_weights have valid probability values, whereas the other
  lists have 0.0 probability value for each of their elements, which is an invalid value.

Outputs:
  If weight_id < 0, initialized_plus_or_minus_weights: A list with 6 nested lists.
  Each nested list contains the decrement and increment probabilities of the corresponding optimization parameter.

  If weight_id >= 0, None. In this case, indirectly, the output is the plus_or_minus_weights parameter."""

	if weight_id < 0: # Initialization mode.
		if number_of_active_weights == 4:
			initialized_plus_or_minus_weights = [[0.5, 0.5]] * 4 + [[0.0, 0.0]] * 2

		elif number_of_active_weights == 5:
			initialized_plus_or_minus_weights = [[0.5, 0.5]] * 5 + [[0.0, 0.0]]

		else:
			initialized_plus_or_minus_weights = [[0.5, 0.5]] * 6

		return initialized_plus_or_minus_weights

	else:
		minus_probability = plus_or_minus_weights[weight_id][0]
		plus_probability = plus_or_minus_weights[weight_id][1]

		if sign_id == 0:
			minus_probability = round(minus_probability + change_constant, 2) # change_constant can be negative.

			if minus_probability < 0.01:
				minus_probability = 0.01 # This is the resolution of the plus or minus weights.

			elif minus_probability > 1.0:
				minus_probability = 1.0

			plus_probability = round(1.0 - minus_probability, 2)

		else:
			plus_probability = round(plus_probability + change_constant, 2) # change_constant can be negative.

			if plus_probability < 0.01:
				plus_probability = 0.01 # This is the resolution of the plus or minus weights.

			elif plus_probability > 1.0:
				plus_probability = 1.0

			minus_probability = round(1.0 - plus_probability, 2)

		plus_or_minus_weights[weight_id] = [minus_probability, plus_probability]

	return

def creation_of_population(weights, resolution_condition):

	"""A utility function that creates the population needed for the sample() method of the random.py module.

Inputs:
  weights: A list that contains the probabilities needed for the creation of the population. It can be the probability_weights
  (see probability_weights_computation()) or a nested list of the plus_or_minus_weights (see plus_or_minus_weights_computation()).

  resolution_condition: If weights = probability_weights, then resolution_condition must be True. Otherwise, it must be False.

Outputs:
  population (described with an example):
    If weights = probability_weights and probability_weights = [0.28, 0.17, 0.17, 0.17, 0.13, 0.08], then
    population = [0] * 280 + [1] * 170 + [2] * 170 + [3] * 170 + [4] * 130 + [5] * 80.

    If weights = plus_or_minus_weights[parameter_id] and plus_or_minus_weights[parameter_id] = [0.7, 0.3], then
    population = [0] * 70 + [1] * 30, where parameter_id = index of some optimization parameter."""

	if resolution_condition:
		resolution = 1000

	else:
		resolution = 100

	population = []
	population_parameters = [int(weights[w] * resolution) for w in range(len(weights))]

	for w in range(len(weights)):
		for p in range(population_parameters[w]):
			population.append(w)

	return population

def parameters_selection_and_change(parameters, probability_weights, plus_or_minus_weights, \
									low_boundaries, high_boundaries, step_sizes):

	"""Randomly selects which optimization parameter will be modified for the next computational step of the
Random Restart and Stochastic Hill Climbing (RRSHC) algorithm and increases or decreases its value.

At each iteration of the RRSHC, it is possible to take place more than 1 computational steps and this depends on the value of the
number_of_attempts_before_initialization parameter of the optimization_algorithm().

Inputs:
  parameters: A list that contains the values of the optimization parameters.
  The order of the optimization parameters is specific and defined in the optimization_parameters_space().

  probability_weights: A list that contains the selection probabilities of the optimization parameters.
  Each valid probability is between 0.001 and 1.0 and its resolution is 0.001. It may contains invalid probabilities as well, which
  have a 0.0 value and thus, the corresponding optimization parameters cannot be selected.

  plus_or_minus_weights: A list that contains 6 nested lists because the maximum number of optimization parameters that can take part
  in the optimization procedure is 6. Each nested list contains 2 elements. The first represents the decrement probability of the
  corresponding parameter while the second its increment probability. Each valid probability is between 0.01 and 1.0 and its
  resolution is 0.01. It is possible to exist invalid probabilities as well, which have 0.0 value.

  low_boundaries: A list which contains the low boundaries of the ranges of the optimization parameters.
  high_boundaries: A list which contains the high boundaries of the ranges of the optimization parameters.
  step_sizes: A list which contains the resolutions of the optimization parameters.

Outputs:
  parameter_id: The index of the parameters list that corresponds to the selected optimization parameter.
  plus_or_minus: If 1, then the value of the selected optimization parameter has been increased. If 0, it has been decreased.
  parameters: Indirectly, it is an output."""

	# Selection ----------------------------------------------------------------------------------------------------------------------
	random.seed()
	parameters_id_population = creation_of_population(probability_weights, True)
	parameter_id = random.sample(parameters_id_population, 1)[0]
	plus_or_minus_population = creation_of_population(plus_or_minus_weights[parameter_id], False)
	plus_or_minus = random.sample(plus_or_minus_population, 1)[0]
	# --------------------------------------------------------------------------------------------------------------------------------

	# Change -------------------------------------------------------------------------------------------------------------------------
	if plus_or_minus == 1: # Increment of the selected optimization parameter.
		temp_parameter = parameters[parameter_id] + step_sizes[parameter_id]

		if temp_parameter <= high_boundaries[parameter_id]:
			parameters[parameter_id] = round(temp_parameter, 2)

		else: # Decrement of the optimization parameter because its increment led to out of bounds.
			parameters[parameter_id] = round(parameters[parameter_id] - step_sizes[parameter_id], 2)
			plus_or_minus = 0

	else: # Decrement of the selected optimization parameter.
		temp_parameter = parameters[parameter_id] - step_sizes[parameter_id]

		if temp_parameter >= low_boundaries[parameter_id]:
			parameters[parameter_id] = round(temp_parameter, 2)

		else: # Increment of the selected optimization parameter because its decrement led to out of bounds.
			parameters[parameter_id] = round(parameters[parameter_id] + step_sizes[parameter_id], 2)
			plus_or_minus = 1
	# --------------------------------------------------------------------------------------------------------------------------------

	return [parameter_id, plus_or_minus]

def computation_with_objective_metrics(input_queue, output_queue):

	"""A utility function for computing an objective metric using multiprocessing.

Inputs:
  input_queue: A multiprocessing.Queue object that contains all the necessary parameters for the computation of the
  automated_denoising_and_evaluation.wdwfs_and_objective_metrics() function.

  output_queue: A multiprocessing.Queue object that contains the computed metric values. At first, it must be empty. Every time a
  process computes a metric value, this queue object is filled with that value.

Outputs:
  None. Indirectly, the output_queue parameter is the output of this function."""

	while True:
		try:
			clean_signal, noisy_signal, metric_id, wdwf_id, in_use, is_dwt, dwt_level, pars = input_queue.get_nowait()

		except Queue.Empty:
			break

		else:
			metric_value = wdwfs_and_objective_metrics(clean_signal, noisy_signal, sampling_rate, False, \
													   metric_id, wdwf_id, in_use, is_dwt, dwt_level, *pars)

			output_queue.put(metric_value)

	return

def total_metric_computation(parameters, metric_id, rapp_platform_api_instance, asr_id, audio_source, vocabulary, language, \
							 recognition_content, dataset, noise_estimation_process_id, wdwf_id, in_use, is_dwt, dwt_level):

	"""Evaluates the performance, by computing a metric, of an automated denoising procedure applied to a dataset.

Inputs:
  parameters: A list that contains the parameters needed for the automated denoising procedure to operate.
  The order of the list must be [db_cutoff, a, b, c, d, dps]. More in optimization_algorithm().

  metric_id: Determines which metric will be computed. More in optimization_algorithm().
  If metric_id = 1, the Magnitude-Squared Coherence (MSC) will be computed.
  If metric_id = 2, the Coherence-based Speech Intelligibility Index (CSII) will be computed.
  If metric_id = 3, the high-level CSII (CSII_high) will be computed.
  If metric_id = 4, the mid-level CSII (CSII_mid) will be computed.
  If metric_id = 5, the low-level CSII (CSII_low) will be computed.
  If metric_id != 1, 2, 3, 4, 5, 7 the I3 will be computed.
  If metric_id = 7, the Automatic Word Recognition Precision (AWRP) will be computed.

  rapp_platform_api_instance: An instance of the class RappPlatformAPI of the module RappPlatformApi.py of the package RappCloud.
  If metric_id != 7, it is redundant.

  asr_id: Determines the Automatic Speech Recognition (ASR) system that will be used. More in optimization_algorithm().
  If metric_id != 7, it is redundant.

  audio_source: Defined in [1]. Usually, audio_source = "headset". If metric_id != 7, it is redundant.
  vocabulary: Defined as words in [1]. If metric_id != 7, it is redundant. If metric_id = 7 and asr_id != 1, it is redundant too.

  language:
    If metric_id != 7: It is redundant.
    If metric_id = 7 and asr_id = 1: It is defined as in [1]. Either language = "en" or language = "el".
    If metric_id = 7 and asr_id != 1: language = ["en", "el"] or language = "en" or language = "el".

  recognition_content: If metric_id = 7, it is a list with so many strings as the number of wav files that will participate in the
  optimization procedure. Each string represents the word that is recorded in the corresponding wav file. Otherwise, it is redundant.

  dataset:
    If metric_id != 7: dataset is a list with so many nested lists as the number of sentences that will be used for the computation of
    the selected metric. Each nested list corresponds to a sentence and consists of so many 1-D numpy arrays as the number of the wav
    files that correspond to that specific sentence. The first 1-D numpy array must be the clean wav file while the others must
    represent noisy versions of that file.

    If metric_id = 7: dataset is a list with so many nested 2-D numpy arrays as the number of the wav files that will be used for the
    computation of the AWRP metric.

  noise_estimation_process_id: 1 if the noise_footprint_creation.best_esnr_channel_selection() will be used.
  Different from 1 if the noise_footprint_creation.esnr_and_common_zero_areas() will be used. More in optimization_algorithm().

  wdwf_id: Determines which Wavelet Domain Wiener Filter (WDWF) will be used. More in optimization_algorithm().
  If wdwf_id = 1, the WDWF_0 will be used.
  If wdwf_id = 2, the WDWF_I will be used.
  If wdwf_id = 3, the WDWF_I&II will be used.
  If wdwf_id != 1, 2, 3, the WDWF_II will be used.

  in_use: True if the perceptual filter for broad-band acoustic noise will be used. More in optimization_algorithm().

  is_dwt: True if a Discrete Wavelet Tranform (DWT) will be used (preferably 6-level or 7-level).
  False if the 22-bands Wavelet Packet Analysis (WPA) will be used.

  dwt_level: Defines the level of the DWT (preferably 6 or 7) that will be used. If the WPA is selected, dwt_level = 6.

Outputs:
  mean_metric: The value of the selected metric. Normally, this value is between 0.0 and 100.0 and has a resolution of 0.1.
  If the value is -1.0, then the automated denoising procedure couldn't be applied normally.

[1] RAPP Speech Detection using Sphinx4
(https://github.com/rapp-project/rapp-platform/wiki/RAPP-Speech-Detection-using-Sphinx4)."""

	sum_metrics, number_of_denoisings = 0.0, 0
	invalid_metric_value = False

	if metric_id != 7: # Speech intelligibility metrics.
		for sentence in dataset:
			processes = []
			clean_signal, noisy_dataset = sentence[0], sentence[1:]
			input_queue = multiprocessing.Queue(len(noisy_dataset))
			metrics_queue = multiprocessing.Queue(len(noisy_dataset))

			for noisy_signal in noisy_dataset:
				input_tuple = (clean_signal[:], noisy_signal, metric_id, wdwf_id, in_use, is_dwt, dwt_level, parameters)
				input_queue.put(input_tuple)

			for cpus_count in range(multiprocessing.cpu_count()):
				process = multiprocessing.Process(target = computation_with_objective_metrics, args = (input_queue, metrics_queue))
				process.start()
				processes.append(process)

			for proc in processes:
				proc.join()

			while metrics_queue.empty() == False:
				metric_value = metrics_queue.get()
				sum_metrics = sum_metrics + metric_value

				if metric_value < 0.0:
					invalid_metric_value = True
					break

			if invalid_metric_value:
				break

			number_of_denoisings = number_of_denoisings + len(noisy_dataset)

		if invalid_metric_value:
			mean_metric = -1.0

		else:
			mean_metric = round(sum_metrics / number_of_denoisings, 1)

	else: # Speech recognition system.
		number_of_recognitions = len(recognition_content)

		if type(language) is str:
			if asr_id == 1:
				asr_parameters = (audio_source, language, vocabulary)

			else:
				asr_parameters = (audio_source, language)

		for r_id in range(number_of_recognitions):
			if type(language) is list:
				if r_id < (number_of_recognitions / 2):
					asr_parameters = (audio_source, language[0])

				else:
					asr_parameters = (audio_source, language[1])

			metric_value, error = wdwfs_and_speech_recognition(rapp_platform_api_instance, asr_id, asr_parameters, \
															   recognition_content[r_id], dataset[r_id], sampling_rate, False, \
															   noise_estimation_process_id, wdwf_id, in_use, is_dwt, dwt_level, \
															   *parameters)

			if metric_value < 0.0:
				invalid_metric_value = True
				break

			else:
				if error == "":
					sum_metrics = sum_metrics + metric_value
					number_of_denoisings = number_of_denoisings + 1

		if invalid_metric_value:
			mean_metric = -1.0

		else:
			mean_metric = round(sum_metrics / number_of_denoisings, 1)

	return mean_metric

def peculiar_situation_handler(parameters, step_sizes, total_metric_computation_arguments):

	"""A utility function called from optimization_algorithm() when the value of the evaluation metric is negative.

This situation can arise from a negative enough value of the db_cutoff optimization parameter. More in optimization_algorithm().
Consequently, the db_cutoff parameter must be increased. It is increased by 2 times its step size.

Inputs:
  parameters: A list that contains the values of the optimization parameters.
  Their order is specific and defined in optimization_parameters_space().

  step_sizes: A list which contains the resolutions of the optimization parameters.

  total_metric_computation_arguments: A tuple that contains all the necessary arguments for the total_metric_computation() with the
  correct order apart from the parameters argument.

Outputs:
  metric_value: The value of the evaluation metric after the increment of the db_cutoff parameter.
  parameters: Indirectly, it is an output."""

	multiplier = 2

	while True:
		parameters[0] = round(parameters[0] + (multiplier * step_sizes[0]), 2)
		metric_value = total_metric_computation(parameters, *total_metric_computation_arguments)

		if metric_value >= 0.0:
			break

	return metric_value

def optimization_algorithm(dataset_directory, metric_id, asr_id, language_id, noise_estimation_process_id, wdwf_id, \
						   in_use, is_dwt, dwt_level, number_of_iterations, number_of_attempts_before_initialization):

	"""Implements a combination of Random Restart Hill Climbing (RRHC) and Stochastic Hill Climbing (SHC) algorithms in order to find
out the optimal parameters for the automated denoising procedures.

This combination could be called the Random Restart and Stochastic Hill Climbing (RRSHC) optimization algorithm.
The RRSHC can operate in cooperation with either some objective metric or some speech recognition metric.

At each iteration, the RRSHC changes the value of 1 optimization parameter only. If this change leads to a better metric value, the
RRSHC will proceed to the next iteration. Otherwise, its operation will depend on the number_of_attempts_before_initialization
parameter. If this parameter equals 0, the RRSHC will randomly initialize the optimization parameters and proceed to the next
iteration. However, if number_of_attempts_before_initialization equals 1, the RRSHC will change the value of another optimization
parameter (different from the first) and if this (second) change leads to a better metric value, the RRSHC will proceed to the next
iteration normally (without random initialization). If number_of_attemts_before_initialization equals 2, the RRSHC can fail 3 times in
total, changing the values of 3 different optimization parameters, before proceeds to their initialization and to the next iteration,
and so on. For more information about the operation of the RRSHC, see my Diploma Thesis.

Inputs:
  dataset_directory: A string that represents the path of the directory that contains the dataset that is necessary to the RRSHC.

  metric_id: Determines which metric will be used. The first 6 metrics are described in [1] whereas the last 1 in [2].
  If metric_id = 1, the Magnitude-Squared Coherence (MSC) will be used.
  If metric_id = 2, the Coherence-based Speech Intelligibility Index (CSII) will be used.
  If metric_id = 3, the high-level CSII (CSII_high) will be used.
  If metric_id = 4, the mid-level CSII (CSII_mid) will be used.
  If metric_id = 5, the low-level CSII (CSII_low) will be used.
  If metric_id != 1, 2, 3, 4, 5, 7 the I3 will be used.
  If metric_id = 7, the Automatic Word Recognition Precision (AWRP) will be used.

  asr_id: Determines the Automatic Speech Recognition (ASR) system that will be used. If metric_id != 7, asr_id is redundant.
  If asr_id = 1, Sphinx4 will be used [3]. Otherwise, Google Cloud Speech-to-Text will be used [4].

  language_id: Determines the target language of the ASR system. If metric_id != 7, language_id is redundant.
  If language_id = 1, English is the target language.
  If language_id = 2, Greek is the target language.
  Otherwise, both English and Greek are the target languages.

  noise_estimation_process_id: 1 if the noise_footprint_creation.best_esnr_channel_selection() will be used.
  Different from 1 if the noise_footprint_creation.esnr_and_common_zero_areas() will be used.
  These 2 functions are imported in the automated_denoising_and_evaluation.py module.

  wdwf_id: Determines which Wavelet Domain Wiener Filter (WDWF) will be used. The WDWFs are described in [5] and they are 4.
  If wdwf_id = 1, the WDWF_0 will be used.
  If wdwf_id = 2, the WDWF_I will be used.
  If wdwf_id = 3, the WDWF_I&II will be used.
  If wdwf_id != 1, 2, 3, the WDWF_II will be used.

  in_use: True if the perceptual filter for broad-band acoustic noise, presented in [5], will be used.

  is_dwt: True if a Discrete Wavelet Tranform (DWT) will be used (preferably 6-level or 7-level).
  False if the 22-bands Wavelet Packet Analysis (WPA) will be used.

  dwt_level: Defines the level of the DWT (preferably 6 or 7) that will be used. If the WPA is selected, dwt_level = 6.
  number_of_iterations: The number of iterations of the RRSHC algorithm. Usually, number_of_iterations = 600.

  number_of_attempts_before_initialization: Determines the number of failure times as described above.
  Usually, number_of_attempts_before_initialization = 2.

Outputs:
  best_parameters: A list that contains the optimal parameters that were found. The order is [db_cutoff, a, b, c, d, dps] [2][5].

[1] Ma, J., Hu, Y., and Loizou, P.C. (2009). Objective measures for predicting speech intelligibility in noisy conditions based on new
band-importance functions. The Journal of the Acoustical Society of America, volume 125, issue 5, pages 3387-3405, May 2009.

[2] Tsardoulias, E.G., Thallas, A.G., Symeonidis, A.L., and Mitkas, P.A. (2016). Improving multilingual interaction for consumer
robots through signal enhancement in multichannel speech. JAES, volume 64, issue 7/8, pages 514-524, July 2016.

[3] Sphinx-4 is a state-of-the-art, speaker-independent, continuous speech recognition system written entirely in the Java programming
language (https://github.com/cmusphinx/sphinx4).

[4] Google Cloud Speech-to-Text is a speech-to-text conversion system powered by machine learning and it is available for short-form
or long-form audio (https://cloud.google.com/speech-to-text/).

[5] Dimoulas, C., Kalliris, G., Papanikolaou, G., and Kalampakas, A. (2006). Novel wavelet domain Wiener filtering de-noising
techniques: Application to bowel sounds captured by means of abdominal surface vibrations.
Biomedical Signal Processing and Control, volume 1, issue 3, pages 177-218, July 2006."""

	t_start = time.time()

	if metric_id != 7: # Speech intelligibility metrics.
		rapp_platform_api_instance, reformed_simple_dataset, transcription = None, [], []
		audio_source, vocabulary, language = "", [], ""
		simple_dataset, optimization_dataset = dataset_configuration_for_objective_metrics(dataset_directory)

	else: # Speech recognition system.
		rapp_platform_api_instance = RappPlatformAPI()
		simple_dataset, optimization_dataset, transcription = dataset_configuration_for_speech_recognition(dataset_directory)
		reformed_simple_dataset = simple_dataset_reformation(simple_dataset)
		audio_source = "headset"
		vocabulary_english = ["medicine", "thirsty", "thirteen"]
		vocabulary_greek = ["δεκατρία", "εικοσιδύο", "επτά"]

		if language_id == 1:
			vocabulary = vocabulary_english
			language = "en"

		elif language_id == 2:
			vocabulary = vocabulary_greek
			language = "el"

		else:
			vocabulary = vocabulary_english + vocabulary_greek

			if asr_id == 1:
				language = "el"

			else:
				language = ["en", "el"]

	if len(simple_dataset) == 0:
		return

	print("")
	print("Iteration: 0")

	parameters_vector, metrics_vector = [], []
	initialization_mode = False
	selected_parameters, selected_signs = [], []
	local_maximums_states = []

	if wdwf_id == 1: # WDWF_0.
		number_of_active_weights = 4

	elif wdwf_id == 2: # WDWF_I.
		number_of_active_weights = 5

	else: # WDWF_I&II, WDWF_II.
		number_of_active_weights = 6

	recognition_content, training_dataset = training_dataset_formation(metric_id, simple_dataset, reformed_simple_dataset, \
																	   optimization_dataset, transcription)

	low_boundaries, high_boundaries, step_sizes, parameters = optimization_parameters_space(noise_estimation_process_id)
	probability_weights = probability_weights_computation([], metric_id, -1, 0.0, number_of_active_weights)
	plus_or_minus_weights = plus_or_minus_weights_computation([], -1, -1, 0.0, number_of_active_weights)

	metric_value = total_metric_computation(parameters, metric_id, rapp_platform_api_instance, asr_id, audio_source, vocabulary, \
											language, recognition_content, training_dataset, noise_estimation_process_id, \
											wdwf_id, in_use, is_dwt, dwt_level)

	if metric_value < 0.0:
		arguments = (metric_id, rapp_platform_api_instance, asr_id, audio_source, vocabulary, language, recognition_content, \
					 training_dataset, noise_estimation_process_id, wdwf_id, in_use, is_dwt, dwt_level)

		metric_value = peculiar_situation_handler(parameters, step_sizes, arguments)

	parameters_vector.append(parameters[:])
	metrics_vector.append(metric_value)
	print(parameters_vector[0], metrics_vector[0])

	for i in range(1, number_of_iterations):
		print("")
		print("Iteration: " + str(i))

		if initialization_mode:
			initialization_mode = False
			probability_weights = probability_weights_computation([], metric_id, -1, 0.0, number_of_active_weights)
			plus_or_minus_weights = plus_or_minus_weights_computation([], -1, -1, 0.0, number_of_active_weights)

			recognition_content, training_dataset = training_dataset_formation(metric_id, simple_dataset, reformed_simple_dataset, \
																			   optimization_dataset, transcription)

			parameters = initial_parameters_random_generator(parameters_vector, metrics_vector, \
															 low_boundaries, high_boundaries, step_sizes)

			metric_value = total_metric_computation(parameters, metric_id, rapp_platform_api_instance, asr_id, audio_source, \
													vocabulary, language, recognition_content, training_dataset, \
													noise_estimation_process_id, wdwf_id, in_use, is_dwt, dwt_level)

			if metric_value < 0.0:
				arguments = (metric_id, rapp_platform_api_instance, asr_id, audio_source, vocabulary, language, recognition_content, \
							 training_dataset, noise_estimation_process_id, wdwf_id, in_use, is_dwt, dwt_level)

				metric_value = peculiar_situation_handler(parameters, step_sizes, arguments)

			parameters_vector.append(parameters[:])
			metrics_vector.append(metric_value)

			print(parameters_vector[i], metrics_vector[i])
			continue

		parameter_id, plus_or_minus = parameters_selection_and_change(parameters, probability_weights, plus_or_minus_weights, \
																	  low_boundaries, high_boundaries, step_sizes)

		selected_parameters.append(parameter_id)
		selected_signs.append(plus_or_minus)

		metric_value = total_metric_computation(parameters, metric_id, rapp_platform_api_instance, asr_id, audio_source, vocabulary, \
												language, recognition_content, training_dataset, noise_estimation_process_id, \
												wdwf_id, in_use, is_dwt, dwt_level)

		if metric_value < 0.0:
			arguments = (metric_id, rapp_platform_api_instance, asr_id, audio_source, vocabulary, language, recognition_content, \
						 training_dataset, noise_estimation_process_id, wdwf_id, in_use, is_dwt, dwt_level)

			metric_value = peculiar_situation_handler(parameters, step_sizes, arguments)
			selected_parameters.append(0)
			selected_signs.append(1)

		change_constant = (metric_value - metrics_vector[i - 1]) / 100

		if change_constant > 0.0:
			parameters_vector.append(parameters[:])
			metrics_vector.append(metric_value)

			probability_weights_computation(probability_weights, metric_id, selected_parameters[-1], change_constant, \
											number_of_active_weights)

			plus_or_minus_weights_computation(plus_or_minus_weights, selected_parameters[-1], selected_signs[-1], \
											  change_constant, number_of_active_weights)

		else:
			state_parameters = parameters_vector[i - 1][:]
			state_metric = metrics_vector[i - 1]
			state = (state_parameters, state_metric)
			excluded_parameters = [selected_parameters[-1]]

			if state_metric == 100.0:
				perfection = before_initialization(parameters_vector, parameters, metrics_vector, metric_value, \
												   local_maximums_states, state)

				break

			for att in range(number_of_attempts_before_initialization):
				probability_weights_computation(probability_weights, metric_id, selected_parameters[-1], change_constant, \
												number_of_active_weights)

				plus_or_minus_weights_computation(plus_or_minus_weights, selected_parameters[-1], selected_signs[-1], \
												  change_constant, number_of_active_weights)

				temp_probability_weights = probability_weights_redistribution(probability_weights, excluded_parameters, \
																			  number_of_active_weights)

				parameters = state_parameters[:]

				parameter_id, plus_or_minus = parameters_selection_and_change(parameters, temp_probability_weights, \
																			  plus_or_minus_weights, low_boundaries, \
																			  high_boundaries, step_sizes)

				selected_parameters.append(parameter_id)
				selected_signs.append(plus_or_minus)

				metric_value = total_metric_computation(parameters, metric_id, rapp_platform_api_instance, asr_id, audio_source, \
														vocabulary, language, recognition_content, training_dataset, \
														noise_estimation_process_id, wdwf_id, in_use, is_dwt, dwt_level)

				if metric_value < 0.0:
					arguments = (metric_id, rapp_platform_api_instance, asr_id, audio_source, vocabulary, language, \
								 recognition_content, training_dataset, noise_estimation_process_id, \
								 wdwf_id, in_use, is_dwt, dwt_level)

					metric_value = peculiar_situation_handler(parameters, step_sizes, arguments)
					selected_parameters.append(0)
					selected_signs.append(1)

				change_constant = (metric_value - state_metric) / 100

				if change_constant > 0.0:
					parameters_vector.append(parameters[:])
					metrics_vector.append(metric_value)

					probability_weights_computation(probability_weights, metric_id, selected_parameters[-1], change_constant, \
													number_of_active_weights)

					plus_or_minus_weights_computation(plus_or_minus_weights, selected_parameters[-1], selected_signs[-1], \
													  change_constant, number_of_active_weights)

					break

				else:
					if att == (number_of_attempts_before_initialization - 1):
						initialization_mode = before_initialization(parameters_vector, parameters, metrics_vector, metric_value, \
																	local_maximums_states, state)

					else:
						excluded_parameters.append(selected_parameters[-1])

		print(parameters_vector[i], metrics_vector[i])

	if initialization_mode == False:
		local_maximum = (parameters_vector[-1], metrics_vector[-1])
		local_maximums_states.append(local_maximum)

	best_parameters_id = metrics_vector.index(max(metrics_vector))
	best_parameters = parameters_vector[best_parameters_id]

	t_end = time.time()
	print("")
	print("Duration of the optimization procedure (in minutes): " + str((t_end - t_start) / 60))

	return best_parameters