#!/usr/bin/python
# encoding: utf-8

import Queue
import multiprocessing
import time

from RappCloud import RappPlatformAPI
from automated_denoising_and_evaluation import wdwfs_and_objective_metrics, wdwfs_and_speech_recognition
from dataset_configuration import dataset_configuration_for_objective_metrics, dataset_configuration_for_speech_recognition

# Global Variables -------------------------------------------------------------------------------------------------------------------
sampling_rate = 16000 # A different value is going to give a result but it will most probably be wrong.
# ------------------------------------------------------------------------------------------------------------------------------------

# The testing_dataset_formation() has the same form with the optimization.simple_dataset_reformation().
# It is bad design that I get to have 2 so similar functions!
def testing_dataset_formation(dataset):

	"""Forms the dataset in an effective way for the testing procedure to operate when it is based on a speech recognition system.

Inputs:
  dataset: A list with so many nested lists as the number of languages of the dataset. Each nested list contains so many nested
  lists as the number of language-specific words of the dataset. Each nested nested list contains so many nested lists as the number
  of noise levels of the dataset. Each nested nested nested list contains so many 2-D numpy arrays as the number of wav files that
  correspond to a specific language, word and noise level.

Outputs:
  testing_dataset: A list with so many 2-D numpy arrays as the number of wav files of the dataset."""

	testing_dataset = []
	number_of_languages = len(dataset)

	for language_id in range(number_of_languages):
		number_words = len(dataset[language_id])

		for word_id in range(number_words):
			number_of_noise_levels = len(dataset[language_id][word_id])

			for nl_id in range(number_of_noise_levels):
				number_of_wav_files = len(dataset[language_id][word_id][nl_id])

				for wav_id in range(number_of_wav_files):
					testing_dataset.append(dataset[language_id][word_id][nl_id][wav_id])

	return testing_dataset

# The computation_with_objective_metrics() has the same form with the optimization.computation_with_objective_metrics().
# It is bad design that I get to have 2 so similar functions!
def computation_with_objective_metrics(input_queue, output_queue):

	"""A utility function for computing an objective metric using multiprocessing.

Inputs:
  input_queue: A multiprocessing.Queue object that contains all the necessary parameters for the computation of the
  automated_denoising_and_evaluation.wdwfs_and_objective_metrics() function.

  output_queue: A multiprocessing.Queue object that, at first, it must be empty.
  Every time a process computes a metric value, this queue object is filled with a list that contains 3 elements.
  The first one represents the metric value, the second one the denoising time and the third one the evaluation time.

Outputs:
  None. Indirectly, the output_queue parameter is the output of this function."""

	while True:
		try:
			clean_signal, noisy_signal, metric_id, wdwf_id, in_use, is_dwt, dwt_level, pars = input_queue.get_nowait()

		except Queue.Empty:
			break

		else:
			metric_value, denoising_time, evaluation_time = wdwfs_and_objective_metrics(clean_signal, noisy_signal, sampling_rate, \
																						True, metric_id, wdwf_id, in_use, is_dwt, \
																						dwt_level, *pars)

			output_queue.put([metric_value, denoising_time, evaluation_time])

	return

def total_metric_computation(parameters, metric_id, rapp_platform_api_instance, asr_id, audio_source, vocabulary, language, \
							 recognition_content, dataset, noise_estimation_process_id, wdwf_id, in_use, is_dwt, dwt_level):

	"""Evaluates the performance, by computing a metric, of an automated denoising procedure applied to a dataset.

Inputs:
  parameters: A list that contains the parameters needed for the automated denoising procedure to operate.
  The order of the list must be [db_cutoff, a, b, c, d, dps]. More in testing().

  metric_id: Determines which metric will be computed. More in testing().
  If metric_id = 1, the Magnitude-Squared Coherence (MSC) will be computed.
  If metric_id = 2, the Coherence-based Speech Intelligibility Index (CSII) will be computed.
  If metric_id = 3, the high-level CSII (CSII_high) will be computed.
  If metric_id = 4, the mid-level CSII (CSII_mid) will be computed.
  If metric_id = 5, the low-level CSII (CSII_low) will be computed.
  If metric_id != 1, 2, 3, 4, 5, 7 the I3 will be computed.
  If metric_id = 7, the Automatic Word Recognition Precision (AWRP) will be computed.

  rapp_platform_api_instance: An instance of the class RappPlatformAPI of the module RappPlatformApi.py of the package RappCloud.
  If metric_id != 7, it is redundant.

  asr_id: Determines the Automatic Speech Recognition (ASR) system that will be used. More in testing().
  If metric_id != 7, it is redundant.

  audio_source: Defined in [1]. Usually, audio_source = "headset". If metric_id != 7, it is redundant.
  vocabulary: Defined as words in [1]. If metric_id != 7, it is redundant. If metric_id = 7 and asr_id != 1, it is redundant too.

  language:
    If metric_id != 7: It is redundant.
    If metric_id = 7 and asr_id = 1: It is defined as in [1]. Either language = "en" or language = "el".
    If metric_id = 7 and asr_id != 1: language = ["en", "el"] or language = "en" or language = "el".

  recognition_content: If metric_id = 7, it is a list with so many strings as the number of wav files that will participate in the
  testing procedure. Each string represents the word that is recorded in the corresponding wav file. Otherwise, it is redundant.

  dataset:
    If metric_id != 7: dataset is a list with so many nested lists as the number of sentences that will be used for the computation of
    the selected metric. Each nested list corresponds to a sentence and consists of so many 1-D numpy arrays as the number of the wav
    files that correspond to that specific sentence. The first 1-D numpy array must be the clean wav file while the others must
    represent noisy versions of that file.

    If metric_id = 7: dataset is a list with so many nested 2-D numpy arrays as the number of the wav files that will be used for the
    computation of the AWRP metric.

  noise_estimation_process_id: 1 if the noise_footprint_creation.best_esnr_channel_selection() will be used.
  Different from 1 if the noise_footprint_creation.esnr_and_common_zero_areas() will be used. More in testing().

  wdwf_id: Determines which Wavelet Domain Wiener Filter (WDWF) will be used. More in testing().
  If wdwf_id = 1, the WDWF_0 will be used.
  If wdwf_id = 2, the WDWF_I will be used.
  If wdwf_id = 3, the WDWF_I&II will be used.
  If wdwf_id != 1, 2, 3, the WDWF_II will be used.

  in_use: True if the perceptual filter for broad-band acoustic noise will be used. More in testing().

  is_dwt: True if a Discrete Wavelet Tranform (DWT) will be used (preferably 6-level or 7-level).
  False if the 22-bands Wavelet Packet Analysis (WPA) will be used.

  dwt_level: Defines the level of the DWT (preferably 6 or 7) that will be used. If the WPA is selected, dwt_level = 6.

Outputs:
  mean_metric: The average value of the selected metric. It has a 0.1 resolution and it must be between 0.0 and 100.0.
  sum_denoising_times: The total denoising time in minutes with a 0.1 resolution.
  sum_evaluation_times: The total evaluation time in minutes with a 0.1 resolution.

[1] RAPP Speech Detection using Sphinx4
(https://github.com/rapp-project/rapp-platform/wiki/RAPP-Speech-Detection-using-Sphinx4)."""

	sum_metrics, sum_denoising_times, sum_evaluation_times = 0.0, 0.0, 0.0
	number_of_denoisings = 0

	if metric_id != 7: # Speech intelligibility metrics.
		for sentence in dataset:
			processes = []
			clean_signal, noisy_dataset = sentence[0], sentence[1:]
			input_queue = multiprocessing.Queue(len(noisy_dataset))
			output_queue = multiprocessing.Queue(len(noisy_dataset))

			for noisy_signal in noisy_dataset:
				input_tuple = (clean_signal[:], noisy_signal, metric_id, wdwf_id, in_use, is_dwt, dwt_level, parameters)
				input_queue.put(input_tuple)

			for cpus_count in range(multiprocessing.cpu_count()):
				process = multiprocessing.Process(target = computation_with_objective_metrics, args = (input_queue, output_queue))
				process.start()
				processes.append(process)

			for proc in processes:
				proc.join()

			while output_queue.empty() == False:
				metric_value, denoising_time, evaluation_time = output_queue.get()
				sum_denoising_times = sum_denoising_times + denoising_time
				sum_evaluation_times = sum_evaluation_times + evaluation_time

				if metric_value >= 0.0:
					sum_metrics = sum_metrics + metric_value
					number_of_denoisings = number_of_denoisings + 1
					print("Denoising: " + str(number_of_denoisings))

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

			metric_value, error, denoising_time, evaluation_time = wdwfs_and_speech_recognition(rapp_platform_api_instance, asr_id, \
																								asr_parameters, \
																								recognition_content[r_id], \
																								dataset[r_id], sampling_rate, True, \
																								noise_estimation_process_id, \
																								wdwf_id, in_use, is_dwt, dwt_level, \
																								*parameters)

			sum_denoising_times = sum_denoising_times + denoising_time
			sum_evaluation_times = sum_evaluation_times + evaluation_time

			if (metric_value >= 0.0) and (error == ""):
				sum_metrics = sum_metrics + metric_value
				number_of_denoisings = number_of_denoisings + 1
				print("Denoising: " + str(number_of_denoisings))

	mean_metric = round(sum_metrics / number_of_denoisings, 1)
	sum_denoising_times = round(sum_denoising_times / 60, 1)
	sum_evaluation_times = round(sum_evaluation_times / 60, 1)

	return [mean_metric, sum_denoising_times, sum_evaluation_times]

def testing(dataset_directory, metric_id, asr_id, language_id, noise_estimation_process_id, wdwf_id, \
			in_use, is_dwt, dwt_level, parameters):

	"""Tests the performance of an automated denoising algorithm upon a whole dataset.

Inputs:
  dataset_directory: A string that represents the path of the directory that contains the dataset.

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
  parameters: [db_cutoff, a, b, c, d, dps]. For more information, see the automated_denoising_and_evaluation.py module.

Outputs:
  mean_metric: The average value of the selected metric. It has a 0.1 resolution and it must be between 0.0 and 100.0.
  total_denoising_time: The total denoising time in minutes with a resolution of 0.1.
  total_evaluation_time: The total evaluation time in minutes with a resolution of 0.1.

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

	if metric_id != 7: # Speech intelligibility metrics.
		rapp_platform_api_instance, transcription = None, []
		audio_source, vocabulary, language = "", [], ""
		simple_dataset, testing_dataset = dataset_configuration_for_objective_metrics(dataset_directory)

	else: # Speech recognition system.
		rapp_platform_api_instance = RappPlatformAPI()
		simple_dataset, arrays_dataset, transcription = dataset_configuration_for_speech_recognition(dataset_directory)
		testing_dataset = testing_dataset_formation(arrays_dataset)
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

	t_start = time.time()
	mean_metric, total_denoising_time, total_evaluation_time = total_metric_computation(parameters, metric_id, \
																						rapp_platform_api_instance, asr_id, \
																						audio_source, vocabulary, language, \
																						transcription, testing_dataset, \
																						noise_estimation_process_id, wdwf_id, \
																						in_use, is_dwt, dwt_level)
	t_end = time.time()
	print("Duration of the testing procedure (in minutes): " + str((t_end - t_start) / 60))

	return [mean_metric, total_denoising_time, total_evaluation_time]