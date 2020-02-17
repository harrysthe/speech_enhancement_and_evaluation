import numpy
import os

from scipy.io import wavfile
from noise_footprint_creation import get_channels_from_numpy_array # This is bad design!

# Global Variables -------------------------------------------------------------------------------------------------------------------
sampling_rate = 16000
encoding_type = numpy.int16
# ------------------------------------------------------------------------------------------------------------------------------------

def lengths_equality_checking(strings_dataset, arrays_dataset):

	"""Checks if all the signals that correspond to a sentence have the same length.

Inputs:
  strings_dataset: A list with so many nested lists as the number of sentences of the dataset.
  Each nested list contains so many strings as the number of wav files that correspond to a specific sentence.
  The first string is the name of the clean wav file while the other strings are the names of the noisy wav files.

  arrays_dataset: Its structure is identical to that of the strings_dataset. The difference is that the strings_dataset contains
  the names of the wav files while the arrays_dataset replaces these names with their 1-D numpy arrays.

Outputs:
  checked: True if all the signals that correspond to a sentence have the same length. False otherwise."""

	number_of_sentences = len(strings_dataset)
	checked = True

	for sentence_id in range(number_of_sentences):
		number_of_wav_files = len(strings_dataset[sentence_id])

		if checked == False:
			break

		for wav_id in range(number_of_wav_files):
			if wav_id == 0:
				sentence_length = arrays_dataset[sentence_id][0].shape[0]

			else:
				temp_length = arrays_dataset[sentence_id][wav_id].shape[0]

				if temp_length != sentence_length:
					print("")
					print("dataset_configuration.lengths_equality_checking():")
					print("The clean and noisy wav files have not the same length!")
					print("Name of the clean wav file: " + strings_dataset[sentence_id][0])
					print("Name of the noisy wav file: " + strings_dataset[sentence_id][wav_id])
					print("")

					checked = False
					break

	return checked

def dataset_configuration_for_objective_metrics(dataset_directory):

	"""Forms the dataset in a way effective for reading its signals.

The wav files in the dataset can be at any sampling rate but they are converted to the global variable sampling_rate using SoX [1].
The wav files in the dataset must be single-channel and their format must be the one specified by the global variable encoding_type.
The wav files in the dataset cannot be zero-signals.

Any clean signal and its corresponding noisy signals must have the same length.

Inputs:
  dataset_directory: A string that represents the path of the directory that contains the dataset.

Outputs:
  strings_dataset: A list with so many nested lists as the number of sentences of the dataset.
  Each nested list contains so many strings as the number of wav files that correspond to a specific sentence.
  The first string is the name of the clean wav file while the other strings are the names of the noisy wav files.

  arrays_dataset: Its structure is identical to that of the strings_dataset. The difference is that the strings_dataset contains
  the names of the wav files while the arrays_dataset replaces these names with their 1-D numpy arrays.

  Both of the outputs will be empty lists if an unpleasant situation come to surface.

[1] SoX is cross-platform command line utility for computer audio files (http://sox.sourceforge.net/Main/HomePage)."""

	strings_dataset, arrays_dataset = [], []
	sentences = os.listdir(dataset_directory)
	sentences.sort() # Practically, this sort is unnecessary.

	for sentence in sentences:
		temp_arrays_dataset = []
		temp_strings_dataset = os.listdir(dataset_directory + "/" + sentence)
		temp_strings_dataset.sort() # This sort is necessary so as the clean wav file to be the first string in the list.

		for wav_file in temp_strings_dataset:
			temp_path = dataset_directory + "/" + sentence + "/" + wav_file
			sr, wav_data = wavfile.read(temp_path)

			# Checking if the wav_file is a single-channel signal. -------------------------------------------------------------------
			if len(wav_data.shape) != 1:
				print("")
				print("dataset_configuration.dataset_configuration_for_objective_metrics():")
				print("The wav file is not a single-channel signal!")
				print("Name of the file: " + temp_path)
				print("")

				return [[], []]
			# ------------------------------------------------------------------------------------------------------------------------

			# Checking if the wav_file has the same format with the one specified by the global variable encoding_type. --------------
			if wav_data.dtype != encoding_type:
				print("")
				print("dataset_configuration.dataset_configuration_for_objective_metrics():")
				print("The wav file is not at " + str(encoding_type) + " format!")
				print("Name of the file: " + temp_path)
				print("")

				return [[], []]
			# ------------------------------------------------------------------------------------------------------------------------

			# Checking if the wav_file is a zero-signal. -----------------------------------------------------------------------------
			wav_data_list = list(wav_data)

			if wav_data_list.count(0) == len(wav_data_list):
				print("")
				print("dataset_configuration.dataset_configuration_for_objective_metrics():")
				print("The wav file is a zero-signal!")
				print("Name of the file: " + temp_path)
				print("")

				return [[], []]
			# ------------------------------------------------------------------------------------------------------------------------

			# Checking the sampling rate of the wav_file. ----------------------------------------------------------------------------
			if sr != sampling_rate:
				command = "sox " + temp_path + " -r " + str(sampling_rate) + " wav_file.wav"
				os.system(command)
				sr, wav_data = wavfile.read("wav_file.wav")
				command = "rm wav_file.wav"
				os.system(command)
			# ------------------------------------------------------------------------------------------------------------------------

			temp_arrays_dataset.append(wav_data)

		strings_dataset.append(temp_strings_dataset)
		arrays_dataset.append(temp_arrays_dataset)

	if lengths_equality_checking(strings_dataset, arrays_dataset) == False:
		return [[], []]

	return [strings_dataset, arrays_dataset]

def dataset_configuration_for_speech_recognition(dataset_directory):

	"""Forms the dataset in a way effective for reading its signals.

The wav files in the dataset can be at any sampling rate but they are converted to the global variable sampling_rate using SoX [1].
The wav files in the dataset must be multi-channel and their format must be the one specified by the global variable encoding_type.
The wav files in the dataset cannot have even one zero-signal channel.

Inputs:
  dataset_directory: A string that represents the path of the directory that contains the dataset.

Outputs:
  strings_dataset: A list with so many nested lists as the number of languages of the dataset. Each nested list contains so many
  nested lists as the number of language-specific words of the dataset. Each nested nested list contains so many nested lists as the
  number of noise levels of the dataset. Each nested nested nested list contains so many strings as the number of wav files that
  correspond to a specific language, word and noise level. Each string represents the name of a wav file.

  arrays_dataset: Its structure is identical to that of the strings_dataset. The difference is that the strings_dataset contains
  the names of the wav files while the arrays_dataset replaces these names with their 2-D numpy arrays.

  transcription: A list with so many strings as the number of wav files of the dataset.
  Each string represents the word that corresponds to the specific wav file.

  If an unpleasant situation come to surface, the 3 outputs will be empty lists.

[1] SoX is cross-platform command line utility for computer audio files (http://sox.sourceforge.net/Main/HomePage)."""

	strings_dataset, arrays_dataset = [], []
	languages = os.listdir(dataset_directory)
	transcription_id = languages.index("transcription.txt")
	transcription_txt_file = languages.pop(transcription_id)
	languages.sort() # This sort is necessary so as the languages directories to be accessed with alphabetical order.

	for language_dir in languages:
		strings_dataset_at_language_dir, arrays_dataset_at_language_dir = [], []
		words = os.listdir(dataset_directory + "/" + language_dir)
		words.sort() # This sort is necessary so as the words directories to be accessed with alphabetical order.

		for word_dir in words:
			strings_dataset_at_word_dir, arrays_dataset_at_word_dir = [], []
			noise_levels = os.listdir(dataset_directory + "/" + language_dir + "/" + word_dir)
			noise_levels.sort() # Practically, this sort is unnecessary.

			for noise_level_dir in noise_levels:
				temp_arrays_dataset = []
				temp_strings_dataset = os.listdir(dataset_directory + "/" + language_dir + "/" + word_dir + "/" + noise_level_dir)
				temp_strings_dataset.sort() # Practically, this sort is unnecessary.

				for wav_file in temp_strings_dataset:
					temp_path = dataset_directory + "/" + language_dir + "/" + word_dir + "/" + noise_level_dir + "/" + wav_file
					sr, wav_data = wavfile.read(temp_path)

					# Checking if the wav_file is a multi-channel signal. ------------------------------------------------------------
					if len(wav_data.shape) == 1:
						print("")
						print("dataset_configuration.dataset_configuration_for_speech_recognition():")
						print("The wav file is not a multi-channel signal!")
						print("Name of the file: " + temp_path)
						print("")

						return [[], [], []]
					# ----------------------------------------------------------------------------------------------------------------

					# Checking if the wav_file has the same format with the one specified by the global variable encoding_type. ------
					if wav_data.dtype != encoding_type:
						print("")
						print("dataset_configuration.dataset_configuration_for_speech_recognition():")
						print("The wav file is not at " + str(encoding_type) + " format!")
						print("Name of the file: " + temp_path)
						print("")

						return [[], [], []]
					# ----------------------------------------------------------------------------------------------------------------

					# Checking if at least one channel of the wav_file is a zero-signal. ---------------------------------------------
					wav_data_list = get_channels_from_numpy_array(wav_data)
					zero_channel_flag = False

					for wav_channel in wav_data_list:
						if wav_channel.count(0) == len(wav_channel):
							zero_channel_flag = True
							break

					if zero_channel_flag:
						print("")
						print("dataset_configuration.dataset_configuration_for_speech_recognition():")
						print("At least one channel of the wav file is a zero-signal!")
						print("Name of the file: " + temp_path)
						print("")

						return [[], [], []]
					# ----------------------------------------------------------------------------------------------------------------

					# Checking the sampling rate of the wav_file. --------------------------------------------------------------------
					if sr != sampling_rate:
						command = "sox -V1 " + temp_path + " -r " + str(sampling_rate) + " wav_file.wav"
						os.system(command)
						sr, wav_data = wavfile.read("wav_file.wav")
						command = "rm wav_file.wav"
						os.system(command)
					# ----------------------------------------------------------------------------------------------------------------

					temp_arrays_dataset.append(wav_data)

				strings_dataset_at_word_dir.append(temp_strings_dataset)
				arrays_dataset_at_word_dir.append(temp_arrays_dataset)

			strings_dataset_at_language_dir.append(strings_dataset_at_word_dir)
			arrays_dataset_at_language_dir.append(arrays_dataset_at_word_dir)

		strings_dataset.append(strings_dataset_at_language_dir)
		arrays_dataset.append(arrays_dataset_at_language_dir)

	with open(dataset_directory + "/" + transcription_txt_file) as ttf:
		transcription = ttf.readlines()

	transcription = [word.strip() for word in transcription]

	return [strings_dataset, arrays_dataset, transcription]