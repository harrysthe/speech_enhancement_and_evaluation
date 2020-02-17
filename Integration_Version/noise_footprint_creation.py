import numpy
from math import log

# Global Variables -------------------------------------------------------------------------------------------------------------------
window_duration = 0.016 # In seconds. Represents the duration of the rectangular non-overlapping windows used in the computation
						# of the Short-Term Energy. Referenced in best_esnr_channel_selection() and esnr_and_common_zero_areas().

energy_low_boundary = -60.0 # In dB. Represents the minimum short-term energy value. Any short-term energy value lesser than this
							# constant value is equalised with this constant value. In other_words,
							# if short_term_energy < energy_low_boundary, then short_term_energy = energy_low_boundary.
							# Referenced in short_term_energy_computation_in_db().
# ------------------------------------------------------------------------------------------------------------------------------------

def get_channels_from_numpy_array(multichannel_signal):

	"""Returns the channels of a multichannel signal.

Inputs:
  multichannel_signal: A 2-D numpy array of shape (number_of_samples, number_of_channels).
  It must be multichannel, not single-channel.

Outputs:
  channels: A list with so many nested lists as the number of the channels."""

	channels = []

	for c in range(multichannel_signal.shape[1]):
		channels.append(list(multichannel_signal[:, c]))

	return channels

def short_term_energy_computation_in_db(signal, window_length):

	"""Computes the short-term energies of a signal.

The signal is divided into rectangular non-overlapping windows of window_length samples.
The short-term energy is calculated in dB, with 0.0 dB being the maximum value.
The minimum value of the short-term energy is defined by the global variable energy_low_boundary.
Consequently, the function returns normalized short-term energies expressed in dB.

Inputs:
  signal: Each element of it must represent a sample of the signal.
  window_length: Number of samples of the rectangular window.

Outputs:
  short_term_energies_db: A list with so many elements as the number of the rectangular windows.
  Each element of it represents a normalized short-term energy expressed in dB."""

	short_term_energies, short_term_energies_db = [], []
	energy_counter, energy_sum = 0, 0.0

	for sample in signal:
		energy_sum = energy_sum + (sample ** 2.0)
		energy_counter = energy_counter + 1

		if energy_counter == window_length:
			short_term_energies.append(energy_sum)
			energy_counter, energy_sum = 0, 0.0

	if energy_counter != 0: # This will happen if the last window has less samples than window_length.
		short_term_energies.append(energy_sum)

	max_energy = max(short_term_energies)

	for energy in short_term_energies:
		if energy == 0.0:
			short_term_energies_db.append(energy_low_boundary)

		else:
			short_term_energies_db.append(10.0 * log(energy / max_energy, 10))

	return short_term_energies_db

def esnr_calculation_of_a_single_noisy_channel(noisy_signal, db_cutoff, esnr_method, window_length, noise_estimation_process_id):

	"""Calculates the ESNR (Effective Signal to Noise Ratio) of a noisy single-channel signal.

In order the ESNR to be computed, a simplistic denoising process, based on a threshold, takes place.
As a result, a simplistic denoised signal and a noise footprint are estimated.
The noise footprint consists of the maximum number of consecutive noisy signal's samples that correspond to 0 in the denoised signal.
The ESNR is calculated by using either the maximum value or the maximum short-term energy of the noisy signal.

Inputs:
  noisy_signal: Each element of it must represent a sample of the noisy signal.

  db_cutoff: The threshold needed for the ESNR to be computed. It's value is expressed in negative dB.
  If esnr_method = 1, usually db_cutoff = -20.0 dB.
  If esnr_method != 1, usually db_cutoff = -7.0 dB.

  esnr_method: 1 if the ESNR is calculated using the maximum value of the noisy signal.
  Different from 1 if the ESNR is calculated using the maximum short-term energy of the noisy signal.

  window_length: Number of samples of the rectangular window that is being used in short_term_energy_computation().
  If esnr_method = 1, window_length is redundant.

  noise_estimation_process_id: 1 if the function is called from the best_esnr_channel_selection().
  Different from 1 if the function is called from esnr_and_common_zero_areas().

Outputs:
  If noise_estimation_process_id = 1:
    esnr: The ESNR value of the noisy signal. If it is not feasible to be computed, esnr = None.
    noise_footprint: A list which represents the noise footprint. If db_cutoff = negative enough value, noise_footprint = [].

  If noise_estimation_process_id != 1:
    denoised_signal: A list which represents the simplistic denoised signal."""

	denoised_signal_energy_list, extracted_noise_energy_list = [], []
	denoised_signal, noise_footprints = [], []
	noise_footprint_flag = False

	if esnr_method == 1:
		noisy_signal_energy_list = [sample ** 2.0 for sample in noisy_signal]
		max_energy = max(noisy_signal_energy_list)
		threshold_boundary = 10.0 ** (db_cutoff / 10.0) # Division with 10.0 because I consider power magnitudes, not amplitudes.
		power_threshold = max_energy * threshold_boundary

		for i in range(len(noisy_signal_energy_list)):
			if noisy_signal_energy_list[i] < power_threshold:
				if noise_footprint_flag == False:
					noise_footprint = []
					noise_footprint_flag = True

				extracted_noise_energy_list.append(noisy_signal_energy_list[i])
				noise_footprint.append(noisy_signal[i])
				denoised_signal.append(0)

			else:
				if noise_footprint_flag == True:
					noise_footprints.append(noise_footprint)
					noise_footprint_flag = False

				denoised_signal_energy_list.append(noisy_signal_energy_list[i])
				denoised_signal.append(noisy_signal[i])

		if noise_footprint_flag == True:
			noise_footprints.append(noise_footprint)
			noise_footprint_flag = False

	else:
		short_term_energies_db = short_term_energy_computation_in_db(noisy_signal, window_length)
		max_energy_db = max(short_term_energies_db)
		energy_threshold_db = max_energy_db + db_cutoff # db_cutoff must be negative.

		for energy_time in range(len(short_term_energies_db)):
			noisy_signal_start = energy_time * int(window_length) # In samples.

			if energy_time != (len(short_term_energies_db) - 1):
				noisy_signal_end = ((energy_time + 1) * int(window_length)) # In samples.

			else: # Last window, which may has less samples than window_length.
				noisy_signal_end = len(noisy_signal)

			short_term_noisy_signal = noisy_signal[noisy_signal_start : noisy_signal_end]
			short_term_noisy_energy_list = [sample ** 2.0 for sample in short_term_noisy_signal]

			if short_term_energies_db[energy_time] < energy_threshold_db:
				if noise_footprint_flag == False:
					noise_footprint = []
					noise_footprint_flag = True

				extracted_noise_energy_list.extend(short_term_noisy_energy_list)
				noise_footprint.extend(short_term_noisy_signal)

				for sample in short_term_noisy_signal:
					denoised_signal.append(0)

			else:
				if noise_footprint_flag == True:
					noise_footprints.append(noise_footprint)
					noise_footprint_flag = False

				denoised_signal_energy_list.extend(short_term_noisy_energy_list)
				denoised_signal.extend(short_term_noisy_signal)

		if noise_footprint_flag == True:
			noise_footprints.append(noise_footprint)
			noise_footprint_flag = False

	if noise_estimation_process_id != 1: # esnr_and_common_zero_areas().
		return denoised_signal

	else: # best_esnr_channel_selection().
		# Checking if the noise footprint has samples. -------------------------------------------------------------------------------
		if len(extracted_noise_energy_list) == 0:
			return [None, []]
		# ----------------------------------------------------------------------------------------------------------------------------

		denoised_signal_power = sum(denoised_signal_energy_list) / len(denoised_signal_energy_list)
		extracted_noise_power = sum(extracted_noise_energy_list) / len(extracted_noise_energy_list)
		noise_footprints_lengths = [len(noise_footprint) for noise_footprint in noise_footprints]
		max_length_index = noise_footprints_lengths.index(max(noise_footprints_lengths))
		noise_footprint = noise_footprints[max_length_index]

		# Checking if noise_footprint is a zero-signal. ------------------------------------------------------------------------------
		if extracted_noise_power == 0.0:
			esnr = None

		else:
			esnr = 10.0 * log(denoised_signal_power / extracted_noise_power, 10)

		return [esnr, noise_footprint]
		# ----------------------------------------------------------------------------------------------------------------------------

def find_common_zero_areas(multichannel_signal, number_of_channels, minimum_number_of_samples):

	"""Finds the segments of a multichannel signal wherein all the channels have 0 value.

Inputs:
  multichannel_signal: A list with so many nested lists as the number of channels.
  number_of_channels: The number of channels of the multichannel signal.
  minimum_number_of_samples: By definition, a segment has at least minimum_number_of_samples consecutive 0-valued samples.

Outputs:
  common_zero_areas: A list with so many nested lists as the number of segments wherein all the channels have 0 value.
  Each list consists of 2 values. The first value represents the sample index from which the segment starts.
  The second value represents the sample index at which the segment ended."""

	start, end = 0, 0
	common_zero_areas = []
	number_of_samples = len(multichannel_signal[0])

	for s in range(number_of_samples):
		for c in range(number_of_channels):
			if multichannel_signal[c][s] != 0:
				if (end - start) >= minimum_number_of_samples:
					common_zero_areas.append([start, end])

				start, end = s, s - 1
				break

		end = end + 1

	if end > start:
		common_zero_areas.append([start, end])

	return common_zero_areas

def noise_footprint_creation_using_common_zero_areas(noisy_signal, common_zero_areas):

	"""Computes the noise footprint by finding the largest common zero area of the multichannel noisy signal.

Inputs:
  noisy_signal: A 1-D numpy array that represents a channel of the multichannel noisy signal.
  common_zero_areas: The output of the find_common_zero_areas().

Outputs:
  noise_footprint: A 1-D numpy array that represents the noise footprint."""

	common_zero_areas_lengths = [common_zero_area[1] - common_zero_area[0] for common_zero_area in common_zero_areas]
	max_length_index = common_zero_areas_lengths.index(max(common_zero_areas_lengths))
	noisy_signal_start = common_zero_areas[max_length_index][0]
	noisy_signal_end = common_zero_areas[max_length_index][1]
	noise_footprint = noisy_signal[noisy_signal_start : noisy_signal_end]

	return noise_footprint

def best_esnr_channel_selection(noisy_signal, sampling_rate, number_of_channels, db_cutoff, esnr_method):

	"""Selects the noisy channel with the best ESNR (Effective Signal to Noise Ratio) and creates a noise footprint from it.

Inputs:
  noisy_signal: A 1-D or 2-D numpy array that represents the noisy signal.
  If it is a 2-D numpy array, its shape is (number_of_samples, number_of_channels).

  sampling_rate: Usually, sampling_rate = 16000 Hz.
  number_of_channels: The number of channels of the noisy signal.

  db_cutoff: The threshold needed for the ESNRs to be computed. It's value is expressed in negative dB.
  If esnr_method = 1, usually db_cutoff = -20.0 dB.
  If esnr_method != 1, usually db_cutoff = -7.0 dB.

  esnr_method: 1 if the ESNR is calculated using the maximum value of the noisy channel.
  Different from 1 if the ESNR is calculated using the maximum short-term energy of the noisy channel.

Outputs:
  If number_of_channels = 1:
    noise_footprint: A 1-D numpy array that represents the noise footprint created from the noisy channel with the best ESNR value.
    If db_cutoff = negative enough value, noise_footprint.shape[0] = 0.

  If number_of_channels != 1:
    best_esnr_noisy_channel: A 1-D numpy array that represents the noisy channel with the best ESNR value.

    noise_footprint: A 1-D numpy array that represents the noise footprint created from the best_esnr_noisy_channel.
    If db_cutoff = negative enough value, noise_footprint.shape[0] = 0."""

	encoding_type = noisy_signal.dtype
	window_length = round(window_duration * sampling_rate) # In samples.

	if number_of_channels == 1:
		esnr, noise_footprint = esnr_calculation_of_a_single_noisy_channel(noisy_signal, db_cutoff, esnr_method, window_length, 1)
		noise_footprint = numpy.array(noise_footprint, dtype = encoding_type) # It can have 0 samples or be a zero-signal.

		return noise_footprint

	else:
		esnr_values, noise_footprints_list = [], []

		for c in range(number_of_channels):
			esnr, noise_footprint = esnr_calculation_of_a_single_noisy_channel(noisy_signal[:, c], db_cutoff, \
																			   esnr_method, window_length, 1)

			esnr_values.append(esnr)
			noise_footprints_list.append(noise_footprint)

		best_esnr_channel = esnr_values.index(max(esnr_values))
		best_esnr_noisy_channel = noisy_signal[:, best_esnr_channel]
		noise_footprint = noise_footprints_list[best_esnr_channel]
		noise_footprint = numpy.array(noise_footprint, dtype = encoding_type) # It can have 0 samples or be a zero-signal.

		return [best_esnr_noisy_channel, noise_footprint]

def esnr_and_common_zero_areas(noisy_signal, sampling_rate, number_of_channels, db_cutoff, esnr_method):

	"""Creates a noise footprint for each channel by computing the ESNRs (Effective Signal to Noise Ratio) of all the noisy channels
and finding the common zero areas among them afterwards.

Calculates the ESNR of each channel.
Denoises each channel separately using its ESNR.
Finds the common zero areas from the denoised channels.
Estimates the noise for each channel using the common zero areas.

Inputs:
  noisy_signal: A 2-D numpy array of shape (number_of_samples, number_of_channels) that represents the noisy signal.
  sampling_rate: Usually, sampling_rate = 16000 Hz.
  number_of_channels = The number of channels of the noisy signal. Must be more than 1.

  db_cutoff: The threshold needed for the ESNRs to be computed. It's value is expressed in negative dB.
  If esnr_method = 1, usually db_cutoff = -20.0 dB.
  If esnr_method != 1, usually db_cutoff = -7.0 dB.

  esnr_method: 1 if the ESNR is calculated using the maximum value of the noisy channel.
  Different from 1 if the ESNR is calculated using the maximum short-term energy of the noisy channel.

Outputs:
  noise_footprints_list: A list with so many nested 1-D numpy arrays as the number_of_channels.
  Each one of the nested arrays represents the noise footprint of the corresponding channel.
  If no common zero areas were found, then noise_footprints_list = []."""

	first_level_denoised_signals_list = []
	window_length = round(window_duration * sampling_rate) # In samples.

	for c in range(number_of_channels):
		first_level_denoised_channel = esnr_calculation_of_a_single_noisy_channel(noisy_signal[:, c], db_cutoff, \
																				  esnr_method, window_length, 2)

		first_level_denoised_signals_list.append(first_level_denoised_channel)

	noise_footprints_list = []
	noise_footprint_segments_list = find_common_zero_areas(first_level_denoised_signals_list, number_of_channels, window_length)

	# Checking if there are common zero areas. ---------------------------------------------------------------------------------------
	if len(noise_footprint_segments_list) == 0:
		return noise_footprints_list
	# --------------------------------------------------------------------------------------------------------------------------------

	for c in range(number_of_channels):
		channel_noise_array = noise_footprint_creation_using_common_zero_areas(noisy_signal[:, c], noise_footprint_segments_list)
		noise_footprints_list.append(channel_noise_array)

	return noise_footprints_list