import numpy

# This module is imported from the wdwf.py module.
# Its functions are useful when the 22-bands Wavelet Packet Analysis (WPA) is being used.

def find_number_of_additional_samples(signal_length, max_level):

	"""Finds the number of samples to be added at the end of the signal so as the signal to have an appropriate length for the
22-bands Wavelet Packet Analysis (WPA).

For more information, see the module wdwf.py.

Inputs:
  signal_length: The initial length of the signal.
  max_level: Maximum allowed level of decomposition. Usually, max_level = 6.

Outputs:
  number_of_additional_samples: The number of zeros that will be added at the end of the signal."""

	number_of_additional_samples = 0
	number_of_accepted_divisions = 0

	while True:
		next_step_variable_condition = signal_length % 2

		if next_step_variable_condition == 0:
			number_of_accepted_divisions = number_of_accepted_divisions + 1

			if number_of_accepted_divisions == max_level:
				break

			else:
				signal_length = signal_length / 2
				continue

		else:
			signal_length = signal_length + 1
			number_of_additional_samples = number_of_additional_samples + 1

	return number_of_additional_samples

def find_the_appropriate_signal_length(signal_length, max_level):

	"""Calculates the new length of the signal so as the signal to have an appropriate length for the
22-bands Wavelet Packet Analysis (WPA).

For more information, see the module wdwf.py.

Inputs:
  signal_length: The initial length of the signal.
  max_level: Maximum allowed level of decomposition. Usually, max_level = 6.

Outputs:
  current_length: The new length of the signal."""

	current_length = signal_length

	while True:
		number_of_additional_samples = find_number_of_additional_samples(current_length, max_level)

		if number_of_additional_samples == 0:
			break

		else:
			current_length = current_length + number_of_additional_samples

	return current_length

def signal_adaptation_for_22_bands_wpa(signal, max_level):

	"""Extends with zeros the length of the signal so as the signal to have an appropriate length for the
22-bands Wavelet Packet Analysis (WPA).

For more information, see the module wdwf.py.

Inputs:
  signal: A 1-D numpy array that represents the samples of the signal in the time domain.
  max_level: Maximum allowed level of decomposition. Usually, max_level = 6.

Outputs:
  new_signal: A 1-D numpy array that represents the samples of the zero-extended signal in the time domain."""

	new_length = find_the_appropriate_signal_length(signal.shape[0], max_level)
	new_signal = list(signal)

	if new_length != len(new_signal):
		length_difference = new_length - len(new_signal)
		[new_signal.append(0) for i in range(length_difference)]

	new_signal = numpy.array(new_signal, dtype = signal.dtype)

	return new_signal