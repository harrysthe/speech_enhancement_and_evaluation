import numpy

from fix_wpa_bug import signal_adaptation_for_22_bands_wpa
from pywt import Wavelet, WaveletPacket, dwt_max_level, wavedec, waverec
from scipy.io import wavfile

def wpa_implementation_22_bands(signal, wavelet_name, wavelet_mode, max_level):

	"""Implements a Wavelet Packet Analysis (WPA) of 22 bands.

The 22 bands of the WPA try to match the Bark scale as much as possible.
The highest frequency of the signal is assumed to be 8000 Hz.

Inputs:
  signal: A 1-D numpy array.
  wavelet_name: Wavelet to use in the transform. Usually, wavelet_name = "db6".
  wavelet_mode: Signal extension mode for decomposition. Usually, wavelet_mode = "per".
  max_level: Maximum allowed level of decomposition. Usually, max_level = 6.

Outputs:
  wavelet_coefficients: A list with so many nested 1-D numpy arrays as the number of bands, namely 22.
  Each array contains the wavelet coefficients of the signal at the corresponding band.
  The first array corresponds to the lowest band."""

	wavelet_coefficients = []
	wp = WaveletPacket(signal, wavelet_name, wavelet_mode, max_level)

	wavelet_coefficients.append(wp['aaaaaa'].data) # 0 - 125 Hz.
	wavelet_coefficients.append(wp['aaaaad'].data) # 125 - 250 Hz.
	wavelet_coefficients.append(wp['aaaada'].data) # 250 - 375 Hz.
	wavelet_coefficients.append(wp['aaaadd'].data) # 375 - 500 Hz.
	wavelet_coefficients.append(wp['aaadaa'].data) # 500 - 625 Hz.
	wavelet_coefficients.append(wp['aaadad'].data) # 625 - 750 Hz.
	wavelet_coefficients.append(wp['aaadda'].data) # 750 - 875 Hz.
	wavelet_coefficients.append(wp['aaaddd'].data) # 875 - 1000 Hz.
	wavelet_coefficients.append(wp['aadaaa'].data) # 1000 - 1125 Hz.
	wavelet_coefficients.append(wp['aadaad'].data) # 1125 - 1250 Hz.

	wavelet_coefficients.append(wp['aadad'].data) # 1250 - 1500 Hz.
	wavelet_coefficients.append(wp['aadda'].data) # 1500 - 1750 Hz.
	wavelet_coefficients.append(wp['aaddd'].data) # 1750 - 2000 Hz.
	wavelet_coefficients.append(wp['adaaa'].data) # 2000 - 2250 Hz.
	wavelet_coefficients.append(wp['adaad'].data) # 2250 - 2500 Hz.

	wavelet_coefficients.append(wp['adad'].data) # 2500 - 3000 Hz.
	wavelet_coefficients.append(wp['adda'].data) # 3000 - 3500 Hz.
	wavelet_coefficients.append(wp['addd'].data) # 3500 - 4000 Hz.

	wavelet_coefficients.append(wp['daa'].data) # 4000 - 5000 Hz.
	wavelet_coefficients.append(wp['dad'].data) # 5000 - 6000 Hz.
	wavelet_coefficients.append(wp['dda'].data) # 6000 - 7000 Hz.
	wavelet_coefficients.append(wp['ddd'].data) # 7000 - 8000 Hz.

	return wavelet_coefficients

def bands_frequencies_computation(sampling_rate, number_of_bands, is_dwt):

	"""Computation of the lowest, highest and center frequencies of each band.

The function takes into consideration any Discrete Wavelet Transform (DWT) implementation.
The function takes into consideration the 22-bands Wavelet Packet Analysis (WPA).
Highest frequency taken into consideration = sampling_rate / 2.0 Hz.
Lowest frequency taken into consideration = 0.0 Hz.
The center frequency of each band is the geometric mean of the corresponding lowest and highest frequencies.
However, as far as the lowest band is concerned, the center frequency is the arithmetic mean.

Inputs:
  sampling_rate: Usually, sampling_rate = 16000 Hz. If the WPA is chosen, sampling_rate = 16000 Hz definitely.
  number_of_bands: If any DWT is chosen, number_of_bands = dwt_level + 1. Otherwise, number_of_bands = 22.
  is_dwt: True if a DWT is being used. False if the 22-bands WPA is being used.

Outputs:
  bands_frequencies: A list with 3 nested lists. One for the lowest frequencies, one for the highest ones and one for center ones.
  The first element of each of the 3 lists corresponds to the lowest band."""

	min_frequency = 0.0
	max_frequency = sampling_rate / 2.0
	low_frequencies, center_frequencies, high_frequencies = [], [], []

	if is_dwt:
		for nb in range(number_of_bands):
			high_frequency = max_frequency
			high_frequencies.append(high_frequency)

			if nb < (number_of_bands - 1):
				low_frequency = max_frequency / 2.0
				center_frequency = (high_frequency * low_frequency) ** 0.5 # Geometric mean.

			else:
				low_frequency = min_frequency
				center_frequency = (high_frequency + low_frequency) / 2.0 # Arithmetic mean.

			low_frequencies.append(low_frequency)
			center_frequencies.append(center_frequency)
			max_frequency = max_frequency / 2.0

		# The list.reverse() method is used in order to align the lowest band with the lowest index.
		high_frequencies.reverse()
		low_frequencies.reverse()
		center_frequencies.reverse()

	else:
		low_frequencies.append(min_frequency)
		frequency_range_constant = 125.0 # This constant is acceptable because I have to do with the 22-bands WPA.
		high_frequencies.append(frequency_range_constant)
		center_frequency = (high_frequencies[0] + low_frequencies[0]) / 2.0 # Arithmetic mean.
		center_frequencies.append(center_frequency)

		for nb in range(1, number_of_bands):
			low_frequency = high_frequencies[nb - 1]

			if nb <= 9:
				high_frequency = low_frequency + frequency_range_constant

			elif nb <= 14:
				high_frequency = low_frequency + (2 * frequency_range_constant)

			elif nb <= 17:
				high_frequency = low_frequency + (4 * frequency_range_constant)

			else:
				high_frequency = low_frequency + (8 * frequency_range_constant)

			center_frequency = (high_frequency * low_frequency) ** 0.5 # Geometric mean.
			low_frequencies.append(low_frequency)
			center_frequencies.append(center_frequency)
			high_frequencies.append(high_frequency)

	bands_frequencies = [low_frequencies, high_frequencies, center_frequencies]

	return bands_frequencies

def perceptual_filter_computation(center_frequencies, in_use):

	"""Computes a perceptual filter that models the human auditory system's filtering of broad-band acoustic noise.

Inputs:
  center_frequencies: A list with the center frequencies of each band, starting from the lowest band.

  in_use: True if the filtering of the perceptual filter takes place.
  Otherwise, all values of the perceptual filter are equal to 1.0.

Outputs:
  filter_values: A list with so many elements as the number of bands of some band analysis.
  Each element represents the value of the perceptual filter at the corresponding band, starting from the lowest one."""

	filter_boundary = 500.0

	if in_use:
		filter_values = [1.0 if center_frequency <= filter_boundary \
						 else center_frequency / filter_boundary for center_frequency in center_frequencies]

	else:
		filter_values = [1.0 for center_frequency in center_frequencies]

	return filter_values

def estimation_of_the_signal_power_in_wavelet_domain(signal_wavelet_coefficients, wdwf_id, a):

	"""Estimates the signal's power at each band in the wavelet domain.

Inputs:
  signal_wavelet_coefficients: A list with so many nested 1-D numpy arrays as the number of bands of some band analysis.
  Each array contains the wavelet coefficients of the signal at the corresponding band.
  The first array corresponds to the lowest band.

  wdwf_id: Determines the Wavelet Domain Wiener Filter (WDWF) that is being used. More in denoising().
  If wdwf_id = 1, the WDWF_0 is being used.
  If wdwf_id = 2, the WDWF_I is being used.
  If wdwf_id = 3, the WDWF_I&II is being used.
  If wdwf_id != 1, 2, 3, the WDWF_II is being used.

  a: Parameter of the WDWFs. Usually, a = 2.0. More in denoising(). If wdwf_id = 1, then a is redundant.

Outputs:
  p_signal_band_level: A list with so many elements as the number of bands of the band analysis.
  Each element represents the power of the signal at the corresponding band."""

	p_signal_band_level = []

	for band_wavelet_coefficients in signal_wavelet_coefficients:
		number_of_coefficients = len(band_wavelet_coefficients)
		p_signal_sum = 0.0

		for wavelet_coefficient_value in band_wavelet_coefficients:
			if wdwf_id != 1:
				p_signal_sum = p_signal_sum + (abs(wavelet_coefficient_value) ** a)

			else:
				p_signal_sum = p_signal_sum + (abs(wavelet_coefficient_value) ** 2.0)

		if wdwf_id == 1:
			p_signal_sum = p_signal_sum ** 0.5

		p_signal_value = p_signal_sum / number_of_coefficients
		p_signal_band_level.append(p_signal_value)

	return p_signal_band_level

def a_powered_signal_magnitude_ema_filtering(signal_wavelet_coefficients, a, d):

	"""Filters the a-powered signal's magnitude using an Exponential Moving Average (EMA) scheme.

Inputs:
  signal_wavelet_coefficients: A list with so many nested 1-D numpy arrays as the number of bands of some band analysis.
  Each array contains the wavelet coefficients of the signal at the corresponding band.
  The first array corresponds to the lowest band.

  a: Parameter of the Wavelet Domain Filters (WDWFs). Usually, a = 2.0. More in denoising().

  d: Parameter of the WDWFs which determines the parameters of the EMA procedure.
  Its value depends on the highest frequency of the signal. If highest_frequency = 8000 Hz, usually d = 0.22. More in denoising().

Outputs:
  p_signal_band_level: A list with so many nested lists as the number of bands of the band analysis.
  Each list contains so many elements as the number of the wavelet coefficients of the signal at the corresponding band.
  Each element represents the a-powered signal's magnitude at the corresponding wavelet coefficient."""

	p_signal_band_level = []

	for signal_band_wavelet_coefficients in signal_wavelet_coefficients:
		p_signal_point_level = []
		number_of_coefficients = len(signal_band_wavelet_coefficients)

		for nc in range(number_of_coefficients):
			a_powered_signal_magnitude = abs(signal_band_wavelet_coefficients[nc]) ** a

			if nc == 0:
				p_signal_value = a_powered_signal_magnitude

			else:
				p_signal_value = (d * a_powered_signal_magnitude) + ((1.0 - d) * p_signal_point_level[nc - 1])

			p_signal_point_level.append(p_signal_value)
		p_signal_band_level.append(p_signal_point_level)

	return p_signal_band_level

def wdwf_implementation_1(p_noisy_signal_band_level, p_noise_footprint_band_level, perceptual_filter_band_level, a, b, c):

	"""Implements the first Wavelet Domain Wiener Filter (WDWF_0). More in denoising().

Inputs:
  p_noisy_signal_band_level: A list with so many elements as the number of bands of some band analysis.
  Each element represents the power of the noisy signal at the corresponding band.

  p_noise_footprint_band_level: A list with so many elements as the number of bands of the band analysis.
  Each element represents the power of the noise footprint at the corresponding band.

  perceptual_filter_band_level: A list with the values of the perceptual filter at each band of the band analysis.
  The perceptual filter models the human auditory system's filtering of broad-band acoustic noise.

  a: Parameter of the WDWF_0. Usually, a = 4.0. More in denoising().
  b: Parameter of the WDWF_0. Usually, b = 0.5. More in denoising().
  c: Paremeter of the WDWF_0. Usually, c = 0.25. More in denoising().

Outputs:
  wdwf_1_band_level: A list with so many elements as the number of bands of the band analysis.
  Each element represents the value of the WDWF_0 at the corresponding band."""

	wdwf_1_band_level = []

	for p_noisy_signal_band, p_noise_footprint_band, perceptual_filter_band in \
	zip(p_noisy_signal_band_level, p_noise_footprint_band_level, perceptual_filter_band_level):

		if p_noisy_signal_band == 0.0:
			wdwf_1_value = 0.0

		else:
			ratio = (p_noise_footprint_band / p_noisy_signal_band) ** a
			decision_filter_value = c * perceptual_filter_band * ratio

			if decision_filter_value <= 1.0:
				wdwf_1_value = (1.0 - decision_filter_value) ** b

			else:
				wdwf_1_value = 0.0

		wdwf_1_band_level.append(wdwf_1_value)

	return wdwf_1_band_level

def wdwf_implementation_2(p_noisy_signal_band_level, p_noise_footprint_band_level, perceptual_filter_band_level, b, c):

	"""Implements the second Wavelet Domain Wiener Filter (WDWF_I). More in denoising().

Inputs:
  p_noisy_signal_band_level: A list with so many nested lists as the number of bands of some band analysis.
  Each list contains so many elements as the number of the wavelet coefficients of the noisy signal at the corresponding band.
  Each element represents the a-powered noisy signal's magnitude at the corresponding wavelet coefficient.

  p_noise_footprint_band_level: A list with so many elements as the number of bands of the band analysis.
  Each element represents the power of the noise footprint at the corresponding band.

  perceptual_filter_band_level: A list with the values of the perceptual filter at each band of the band analysis.
  The perceptual filter models the human auditory system's filtering of broad-band acoustic noise.

  b: Parameter of the WDWF_I. Usually, b = 2.0. More in denoising().
  c: Parameter of the WDWF_I. Usually, c = 1.0. More in denoising().

Outputs:
  wdwf_2_band_level: A list with so many nested lists as the number of bands of the band analysis.
  Each list contains so many elements as the number of the wavelet coefficients of the WDWF_I at the corresponding band.
  Each element represents the value of the WDWF_I at a specific band and wavelet coefficient."""

	wdwf_2_band_level = []

	for p_noisy_signal_band, p_noise_footprint_band, perceptual_filter_band in \
	zip(p_noisy_signal_band_level, p_noise_footprint_band_level, perceptual_filter_band_level):

		wdwf_2_band_wavelet_coefficients = []

		for p_noisy_signal_value in p_noisy_signal_band:
			if p_noisy_signal_value == 0.0:
				wdwf_2_wavelet_coefficient_value = 0.0

			else:
				ratio = p_noise_footprint_band / p_noisy_signal_value
				decision_filter_value = c * perceptual_filter_band * ratio

				if decision_filter_value <= 1.0:
					wdwf_2_wavelet_coefficient_value = (1.0 - decision_filter_value) ** b

				else:
					wdwf_2_wavelet_coefficient_value = 0.0

			wdwf_2_band_wavelet_coefficients.append(wdwf_2_wavelet_coefficient_value)
		wdwf_2_band_level.append(wdwf_2_band_wavelet_coefficients)

	return wdwf_2_band_level

def wdwf_implementation_final(p_noisy_signal_value_1, p_noisy_signal_value_2, p_noise_footprint_value, \
							  perceptual_filter_value, b, c):

	"""Implements the third (WDWF_I&II) and the fourth (WDWF_II) Wavelet Domain Wiener Filter.

Inputs:
  p_noisy_signal_value_1: An a-powered magnitude of the noisy signal, which refers to a wavelet coefficient at some band.
  The denoising of the WDWF_I&II is based on this value. More in denoising().

  p_noisy_signal_value_2: An a-powered magnitude of the noisy signal, which refers to a wavelet coefficient at some band.
  Both WDWFs are based on this value to decide if a denoising will take place or not at the specific wavelet coefficient.
  If denoising will not take place, this means that the noise at the specific wavelet coefficient is too powerful
  and the filter equals to 0.0. Furthermore, the denoising of the WDWF_II is based on this value. More in denoising().

  p_noise_footprint_value: A power value of the noise footprint, which refers to some band.

  perceptual_filter_value: A value of the perceptual filter that models the human auditory system's filtering of
  broad-band acoustic noise. The value corresponds to some band.

  b: Parameter of the WDWF_I&II and WDWF_II. Usually, b = 2.0. More in denoising().
  c: Parameter of the WDWF_I&II and WDWF_II. Usually, c = 1.0. More in denoising().

Outputs:
  wdwf_final_value: A value that corresponds either to WDWF_I&II or to the WDWF_II.
  This value represents the value of a wavelet coefficient at some band."""

	ratio_flag = False
	decision_filter_value = (c * perceptual_filter_value * p_noise_footprint_value) - p_noisy_signal_value_2

	if decision_filter_value > 0.0:
		wdwf_final_value = 0.0

	else:
		if p_noisy_signal_value_1 >= 0.0: # WDWF_I&II.
			if p_noisy_signal_value_1 == 0.0:
				wdwf_final_value = 0.0

			else:
				ratio = p_noise_footprint_value / p_noisy_signal_value_1
				ratio_flag = True

		else: # WDWF_II.
			if p_noisy_signal_value_2 == 0.0:
				wdwf_final_value = 0.0

			else:
				ratio = p_noise_footprint_value / p_noisy_signal_value_2
				ratio_flag = True

		if ratio_flag == True:
			if (c * perceptual_filter_value * ratio) <= 1.0:
				wdwf_final_value = (1.0 - (c * perceptual_filter_value * ratio)) ** b

			else:
				wdwf_final_value = 0.0

	return wdwf_final_value

def denoised_signal_wavelet_coefficients_computation_1(noisy_signal_wavelet_coefficients, wdwf_band_level, wdwf_id):

	"""Computes the wavelet coefficients of the denoised signal when the first (WDWF_0) or the second (WDWF_I)
Wavelet Domain Wiener Filter is being used.

Inputs:
  noisy_signal_wavelet_coefficients: A list with so many nested 1-D numpy arrays as the number of bands of some band analysis.
  Each array represents the wavelet coefficients of the noisy signal at the corresponding band.
  The first array corresponds to the lowest band.

  wdwf_band_level: A list with so many elements as the number of bands of the band analysis. If the WDWF_0 is being used, each element
  represents the value of the filter at the corresponding band. If the WDWF_I is being used, each element is a list with so many
  elements as the number of the wavelet coefficients at the corresponding band.

  wdwf_id: Determines the WDWF (WDWF_0 or WDWF_I) that is being used. More in denoising().
  If wdwf_id = 1, the WDWF_0 is being used. If wdwf_id != 1, the WDWF_I is being used.

Outputs:
  denoised_signal_wavelet_coefficients: A list with so many nested lists as the number of bands of the band analysis.
  Each list represents the wavelet coefficients of the denoised signal at the corresponding band.
  The first list corresponds to the lowest band."""

	denoised_signal_wavelet_coefficients = []

	for noisy_signal_band_wavelet_coefficients, wdwf_band in zip(noisy_signal_wavelet_coefficients, wdwf_band_level):
		denoised_signal_band_wavelet_coefficients = []
		number_of_coefficients = len(noisy_signal_band_wavelet_coefficients)

		for nc in range(number_of_coefficients):
			if wdwf_id != 1:
				denoised_signal_wavelet_coefficient_value = noisy_signal_band_wavelet_coefficients[nc] * wdwf_band[nc]

			else:
				denoised_signal_wavelet_coefficient_value = noisy_signal_band_wavelet_coefficients[nc] * wdwf_band

			denoised_signal_band_wavelet_coefficients.append(denoised_signal_wavelet_coefficient_value)
		denoised_signal_wavelet_coefficients.append(denoised_signal_band_wavelet_coefficients)

	return denoised_signal_wavelet_coefficients

def denoised_signal_wavelet_coefficients_computation_2(noisy_signal_wavelet_coefficients, p_noise_footprint_band_level, \
													   perceptual_filter_band_level, number_of_bands, wdwf_id, a, b, c, d, dps):

	"""Computes the wavelet coefficients of the denoised signal when the third (WDWF_I&II) or the fourth (WDWF_II)
Wavelet Domain Wiener Filter is being used.

Inputs:
  noisy_signal_wavelet_coefficients: A list with so many nested 1-D numpy arrays as the number_of_bands.
  Each array represents the wavelet coefficients of the noisy signal at the corresponding band.
  The first array corresponds to the lowest band.

  p_noise_footprint_band_level: A list with so many elements as the number_of_bands.
  Each element represents the power of the noise footprint at the corresponding band.

  perceptual_filter_band_level: A list with so many elements as the number_of_bands.
  Each element represents the value of the perceptual filter at the corresponding band.
  The perceptual filter models the human auditory system's filtering of broad-band acoustic noise.

  number_of_bands: The number of bands at which the noisy signal and the noise footprint have splitted as a result of the
  wavelet analysis that is being used.

  wdwf_id: Determines the WDWF (WDWF_I&II or WDWF_II) that is being used. More in denoising().
  If wdwf_id = 3, the WDWF_I&II is being used. Otherwise, the WDWF_II is being used.

  a: Parameter of the WDWF_I&II and WDWF_II. Usually, a = 2.0. More in denoising().
  b: Parameter of the WDWF_I&II and WDWF_II. Usually, b = 2.0. More in denoising().
  c: Parameter of the WDWF_I&II and WDWF_II. Usually, c = 1.0. More in denoising().

  d: Parameter of the WDWF_I&II and WDWF_II. Its value depends on the highest frequency of the noisy signal.
  For highest_frequency = 8000 Hz, usually d = 0.22. More in denoising().

  dps: Parameter of the WDWF_I&II and WDWF_II. Usually, dps = 0.9. More in denoising().

Outputs:
  denoised_signal_wavelet_coefficients: A list with so many nested lists as the number_of_bands.
  Each list represents the wavelet coefficients of the denoised signal at the corresponding band.
  The first list corresponds to the lowest band."""

	denoised_signal_wavelet_coefficients = []

	if wdwf_id == 3:
		p_noisy_signal_band_level_id_3 = a_powered_signal_magnitude_ema_filtering(noisy_signal_wavelet_coefficients, a, d)
		# Each value of the p_noisy_signal_band_level_id_3 is >= 0.

	for nb in range(number_of_bands):
		denoised_signal_band_wavelet_coefficients, p_denoised_signal_point_level = [], []
		number_of_coefficients = len(noisy_signal_wavelet_coefficients[nb])

		for nc in range(number_of_coefficients):
			a_powered_noisy_signal_magnitude = abs(noisy_signal_wavelet_coefficients[nb][nc]) ** a

			if nc == 0:
				p_noisy_signal_value = a_powered_noisy_signal_magnitude

			else:
				p_noisy_signal_value = (d * a_powered_noisy_signal_magnitude) + ((1.0 - d) * p_denoised_signal_point_level[nc - 1])

			if wdwf_id == 3:
				wdwf_value = wdwf_implementation_final(p_noisy_signal_band_level_id_3[nb][nc], p_noisy_signal_value, \
													   p_noise_footprint_band_level[nb], perceptual_filter_band_level[nb], b, c)

			else:
				wdwf_value = wdwf_implementation_final(-1.0, p_noisy_signal_value, p_noise_footprint_band_level[nb], \
													   perceptual_filter_band_level[nb], b, c)

			denoised_signal_wavelet_coefficient_value = noisy_signal_wavelet_coefficients[nb][nc] * wdwf_value
			denoised_signal_band_wavelet_coefficients.append(denoised_signal_wavelet_coefficient_value)
			a_powered_denoised_signal_magnitude = abs(denoised_signal_wavelet_coefficient_value) ** a

			if nc == 0:
				p_denoised_signal_value = a_powered_denoised_signal_magnitude

			else:
				p_denoised_signal_value = (dps * a_powered_denoised_signal_magnitude) \
										+ ((1.0 - dps) * p_denoised_signal_point_level[nc - 1])

			p_denoised_signal_point_level.append(p_denoised_signal_value)
		denoised_signal_wavelet_coefficients.append(denoised_signal_band_wavelet_coefficients)

	return denoised_signal_wavelet_coefficients

def from_wavelet_domain_to_time_domain(signal_wavelet_coefficients, is_dwt, wavelet_name, wavelet_mode, max_level):

	"""Transforms the signal from the wavelet domain to the time domain.

The highest frequency of the signal is assumed to be 8000 Hz.

The function takes into consideration any Discrete Wavelet Transform (DWT) implementation. However, 6-level or 7-level DWTs are
the best choices. In addition, the function takes into consideration the 22-bands Wavelet Packet Analysis (WPA).

Inputs:
  signal_wavelet_coefficients: A list with so many nested lists as the number of bands of some band analysis.
  Each list contains the wavelet coefficients of the signal at the corresponding band. The first list corresponds to the lowest band.

  is_dwt: True if a DWT is being used (6-level or 7-level). False if the 22-bands WPA is being used.
  wavelet_name: Wavelet to use in the transform. Usually, wavelet_name = "db6".
  wavelet_mode: Signal extension mode for reconstruction. Usually, wavelet_mode = "per".
  max_level: Maximum allowed level of decomposition for the 22-bands WPA case. Usually, max_level = 6.

Outputs:
  signal_time_domain: A list that contains the samples of the signal in the time domain."""

	signal_time_domain = []

	if is_dwt == True:
		array_signal_wavelet_coefficients = []

		if len(signal_wavelet_coefficients) == 8: # This means that I have to do with a 7-level DWT implementation.
			number_of_approximation_coefficients = len(signal_wavelet_coefficients[0])

			for nac in range(number_of_approximation_coefficients):
				signal_wavelet_coefficients[0][nac] = 0.0 # The approximation coefficients correspond to 0 - 62.5 Hz.
														  # In this frequency range, theoretically, I have no speech.

		for band_wavelet_coefficients in signal_wavelet_coefficients:
			array_band_wavelet_coefficients = numpy.array(band_wavelet_coefficients, dtype = numpy.float64)
			array_signal_wavelet_coefficients.append(array_band_wavelet_coefficients)

		signal_samples = waverec(array_signal_wavelet_coefficients, wavelet_name, wavelet_mode)
		signal_time_domain.extend(signal_samples)

	else:
		wp = WaveletPacket(None, wavelet_name, wavelet_mode, max_level)

		wp['aaaaaa'] = numpy.array(signal_wavelet_coefficients[0], dtype = numpy.float64) # 0 - 125 Hz.
		wp['aaaaad'] = numpy.array(signal_wavelet_coefficients[1], dtype = numpy.float64) # 125 - 250 Hz.
		wp['aaaada'] = numpy.array(signal_wavelet_coefficients[2], dtype = numpy.float64) # 250 - 375 Hz.
		wp['aaaadd'] = numpy.array(signal_wavelet_coefficients[3], dtype = numpy.float64) # 375 - 500 Hz.
		wp['aaadaa'] = numpy.array(signal_wavelet_coefficients[4], dtype = numpy.float64) # 500 - 625 Hz.
		wp['aaadad'] = numpy.array(signal_wavelet_coefficients[5], dtype = numpy.float64) # 625 - 750 Hz.
		wp['aaadda'] = numpy.array(signal_wavelet_coefficients[6], dtype = numpy.float64) # 750 - 875 Hz.
		wp['aaaddd'] = numpy.array(signal_wavelet_coefficients[7], dtype = numpy.float64) # 875 - 1000 Hz.
		wp['aadaaa'] = numpy.array(signal_wavelet_coefficients[8], dtype = numpy.float64) # 1000 - 1125 Hz.
		wp['aadaad'] = numpy.array(signal_wavelet_coefficients[9], dtype = numpy.float64) # 1125 - 1250 Hz.

		wp['aadad'] = numpy.array(signal_wavelet_coefficients[10], dtype = numpy.float64) # 1250 - 1500 Hz.
		wp['aadda'] = numpy.array(signal_wavelet_coefficients[11], dtype = numpy.float64) # 1500 - 1750 Hz.
		wp['aaddd'] = numpy.array(signal_wavelet_coefficients[12], dtype = numpy.float64) # 1750 - 2000 Hz.
		wp['adaaa'] = numpy.array(signal_wavelet_coefficients[13], dtype = numpy.float64) # 2000 - 2250 Hz.
		wp['adaad'] = numpy.array(signal_wavelet_coefficients[14], dtype = numpy.float64) # 2250 - 2500 Hz.

		wp['adad'] = numpy.array(signal_wavelet_coefficients[15], dtype = numpy.float64) # 2500 - 3000 Hz.
		wp['adda'] = numpy.array(signal_wavelet_coefficients[16], dtype = numpy.float64) # 3000 - 3500 Hz.
		wp['addd'] = numpy.array(signal_wavelet_coefficients[17], dtype = numpy.float64) # 3500 - 4000 Hz.

		wp['daa'] = numpy.array(signal_wavelet_coefficients[18], dtype = numpy.float64) # 4000 - 5000 Hz.
		wp['dad'] = numpy.array(signal_wavelet_coefficients[19], dtype = numpy.float64) # 5000 - 6000 Hz.
		wp['dda'] = numpy.array(signal_wavelet_coefficients[20], dtype = numpy.float64) # 6000 - 7000 Hz.
		wp['ddd'] = numpy.array(signal_wavelet_coefficients[21], dtype = numpy.float64) # 7000 - 8000 Hz.

		signal_samples = wp.reconstruct()
		signal_time_domain.extend(signal_samples)

	return signal_time_domain

def denoising(noisy_signal_wav_file, noise_footprint_wav_file, wdwf_id, in_use, is_dwt, dwt_level, a, b, c, d, dps):

	"""Implements a single-channel denoising process using a Wavelet Domain Wiener Filter (WDWF) specified by the wdwf_id parameter.

The function has been designed based on the assumptions that the highest frequency of the noisy signal is 8000 Hz and
the sampling rate is 16000 Hz. As a result, a 22-bands Wavelet Packet Analysis (WPA) is one of the wavelet analysis schemes
that can be taken into consideration. On the other hand, any Discrete Wavelet Transform (DWT) can be supported from the current
function, but 6-level and 7-level DWTs are the best choices given the above assumptions. Consequently, a call of the function with
different conditions from the ones specified by the assumptions may be valid but wrong. So, a call like that must be done very
carefully.

The noisy_signal_wav_file and the noise_footprint_wav_file must have the same sampling rate.
The noisy_signal_wav_file and the noise_footprint_wav_file must be coded in the same way.
The noisy_signal_wav_file and the noise_footprint_wav_file must not be zero-signals.
The noisy_signal_wav_file and the noise_footprint_wav_file must have enough samples to undergo the selected wavelet analysis.

One single-channel wav file, that represents the denoised signal, is created.
The sampling rate of the denoised_signal_wav_file is the same with the sampling rate of the noisy_signal_wav_file.
The denoised_signal_wav_file is coded in the same way as the noisy_signal_wav_file.

Inputs:
  noisy_signal_wav_file: A wav file that represents the single-channel noisy signal.
  The wav file may be at 16-bit PCM format, but this is not obligatory.

  noise_footprint_wav_file: A wav file that represents the single-channel noise footprint.
  The wav file may be at 16-bit PCM format, but this is not obligatory.

  wdwf_id: Determines which WDWF will be used. The WDWFs are described in [1] and they are 4.
  If wdwf_id = 1, the WDWF_0 will be used.
  If wdwf_id = 2, the WDWF_I will be used.
  If wdwf_id = 3, the WDWF_I&II will be used.
  If wdwf_id != 1, 2, 3, the WDWF_II will be used.

  in_use: True if the perceptual filter for broad-band acoustic noise, presented in [1], will be used.
  is_dwt: True if a DWT will be used (6-level or 7-level). False if the 22-bands WPA will be used.
  dwt_level: Defines the level of the DWT (6 or 7) that will be used. If the WPA is selected, dwt_level = 6.
  a: Parameter of the WDWFs. If wdwf_id = 1, usually a = 4.0. Otherwise, usually a = 2.0. More in [1].
  b: Parameter of the WDWFs. If wdwf_id = 1, usually b = 0.5. Otherwise, usually b = 2.0. More in [1].
  c: Parameter of the WDWFs. If wdwf_id = 1, usually c = 0.25. Otherwise, usually c = 1.0. More in [1].

  d: Parameter of the WDWFs. Its value depends on the highest frequency of the noisy signal.
  For highest_frequency = 8000 Hz, usually d = 0.22. If wdwf_id = 1, d is redundant. More in [1].

  dps: Parameter of the WDWFs. Usually, dps = 0.9. If wdwf_id = 1 or wdwf_id = 2, dps is redundant. More in [1].

Outputs:
  denoised_signal: A 1-D numpy array that represents the final denoised signal.
  It will be None if an unpleasant situation come to surface. In this case, no wav file is created.

[1] Dimoulas, C., Kalliris, G., Papanikolaou, G., and Kalampakas, A. (2006). Novel wavelet domain Wiener filtering de-noising
techniques: Application to bowel sounds captured by means of abdominal surface vibrations.
Biomedical Signal Processing and Control, volume 1, issue 3, pages 177-218, July 2006."""

	wavelet_name, wavelet_mode = 'db6', 'per'
	temp_wavelet = Wavelet(wavelet_name)

	noisy_signal_sampling_rate, noisy_signal = wavfile.read(noisy_signal_wav_file)
	noise_footprint_sampling_rate, noise_footprint = wavfile.read(noise_footprint_wav_file)

	noisy_signal_max_level = dwt_max_level(noisy_signal.shape[0], temp_wavelet)
	noise_footprint_max_level = dwt_max_level(noise_footprint.shape[0], temp_wavelet)

	# Checking if the noisy_signal_wav_file and the noise_footprint_wav_file are single-channel. -------------------------------------
	if (len(noisy_signal.shape) != 1) or (len(noise_footprint.shape) != 1):
		print("")
		print("wdwf.denoising():")
		print("The " + noisy_signal_wav_file + " or/and the " + noise_footprint_wav_file + " are not single-channel!")
		print("")
		return
	# --------------------------------------------------------------------------------------------------------------------------------

	# Checking if the noisy_signal_wav_file and the noise_footprint_wav_file have the same sampling rate. ----------------------------
	if noisy_signal_sampling_rate != noise_footprint_sampling_rate:
		print("")
		print("wdwf.denoising():")
		print("The " + noisy_signal_wav_file + " and the " + noise_footprint_wav_file + " have not the same sampling rate!")
		print("")
		return

	else:
		sampling_rate = noisy_signal_sampling_rate

		if sampling_rate != 16000:
			print("")
			print("wdwf.denoising():")
			print("The sampling rate of the " + noisy_signal_wav_file + " and of the " + noise_footprint_wav_file + " is not 16 kHz.")
	# --------------------------------------------------------------------------------------------------------------------------------

	# Checking if the noisy_signal_wav_file and the noise_footprint_wav_file are coded in the same way. ------------------------------
	if noisy_signal.dtype != noise_footprint.dtype:
		print("")
		print("wdwf.denoising():")
		print("The " + noisy_signal_wav_file + " and the " + noise_footprint_wav_file + " are not coded in the same way!")
		print("")
		return

	else:
		encoding_type = noisy_signal.dtype

		if encoding_type != numpy.int16:
			print("")
			print("wdwf.denoising():")
			print("The " + noisy_signal_wav_file + " and the " + noise_footprint_wav_file + " are not at 16-bit PCM format.")
	# --------------------------------------------------------------------------------------------------------------------------------

	# Checking if the noisy_signal_wav_file or/and the noise_footprint_wav_file are zero-signals. ------------------------------------
	noisy_signal_list = list(noisy_signal)
	noise_footprint_list = list(noise_footprint)

	if (noisy_signal_list.count(0) == len(noisy_signal_list)) or (noise_footprint_list.count(0) == len(noise_footprint_list)):
		print("")
		print("wdwf.denoising():")
		print("The " + noisy_signal_wav_file + " or/and the " + noise_footprint_wav_file + " are zero-signals!")
		print("")
		return
	# --------------------------------------------------------------------------------------------------------------------------------

	if is_dwt:
		# Checking if the noisy_signal_wav_file or/and the noise_footprint_wav_file have enough samples for the selected DWT. --------
		if (noisy_signal_max_level < dwt_level) or (noise_footprint_max_level < dwt_level):
			print("")
			print("wdwf.denoising():")

			string_1 = "The " + noisy_signal_wav_file + " or/and the " + noise_footprint_wav_file
			string_2 = " have not enough samples for the selected DWT!"
			print(string_1 + string_2)

			print("")
			return
		# ----------------------------------------------------------------------------------------------------------------------------

		noisy_signal_wc = wavedec(noisy_signal, wavelet_name, wavelet_mode, dwt_level)
		noise_footprint_wc = wavedec(noise_footprint, wavelet_name, wavelet_mode, dwt_level)

	else:
		# Checking if the noisy_signal_wav_file or/and the noise_footprint_wav_file have enough samples for the 22-bands WPA. --------
		if (noisy_signal_max_level < dwt_level) or (noise_footprint_max_level < dwt_level):
			print("")
			print("wdwf.denoising():")

			string_1 = "The " + noisy_signal_wav_file + " or/and the " + noise_footprint_wav_file
			string_2 = " have not enough samples for the 22-bands WPA!"
			print(string_1 + string_2)

			print("")
			return
		# ----------------------------------------------------------------------------------------------------------------------------

		new_noisy_signal = signal_adaptation_for_22_bands_wpa(noisy_signal, dwt_level)
		noisy_signal_wc = wpa_implementation_22_bands(new_noisy_signal, wavelet_name, wavelet_mode, dwt_level)
		new_noise_footprint = signal_adaptation_for_22_bands_wpa(noise_footprint, dwt_level)
		noise_footprint_wc = wpa_implementation_22_bands(new_noise_footprint, wavelet_name, wavelet_mode, dwt_level)

	number_of_bands = len(noisy_signal_wc)
	bands_frequencies = bands_frequencies_computation(sampling_rate, number_of_bands, is_dwt)
	center_frequencies = bands_frequencies[-1]
	perceptual_filter_band_level = perceptual_filter_computation(center_frequencies, in_use)
	p_noise_footprint_band_level = estimation_of_the_signal_power_in_wavelet_domain(noise_footprint_wc, wdwf_id, a)

	if wdwf_id == 1:
		p_noisy_signal_band_level = estimation_of_the_signal_power_in_wavelet_domain(noisy_signal_wc, 1, a)

		wdwf_band_level = wdwf_implementation_1(p_noisy_signal_band_level, p_noise_footprint_band_level, \
												perceptual_filter_band_level, a, b, c)

		denoised_signal_wc = denoised_signal_wavelet_coefficients_computation_1(noisy_signal_wc, wdwf_band_level, 1)

	elif wdwf_id == 2:
		p_noisy_signal_band_level = a_powered_signal_magnitude_ema_filtering(noisy_signal_wc, a, d)

		wdwf_band_level = wdwf_implementation_2(p_noisy_signal_band_level, p_noise_footprint_band_level, \
												perceptual_filter_band_level, b, c)

		denoised_signal_wc = denoised_signal_wavelet_coefficients_computation_1(noisy_signal_wc, wdwf_band_level, 2)

	else:
		denoised_signal_wc = denoised_signal_wavelet_coefficients_computation_2(noisy_signal_wc, p_noise_footprint_band_level, \
																				perceptual_filter_band_level, number_of_bands, \
																				wdwf_id, a, b, c, d, dps)

	denoised_signal = from_wavelet_domain_to_time_domain(denoised_signal_wc, is_dwt, wavelet_name, wavelet_mode, dwt_level)
	denoised_signal = numpy.array(denoised_signal, dtype = encoding_type)
	wavfile.write(noisy_signal_wav_file[:-4] + "-final_denoising.wav", sampling_rate, denoised_signal)

	print("")
	return denoised_signal