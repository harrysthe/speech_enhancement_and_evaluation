import numpy

from math import exp, floor, log
from scipy import signal

# Global Variables -------------------------------------------------------------------------------------------------------------------
window_type = "hamming" # Represents the type of the windows used in the computation of the STFTs. Referenced in metric_computation().

window_duration = 0.032 # In seconds. Represents the duration of the windows used in the computation of the STFTs.
						# Referenced in metric_computation() and in bark_scale_computation_8kHz().

overlap_ratio = 0.75 # Defines the overlap of the windows used in the computation of the STFTs. Referenced in metric_computation().
					 # For instance, an overlap ratio of 75% means that the second window starts after the first 25% of the samples of
					 # the first window. Therefore, the next 75% of the samples of the first window are overlapped by the second one.

roex_filter_lower_bound = 0.000001 # 10 * log10(0.000001) = -60 dB. Represents the minimum value of the ro-ex filters. Any value of
								   # the ro-ex filters lesser than this constant value is equalised with this constant value.
								   # For example, if roex_filter_value = 0.0, then roex_filter_value = roex_filter_lower_bound.
								   # Referenced in simplified_roex_filters_and_gaussian_critical_filters().

critical_filter_lower_bound = 0.001 # 20 * log10(0.001) = -60 dB. Exact the same meaning and reference with that of the
									# roex_filter_lower_bound but critical_filter_lower_bound is associated with the critical filters.

sdr_upper_bound = 15.0 # In dB. The maximum value that can correspond to any SDR. Referenced in bif_and_sdr_computation().
					   # If sdr_value > sdr_upper_bound, then sdr_value = sdr_upper_bound.

sdr_lower_bound = -15.0 # In dB. The minimum value that can correspond to any SDR. Referenced in bif_and_sdr_computation().
						# If sdr_value < sdr_lower_bound, then sdr_value = sdr_lower_bound.

rms_low_bound_constant_1 = 0.3163 # 20 * log10(0.3163) = -10 dB. Represents the boundary of the 10 dB below the overall RMS.
								  # Referenced in three_level_csii_computation().

rms_low_bound_constant_2 = 0.0316 # 20 * log10(0.0316) = -30 dB. Represents the boundary of the 30 dB below the overall RMS.
								  # Referenced in three_level_csii_computation().
# ------------------------------------------------------------------------------------------------------------------------------------

def short_time_rms_computation(signal_time_domain, sampling_rate, time_moments):

	"""Computes the short time Root Mean Square (RMS) value of a signal at each time window and its overall RMS as well.

Inputs:
  signal_time_domain: A 1-D numpy array that represents the signal in interest.
  sampling_rate: Usually, sampling_rate = 16000 Hz.
  time_moments: A 1-D numpy array that contains the time moments at which the segments of the short time analysis start.

Outputs:
  short_time_rms: A list which contains the RMS value at each segment.
  overall_rms: The overall RMS value of the signal_time_domain."""

	short_time_rms = []
	rms_counter, rms_sum, overall_rms_sum = 0, 0.0, 0.0
	segment_duration = time_moments[1] - time_moments[0] # In seconds.
	segment_number_of_samples = int(segment_duration * sampling_rate)

	for sample in signal_time_domain:
		overall_rms_sum = overall_rms_sum + (sample ** 2.0)

		if rms_counter == 0:
			short_time_rms.append(abs(sample))

		else:
			rms_sum = rms_sum + (sample ** 2.0)

			if (rms_counter % segment_number_of_samples) == 0:
				rms_value = (rms_sum / segment_number_of_samples) ** 0.5
				short_time_rms.append(rms_value)
				rms_sum = 0.0

		rms_counter = rms_counter + 1

	if rms_sum != 0.0:
		segment_number_of_samples = rms_counter % segment_number_of_samples
		rms_value = (rms_sum / segment_number_of_samples) ** 0.5
		short_time_rms.append(rms_value)

	overall_rms = (overall_rms_sum / signal_time_domain.shape[0]) ** 0.5

	return [short_time_rms, overall_rms]

def bark_scale_computation_8kHz():

	"""Approximates the frequency bands of the Bark scale when the highest frequency is 8 kHz.

Because the Short Time Fourier Transform (STFT) windows have 32 ms duration (see the global variable window_duration),
the lowest frequency that can complete a period is 31.25 Hz. In order to account for that, the first band begins from 30 Hz.

Inputs:
  None.

Outputs:
  bands_low_freqs: A list which contains the lowest frequency of each band starting from the lowest band.
  bands_high_freqs: A list which contains the highest frequency of each band starting from the lowest band.
  bands_center_freqs: A list which contains the center frequency of each band starting from the lowest band."""

	bands_low_freqs = [30, 130, 230, 330, 430, 540, 660, 800, 950, 1110, 1300, 1510, 1750, \
					   2030, 2350, 2720, 3150, 3660, 4270, 5000, 5880, 6930] # In Hz.

	bands_bandwidths = [100, 100, 100, 100, 110, 120, 140, 150, 160, 190, 210, 240, 280, \
						320, 370, 430, 510, 610, 730, 880, 1050, 1070] # In Hz.

	bands_high_freqs = [low_freq + bandwidth for low_freq, bandwidth in zip(bands_low_freqs, bands_bandwidths)]
	bands_center_freqs = [(low_freq + high_freq) / 2 for low_freq, high_freq in zip(bands_low_freqs, bands_high_freqs)]

	return [bands_low_freqs, bands_high_freqs, bands_center_freqs]

def simplified_roex_filters_and_gaussian_critical_filters(sampling_rate, freqs, bands):

	"""Computes the simplified ro-ex filters suggested by Moore and Glasberg [1], as well as some Gaussian-shaped critical filters.

Inputs:
  sampling_rate: Usually, sampling_rate = 16000 Hz.
  freqs: A 1-D array that contains the frequencies of the selected Fourier Transform (FT). Negative frequencies must be excluded.

  bands: Represents an auditory frequency analysis and it is consisted of 3 nested lists.
  One for the lowest frequency of each band, one for the highest one and one for the center one.
  The 3 lists must have the same length and this length equals the number of bands of the auditory frequency analysis.
  The first element of each of the 3 lists corresponds to the lowest band.

Outputs:
  w_band_level: A list with so many nested lists as the number of bands.
  Each of the nested lists corresponds to a simplified ro-ex filter and contains so many elements as the number of freqs.
  Each element represents the value of the simplified ro-ex filter at a specific band and frequency.

  critical_filter_band_level: A list with so many nested lists as the number of bands.
  Each of the nested lists corresponds to a Gaussian-shaped critical filter and contains so many elements as the number of freqs.
  Each element represents the value of the Gaussian-shaped critical filter at a specific band and frequency.

[1] Moore, B. C. J., and Glasberg, B. R. (1983). Suggested formulae for calculating auditory-filter bandwidths and excitation
patterns. The Journal of the Acoustical Society of America, volume 74, issue 3, page 750."""

	q_band_level, p_band_level = [], []
	bands_low_freqs, bands_high_freqs, bands_center_freqs = bands
	number_of_bands = len(bands_low_freqs)

	for b in range(number_of_bands):
		bandwidth = bands_high_freqs[b] - bands_low_freqs[b]
		min_freq_distance = sampling_rate / 2

		for freq in freqs:
			if (b > 0) and (freq <= q_band_level[b - 1]):
				continue

			freq_distance = abs(bands_center_freqs[b] - freq)

			if freq_distance < min_freq_distance:
				min_freq_distance = freq_distance
				q_value = freq

		q_band_level.append(q_value)
		p_band_level.append(4.0 * (q_value / bandwidth))

	w_band_level, critical_filter_band_level = [], []
	max_frequency = sampling_rate / 2.0 # It is important the max_frequency variable to be float.
	min_bandwidth = bands_high_freqs[0] - bands_low_freqs[0]
	number_of_freqs = freqs.shape[0]

	for b in range(number_of_bands):
		w_frequency_level, critical_filter_frequency_level = [], []

		normalized_freq = floor((bands_center_freqs[b] / max_frequency) * number_of_freqs)
		bandwidth = bands_high_freqs[b] - bands_low_freqs[b]
		normalized_bw = (bandwidth / max_frequency) * number_of_freqs
		normalization_factor = log(min_bandwidth) - log(bandwidth) # No log(min_bandwidth / bandwidth) because
																   # min_bandwidth and bandwidth are integers.

		for f in range(number_of_freqs):
			g = abs(1.0 - (freqs[f] / q_band_level[b]))
			w_value = (1.0 + (p_band_level[b] * g)) * exp((-p_band_level[b]) * g)
			critical_filter_value = exp(-11.0 * (((f - normalized_freq) / normalized_bw) ** 2.0) + normalization_factor)

			if w_value < roex_filter_lower_bound:
				w_value = roex_filter_lower_bound

			if critical_filter_value < critical_filter_lower_bound:
				critical_filter_value = critical_filter_lower_bound

			w_frequency_level.append(w_value)
			critical_filter_frequency_level.append(critical_filter_value)

		w_band_level.append(w_frequency_level)
		critical_filter_band_level.append(critical_filter_frequency_level)

	return [w_band_level, critical_filter_band_level]

def msc_computation(clean_signal_stft_values, denoised_signal_stft_values, output_id):

	"""Computes the Magnitude-Squared Coherence (MSC) between a denoised signal and its corresponding clear version.

For information about the MSC equation, see metric_computation().

Inputs:
  clean_signal_stft_values: A 2-D numpy array of shape (number_of_frequencies, number_of_time_moments).
  Output of the function scipy.signal.stft().

  denoised_signal_stft_values: A 2-D numpy array of shape (number_of_frequencies, number_of_time_moments).
  Output of the function scipy.signal.stft().

  output_id: Determines the output of this specific function.

Outputs:
  If output_id = 1, msc: The averaged, across all frequency bins, MSC value which is between 0.0 and 1.0.
  The computation takes into consideration only the frequency bins at which the denominator of the MSC equation is not 0.0.
  If the denominator is 0.0 for all frequency bins (for example, the denoised signal is a zero-signal), msc = 0.0.

  If output_id != 1, msc_frequency_level: A list with so many elements as the number of frequency bins.
  Each element represents the MSC value at a specific frequency bin, which is between 0.0 and 1.0 in normal conditions.
  If an element has a -1.0 value, it means that, at this specific frequency bin, the denominator of the MSC equation is 0.0."""

	msc_frequency_level = []
	denoised_signal_stft_conjugate_values = numpy.conjugate(denoised_signal_stft_values)

	for f in range(clean_signal_stft_values.shape[0]):
		clean_signal_energy, denoised_signal_energy, product_energy = 0.0, 0.0, 0.0

		for t in range(clean_signal_stft_values.shape[1]):
			clean_signal_magnitude = abs(clean_signal_stft_values[f][t]) ** 2.0
			denoised_signal_magnitude = abs(denoised_signal_stft_values[f][t]) ** 2.0
			product = clean_signal_stft_values[f][t] * denoised_signal_stft_conjugate_values[f][t]

			clean_signal_energy = clean_signal_energy + clean_signal_magnitude
			denoised_signal_energy = denoised_signal_energy + denoised_signal_magnitude
			product_energy = product_energy + product

		if (clean_signal_energy == 0.0) or (denoised_signal_energy == 0.0):
			msc_value = -1.0 # Typically, 0.0 <= msc_value <= 1.0.

		else:
			total_product_magnitude = abs(product_energy) ** 2.0
			msc_value = total_product_magnitude / (clean_signal_energy * denoised_signal_energy)

		msc_frequency_level.append(msc_value)

	if output_id == 1:
		if msc_frequency_level.count(-1.0) == len(msc_frequency_level):
			# This situation can arise if, for example, the denoised signal is 0 for all time moments.
			msc = 0.0 # Worst acceptable value for the MSC metric.

		else:
			sum_msc, msc_counter = 0.0, 0

			for msc_value in msc_frequency_level:
				if msc_value >= 0:
					sum_msc = sum_msc + msc_value
					msc_counter = msc_counter + 1

			msc = sum_msc / msc_counter

		return msc

	return msc_frequency_level # It is possible all of the list's elements to have a -1.0 value.

def bif_and_sdr_computation(clean_signal_stft_values, denoised_signal_stft_values, msc_frequency_level, critical_band_filters, \
							roex_filter_band_level, number_of_freqs, number_of_time_moments, number_of_bands, metric_id):

	"""Computes a Band-Importance Function (BIF) based on the excitation spectrum of the clean signal. It, also, computes the
Signal-to-Distortion Ratio (SDR) used by the Coherence-based Speech Intelligibility Index (CSII) metric.

Inputs:
  clean_signal_stft_values: A 2-D numpy array of shape (number_of_freqs, number_of_time_moments).
  Output of the function scipy.signal.stft(). Negative frequencies must be excluded.

  denoised_signal_stft_values: A 2-D numpy array of shape (number_of_freqs, number_of_time_moments).
  Output of the function scipy.signal.stft(). Negative frequencies must be excluded.

  msc_frequency_level: A list with so many elements as the number_of_freqs. Negative frequencies must be excluded.
  Each element represents the Magnitude-Squared Coherence (MSC) value at a specific frequency.
  If at some frequency the denominator of the MSC equation is 0.0, that specific element must have a negative value.

  critical_band_filters: A list with so many nested lists as the number_of_bands.
  Each of the nested lists corresponds to a critical filter and contains so many elements as the number_of_freqs.
  Each element represents the value of the critical filter at a specific band and frequency.
  The excitation spectrum of the clean signal is based on the critical_band_filters.

  roex_filter_band_level: First output of the simplified_roex_filters_and_gaussian_critical_filters().
  Its dimensions must be the same with those of the critical_band_filters.

  number_of_freqs: The number of frequencies of the Short Time Fourier Transform (STFT) that was selected.
  Negative frequencies must be excluded.

  number_of_time_moments: The number of time moments of the STFT that was selected.
  number_of_bands: The number of bands of the auditory frequency analysis that was selected.

  metric_id: Determines the speech intelligibility metric that is being used. More in metric_computation().
  If metric_id = 2, the CSII is being used.
  If metric_id = 3, the high-level CSII (CSII_high) is being used.
  If metric_id = 4, the mid-level CSII (CSII_mid) is being used.
  If metric_id != 2, 3, 4 the low-level CSII (CSII_low) is being used.

Outputs:
  bif_band_level: A list with so many nested lists as the number_of_bands.
  Each nested list has so many elements as the number_of_time_moments.
  Each element represents the value of the BIF at a specific band and time moment.

  sdr_band_level: A list with so many nested lists as the number_of_bands.
  Each nested list has so many elements as the number_of_time_moments.
  Each element represents the value of the SDR at a specific band and time moment."""

	bif_band_level, sdr_band_level = [], []
	denoised_signal_magnitude_lower_bound = 0.000001 # 10.0 * log10(0.000001) = -60 dB.

	# Optimum value of p given that the speech material consists of sentences. -------------------------------------------------------
	if metric_id == 2:
		p = 4.0

	elif metric_id == 3:
		p = 2.0

	elif metric_id == 4:
		p = 1.0

	else:
		p = 0.5
	# --------------------------------------------------------------------------------------------------------------------------------

	for b in range(number_of_bands):
		bif_time_level, sdr_time_level = [], []

		for t in range(number_of_time_moments):
			excitation_energy, nominator_sum, denominator_sum = 0.0, 0.0, 0.0

			for f in range(number_of_freqs):
				clean_signal_magnitude = abs(clean_signal_stft_values[f][t])
				critical_filter_value = critical_band_filters[b][f]
				excitation_energy = excitation_energy + (clean_signal_magnitude * critical_filter_value)

				denoised_signal_magnitude = abs(denoised_signal_stft_values[f][t]) ** 2.0
				msc_value = msc_frequency_level[f]
				roex_filter_value = roex_filter_band_level[b][f]

				if msc_value < 0.0:
					msc_value = 0.0
					opposite_msc_value = 0.0

				else:
					opposite_msc_value = 1.0 - msc_value

				if denoised_signal_magnitude < denoised_signal_magnitude_lower_bound:
					denoised_signal_magnitude = denoised_signal_magnitude_lower_bound

				nominator_value = roex_filter_value * msc_value * denoised_signal_magnitude
				nominator_sum = nominator_sum + nominator_value
				denominator_value = roex_filter_value * opposite_msc_value * denoised_signal_magnitude
				denominator_sum = denominator_sum + denominator_value

			bif_value = excitation_energy ** p # It is possible the excitation_energy to be 0.0.
			bif_time_level.append(bif_value)

			sdr_value = 10.0 * log(nominator_sum / denominator_sum, 10)
			# It is highly unlikely the nominator_sum or denominator_sum to be 0.0.
			# However, denominator_sum = 0.0 if msc_value = 1.0 at each frequency.
			# This can take place if number_of_time_moments = 2. In other words, if number_of_windows = 1.

			if sdr_value > sdr_upper_bound:
				sdr_value = sdr_upper_bound

			elif sdr_value < sdr_lower_bound:
				sdr_value = sdr_lower_bound

			sdr_value = (sdr_value - sdr_lower_bound) / (sdr_upper_bound - sdr_lower_bound)
			sdr_time_level.append(sdr_value)

		bif_band_level.append(bif_time_level)
		sdr_band_level.append(sdr_time_level)

	return [bif_band_level, sdr_band_level]

def csii_computation(clean_signal_stft_values, denoised_signal_stft_values, roex_filter_band_level, \
					 critical_filter_band_level, metric_id):

	"""Computes the Coherence-based Speech Intelligibility Index (CSII) between a denoised signal and its corresponding clear version.

Inputs:
  clean_signal_stft_values: A 2-D numpy array of shape (number_of_frequencies, number_of_time_moments).
  Output of the function scipy.signal.stft(). Negative frequencies must be excluded.

  denoised_signal_stft_values: A 2-D numpy array of shape (number_of_frequencies, number_of_time_moments).
  Output of the function scipy.signal.stft(). Negative frequencies must be excluded.

  roex_filter_band_level: First output of the simplified_roex_filters_and_gaussian_critical_filters().

  critical_filter_band_level: A list with so many nested lists as the number of bands of the selected auditory frequency analysis.
  Each of the nested lists corresponds to a critical filter and contains so many elements as the number of frequencies.
  Each element represents the value of the critical filter at a specific band and frequency.
  The dimensions of the critical_filter_band_level must be the same with those of the roex_filter_band_level.

  metric_id: Determines the speech intelligibility metric that is being used. More in metric_computation().
  If metric_id = 1, an error will be induced.
  If metric_id = 2, the CSII is being used.
  If metric_id = 3, the high-level CSII (CSII_high) is being used.
  If metric_id = 4, the mid-level CSII (CSII_mid) is being used.
  If metric_id != 1, 2, 3, 4 the low-level CSII (CSII_low) is being used.

Outputs:
  csii: The averaged, across all time moments, CSII value which is between 0.0 and 1.0.
  The computation takes into consideration only the segments wherein the clean signal is not zero-signal."""

	number_of_freqs = clean_signal_stft_values.shape[0]
	number_of_time_moments = clean_signal_stft_values.shape[1]
	number_of_bands = len(roex_filter_band_level)
	msc_frequency_level = msc_computation(clean_signal_stft_values, denoised_signal_stft_values, metric_id)

	if msc_frequency_level.count(-1.0) == len(msc_frequency_level):
		# This situation can arise if, for example, the denoised signal is 0 for all time moments.
		csii = 0.0 # Worst acceptable value for the CSII metric.

	else:
		bif_band_level, sdr_band_level = bif_and_sdr_computation(clean_signal_stft_values, denoised_signal_stft_values, \
																 msc_frequency_level, critical_filter_band_level, \
																 roex_filter_band_level, number_of_freqs, \
																 number_of_time_moments, number_of_bands, \
																 metric_id)

		csii_sum = 0.0
		number_of_active_time_moments = 0

		for t in range(number_of_time_moments):
			nominator_sum, denominator_sum = 0.0, 0.0

			for b in range(number_of_bands):
				nominator_sum = nominator_sum + (bif_band_level[b][t] * sdr_band_level[b][t])
				denominator_sum = denominator_sum + bif_band_level[b][t]

			if denominator_sum == 0.0:
				continue

			else:
				csii_sum = csii_sum + (nominator_sum / denominator_sum)
				number_of_active_time_moments = number_of_active_time_moments + 1

		if number_of_active_time_moments > 0: # It seems to me that this checking is a bit redundant, but I will keep it.
			csii = csii_sum / number_of_active_time_moments

		else:
			csii = 0.0 # Worst acceptable value for the CSII metric.

	return csii

def preparation_for_the_csii_computation(clean_signal_stft_values_time_level, denoised_signal_stft_values_time_level):

	"""A utility function that is used in the three_level_csii_computation() in order to save code space.

Inputs:
  clean_signal_stft_values_time_level: A list with nested 1-D numpy arrays.
  The length of the list equals the number of time moments of the Short Time Fourier Transform (STFT) that was selected.
  Each one of the nested 1-D numpy arrays contains so many elements as the number of frequencies of the STFT that was selected.
  Each element represents the STFT coefficient of the clean signal at a specific time moment and frequency.
  Negative frequencies must be excluded.

  denoised_signal_stft_values_time_level: A list with nested 1-D numpy arrays.
  The length of the list equals the number of time moments of the STFT that was selected.
  Each one of the nested 1-D numpy arrays contains so many elements as the number of frequencies of the STFT that was selected.
  Each element represents the STFT coefficient of the denoised signal at a specific time moment and frequency.
  Negative frequencies must be excluded.

Outputs:
  clean_signal_stft_values: A 2-D numpy array of shape (number_of_frequencies, number_of_time_moments).
  Negative frequencies are excluded.

  denoised_signal_stft_values: A 2-D numpy array of shape (number_of_frequencies, number_of_time_moments).
  Negative frequencies are excluded."""

	clean_signal_stft_values = numpy.array(clean_signal_stft_values_time_level, dtype = numpy.complex64)
	clean_signal_stft_values = numpy.transpose(clean_signal_stft_values)

	denoised_signal_stft_values = numpy.array(denoised_signal_stft_values_time_level, dtype = numpy.complex64)
	denoised_signal_stft_values = numpy.transpose(denoised_signal_stft_values)

	return [clean_signal_stft_values, denoised_signal_stft_values]

def three_level_csii_computation(clean_signal_stft_values, denoised_signal_stft_values, roex_filter_band_level, \
								 critical_filter_band_level, short_time_rms, overall_rms, metric_id):

	"""Divides the clean signal envelope into 3 amplitude regions and computes the Coherence-based Speech Intelligibility Index (CSII)
for each one of them.

The high-level CSII (CSII_high) is computed using the segments of the clean signal that are at or above the overall_rms.
The mid-level CSII (CSII_mid) is computed using those segments that are between 0.0 and 10.0 dB below the overall_rms.
The low-level CSII (CSII_low) is computed using those segments that are between 10.0 and 30.0 dB below the overall_rms.

A speech intelligibility model is computed using a linear weighting of the CSII_high, CSII_mid, and CSII_low. The weighted sum is then
transformed using a logistic function to give predictive intelligibility scores. This speech intelligibility model is denoted as I3.

For more information, see metric_computation().

Inputs:
  clean_signal_stft_values: A 2-D numpy array of shape (number_of_frequencies, number_of_time_moments).
  Output of the function scipy.signal.stft(). Negative frequencies must be excluded.

  denoised_signal_stft_values: A 2-D numpy array of shape (number_of_frequencies, number_of_time_moments).
  Output of the function scipy.signal.stft(). Negative frequencies must be excluded.

  roex_filter_band_level: First output of the simplified_roex_filters_and_gaussian_critical_filters().

  critical_filter_band_level: A list with so many nested lists as the number of bands of the selected auditory frequency analysis.
  Each of the nested lists corresponds to a critical filter and contains so many elements as the number of frequencies.
  Each element represents the value of the critical filter at a specific band and frequency.
  The dimensions of the critical_filter_band_level must be the same with those of the roex_filter_band_level.

  short_time_rms: A list which contains the short time Root Mean Square (RMS) value of the clean signal at each time window.
  The clean signal envelope is based on these short time RMS values.

  overall_rms: The overall RMS value of the clean signal.

  metric_id: Determines the speech intelligibility metric that is being used.
  If metric_id = 3, the CSII_high is being used.
  If metric_id = 4, the CSII_mid is being used.
  If metric_id = 5, the CSII_low is being used.
  If metric_id != 3, 4, 5, the I3 is being used.

Outputs:
  metric: The value of the metric that was selected, which is between 0.0 and 1.0."""

	number_of_time_moments = clean_signal_stft_values.shape[1]

	if (metric_id != 3) and (metric_id != 4) and (metric_id != 5):
		clean_signal_low_bound_rms_1 = overall_rms * rms_low_bound_constant_1
		clean_signal_low_bound_rms_2 = overall_rms * rms_low_bound_constant_2

		high_counter, mid_counter, low_counter = 0, 0, 0
		high_clean_signal_stft_values_time_level, high_denoised_signal_stft_values_time_level = [], []
		mid_clean_signal_stft_values_time_level, mid_denoised_signal_stft_values_time_level = [], []
		low_clean_signal_stft_values_time_level, low_denoised_signal_stft_values_time_level = [], []

		for t in range(number_of_time_moments):
			rms_value = short_time_rms[t]

			if rms_value >= overall_rms:
				high_counter = high_counter + 1
				high_clean_signal_stft_values_time_level.append(clean_signal_stft_values[:, t])
				high_denoised_signal_stft_values_time_level.append(denoised_signal_stft_values[:, t])

			elif rms_value >= clean_signal_low_bound_rms_1:
				mid_counter = mid_counter + 1
				mid_clean_signal_stft_values_time_level.append(clean_signal_stft_values[:, t])
				mid_denoised_signal_stft_values_time_level.append(denoised_signal_stft_values[:, t])

			elif rms_value >= clean_signal_low_bound_rms_2:
				low_counter = low_counter + 1
				low_clean_signal_stft_values_time_level.append(clean_signal_stft_values[:, t])
				low_denoised_signal_stft_values_time_level.append(denoised_signal_stft_values[:, t])

		if high_counter > 2:
			high_clean_signal_stft_values, high_denoised_signal_stft_values = preparation_for_the_csii_computation(\
																			  high_clean_signal_stft_values_time_level, \
																			  high_denoised_signal_stft_values_time_level)

			csii_high = csii_computation(high_clean_signal_stft_values, high_denoised_signal_stft_values, \
										 roex_filter_band_level, critical_filter_band_level, 3)

		else:
			csii_high = 0.0 # Worst acceptable value for the CSII_high metric.

		if mid_counter > 2:
			mid_clean_signal_stft_values, mid_denoised_signal_stft_values = preparation_for_the_csii_computation(\
																			mid_clean_signal_stft_values_time_level, \
																			mid_denoised_signal_stft_values_time_level)

			csii_mid = csii_computation(mid_clean_signal_stft_values, mid_denoised_signal_stft_values, \
										roex_filter_band_level, critical_filter_band_level, 4)

		else:
			csii_mid = 0.0 # Worst acceptable value for the CSII_mid metric.

		if low_counter > 2:
			low_clean_signal_stft_values, low_denoised_signal_stft_values = preparation_for_the_csii_computation(\
																			low_clean_signal_stft_values_time_level, \
																			low_denoised_signal_stft_values_time_level)

			csii_low = csii_computation(low_clean_signal_stft_values, low_denoised_signal_stft_values, \
										roex_filter_band_level, critical_filter_band_level, 5)

		else:
			csii_low = 0.0 # Worst acceptable value for the CSII_low metric.

		metric = (-3.47) + (0.001 * csii_high) + (9.99 * csii_mid) + (1.84 * csii_low)
		metric = 1.0 / (1.0 + exp(-metric))

	elif metric_id == 3:
		high_counter = 0
		high_clean_signal_stft_values_time_level, high_denoised_signal_stft_values_time_level = [], []

		for t in range(number_of_time_moments):
			rms_value = short_time_rms[t]

			if rms_value >= overall_rms:
				high_counter = high_counter + 1
				high_clean_signal_stft_values_time_level.append(clean_signal_stft_values[:, t])
				high_denoised_signal_stft_values_time_level.append(denoised_signal_stft_values[:, t])

		if high_counter > 2:
			high_clean_signal_stft_values, high_denoised_signal_stft_values = preparation_for_the_csii_computation(\
																			  high_clean_signal_stft_values_time_level, \
																			  high_denoised_signal_stft_values_time_level)

			metric = csii_computation(high_clean_signal_stft_values, high_denoised_signal_stft_values, \
									  roex_filter_band_level, critical_filter_band_level, 3)

		else:
			metric = 0.0 # Worst acceptable value for the CSII_high metric.

	elif metric_id == 4:
		mid_counter = 0
		mid_clean_signal_stft_values_time_level, mid_denoised_signal_stft_values_time_level = [], []
		clean_signal_low_bound_rms_1 = overall_rms * rms_low_bound_constant_1

		for t in range(number_of_time_moments):
			rms_value = short_time_rms[t]

			if (rms_value < overall_rms) and (rms_value >= clean_signal_low_bound_rms_1):
				mid_counter = mid_counter + 1
				mid_clean_signal_stft_values_time_level.append(clean_signal_stft_values[:, t])
				mid_denoised_signal_stft_values_time_level.append(denoised_signal_stft_values[:, t])

		if mid_counter > 2:
			mid_clean_signal_stft_values, mid_denoised_signal_stft_values = preparation_for_the_csii_computation(\
																			mid_clean_signal_stft_values_time_level, \
																			mid_denoised_signal_stft_values_time_level)

			metric = csii_computation(mid_clean_signal_stft_values, mid_denoised_signal_stft_values, \
									  roex_filter_band_level, critical_filter_band_level, 4)

		else:
			metric = 0.0 # Worst acceptable value for the CSII_mid metric.

	else: # metric_id = 5.
		low_counter = 0
		low_clean_signal_stft_values_time_level, low_denoised_signal_stft_values_time_level = [], []
		clean_signal_low_bound_rms_1 = overall_rms * rms_low_bound_constant_1
		clean_signal_low_bound_rms_2 = overall_rms * rms_low_bound_constant_2

		for t in range(number_of_time_moments):
			rms_value = short_time_rms[t]

			if (rms_value < clean_signal_low_bound_rms_1) and (rms_value >= clean_signal_low_bound_rms_2):
				low_counter = low_counter + 1
				low_clean_signal_stft_values_time_level.append(clean_signal_stft_values[:, t])
				low_denoised_signal_stft_values_time_level.append(denoised_signal_stft_values[:, t])

		if low_counter > 2:
			low_clean_signal_stft_values, low_denoised_signal_stft_values = preparation_for_the_csii_computation(\
																			low_clean_signal_stft_values_time_level, \
																			low_denoised_signal_stft_values_time_level)

			metric = csii_computation(low_clean_signal_stft_values, low_denoised_signal_stft_values, \
									  roex_filter_band_level, critical_filter_band_level, 5)

		else:
			metric = 0.0 # Worst acceptable value for the CSII_low metric.

	return metric

def metric_computation(clean_signal, denoised_signal, sampling_rate, metric_id):

	"""Computes a metric which estimates the speech intelligibility of an enhanced speech given the corresponding clean speech.

The function has been designed based on the assumptions that the highest frequency of the signals is 8000 Hz and the sampling rate
is 16000 Hz. A call of the function with different conditions of the ones specified by the aforementioned assumptions may be valid
but wrong.

Inputs:
  clean_signal: A 1-D numpy array that represents the clean signal.
  denoised_signal: A 1-D numpy array that represents the denoised signal.
  sampling_rate: The sampling rate of the clean and denoised signals. Usually, sampling_rate = 16000 Hz.

  metric_id: Determines which speech intelligibility metric will be used. The metrics are described in [1] and they are 6.
  If metric_id = 1, the Magnitude-Squared Coherence (MSC) will be used.
  If metric_id = 2, the Coherence-based Speech Intelligibility Index (CSII) will be used.
  If metric_id = 3, the high-level CSII (CSII_high) will be used.
  If metric_id = 4, the mid-level CSII (CSII_mid) will be used.
  If metric_id = 5, the low-level CSII (CSII_low) will be used.
  If metric_id != 1, 2, 3, 4, 5, the I3 will be used.

The metrics have a resolution of 0.1%. For instance, if msc_value = 76.9%, then the next better MSC value is better_msc_value = 77%.

In order for any of the metrics to be computed, a Short Time Fourier Transform (STFT) must be applied to the clean and denoised
signals. These STFTs must be represented by more than one window so as a full-biased estimation of the metrics to be prevented [2].

If the denoised signal is a zero-signal, the values of the metrics are standard. Thus, in this case:
MSC = 0.0, CSII = 0.0, CSII_high = 0.0, CSII_mid = 0.0, CSII_low = 0.0, and I3 = 3.0.

Outputs:
  metric: The value of the metric that was selected, which is normally between 0.0 and 100.0.
  If metric = -1.0, then an unpleasant situation came to surface.

[1] Ma, J., Hu, Y., and Loizou, P.C. (2009). Objective measures for predicting speech intelligibility in noisy conditions based on new
band-importance functions. The Journal of the Acoustical Society of America, volume 125, issue 5, pages 3387-3405, May 2009.

[2] Loizou, P.C. (2013). SPEECH ENHANCEMENT: Theory and Practice, Second Edition (page 560). Boca Raton, FL: CRC Press."""

	window_length = round(window_duration * sampling_rate) # In samples.
	overlap_length = round(window_length * overlap_ratio) # In samples.

	stft_freqs, time_moments, clean_signal_stft_values = signal.stft(clean_signal, sampling_rate, window_type, window_length, \
																	 overlap_length, return_onesided = True)

	# Checking if the time windows are more than 1. ----------------------------------------------------------------------------------
	if time_moments.shape[0] <= 2:
		# If the STFT is represented by 1 window, time_moments[0] = start of the window, time_moments[1] = end of the window.
		return -1.0

	stft_freqs, time_moments, denoised_signal_stft_values = signal.stft(denoised_signal, sampling_rate, window_type, window_length, \
																		overlap_length, return_onesided = True)

	if metric_id == 1:
		metric = msc_computation(clean_signal_stft_values, denoised_signal_stft_values, 1)

	else:
		bands = bark_scale_computation_8kHz()

		roex_filter_band_level, critical_filter_band_level = simplified_roex_filters_and_gaussian_critical_filters(sampling_rate, \
																												   stft_freqs, bands)

		if metric_id == 2:
			metric = csii_computation(clean_signal_stft_values, denoised_signal_stft_values, \
									  roex_filter_band_level, critical_filter_band_level, 2)

		else:
			short_time_rms, overall_rms = short_time_rms_computation(clean_signal, sampling_rate, time_moments)

			metric = three_level_csii_computation(clean_signal_stft_values, denoised_signal_stft_values, roex_filter_band_level, \
												  critical_filter_band_level, short_time_rms, overall_rms, metric_id)

	metric = round(100 * metric, 1)

	return metric