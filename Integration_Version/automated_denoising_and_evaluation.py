import numpy
import os
import time

from noise_footprint_creation import best_esnr_channel_selection, esnr_and_common_zero_areas
from scipy.io import wavfile
from speech_intelligibility_metrics import metric_computation
from speech_recognition_metrics import awrp_computation
from wdwf import denoising

# Global Variables -------------------------------------------------------------------------------------------------------------------
ensr_method = 2 # Defines if the ESNR is estimated using the maximum value or the maximum short-term energy of the noisy channel.
				# Maximum value -> esnr_method = 1. Maximum short-term energy value -> esnr_method != 1.
				# Referenced in wdwfs_and_objective_metrics() and in wdwfs_and_speech_recognition().
# ------------------------------------------------------------------------------------------------------------------------------------

def write_multichannel_wav_file(wav_filename, sampling_rate, encoding_type, multichannel_signal):

	"""Creates a multichannel wav file.

Inputs:
  wav_filename: The name of the multichannel wav file that is created. The wav file must be multichannel, not single-channel.
  sampling_rate: The sampling rate of the multichannel wav file. Usually, sampling_rate = 16000 Hz.
  encoding_type: The encoding type of the multichannel wav file. Usually, encoding_type = numpy.int16.
  multichannel_signal: A list with so many nested 1-D numpy arrays as the number of channels of the multichannel wav file.

Outputs:
  None."""

	signal_shape = [len(multichannel_signal[0]), len(multichannel_signal)] # [number_of_samples, number_of_channels].
	signal = numpy.empty(signal_shape, dtype = encoding_type)

	for c in range(signal.shape[1]):
		signal[:, c] = multichannel_signal[c]

	wavfile.write(wav_filename, sampling_rate, signal)

	return

def wdwfs_and_objective_metrics(clean_signal, noisy_signal, sampling_rate, time_information, metric_id, \
								wdwf_id, in_use, is_dwt, dwt_level, db_cutoff, a, b, c, d, dps):

	"""Performs an automated denoising procedure and evaluates its result with a speech intelligibility metric.

Creates a noise footprint from the noisy signal by computing the Effective Signal to Noise Ratio (ESNR) of the noisy signal.
Denoises the noisy signal using a Wavelet Domain Wiener Filter (WDWF) and, as a result, creates a denoised signal.
Evaluates the denoised signal in terms of speech intelligibility with the assistance of the corresponding clean signal.

Inputs:
  clean_signal: A 1-D numpy array that represents the clean signal in the time domain.
  noisy_signal: A 1-D numpy array that represents the noisy signal in the time domain.

  sampling_rate: The sampling rate of the clean and noisy signals. Usually, sampling_rate = 16000 Hz.
  A different value is going to give a result but it will most probably be wrong.

  time_information: True if it is desirable the function to return the denoising time and the evaluation time. False, otherwise.

  metric_id: Determines which speech intelligibility metric will be used. The metrics are described in [1] and they are 6.
  If metric_id = 1, the Magnitude-Squared Coherence (MSC) will be used.
  If metric_id = 2, the Coherence-based Speech Intelligibility Index (CSII) will be used.
  If metric_id = 3, the high-level CSII (CSII_high) will be used.
  If metric_id = 4, the mid-level CSII (CSII_mid) will be used.
  If metric_id = 5, the low-level CSII (CSII_low) will be used.
  If metric_id != 1, 2, 3, 4, 5, the I3 will be used.

  wdwf_id: Determines which WDWF will be used. The WDWFs are described in [2] and they are 4.
  If wdwf_id = 1, the WDWF_0 will be used.
  If wdwf_id = 2, the WDWF_I will be used.
  If wdwf_id = 3, the WDWF_I&II will be used.
  If wdwf_id != 1, 2, 3, the WDWF_II will be used.

  in_use: True if the perceptual filter for broad-band acoustic noise, presented in [2], will be used.

  is_dwt: True if a Discrete Wavelet Transform (DWT) will be used (preferably 6-level or 7-level).
  False if the 22-bands Wavelet Packet Analysis WPA will be used.

  dwt_level: Defines the level of the DWT (preferably 6 or 7) that will be used. If the WPA is selected, dwt_level = 6.

  db_cutoff: The threshold needed for the ESNRs to be computed. It's value is expressed in negative dB.
  If esnr_method = 1, usually db_cutoff = -20.0 dB.
  If esnr_method != 1, usually db_cutoff = -7.0 dB.

  a: Parameter of the WDWFs. If wdwf_id = 1, usually a = 4.0. Otherwise, usually a = 2.0. More in [2].
  b: Parameter of the WDWFs. If wdwf_id = 1, usually b = 0.5. Otherwise, usually b = 2.0. More in [2].
  c: Parameter of the WDWFs. If wdwf_id = 1, usually c = 0.25. Otherwise, usually c = 1.0. More in [2].

  d: Parameter of the WDWFs. Its value depends on the highest frequency of the noisy signal.
  For highest_frequency = 8000 Hz, usually d = 0.22. If wdwf_id = 1, d is redundant. More in [2].

  dps: Parameter of the WDWFs. Usually, dps = 0.9. If wdwf_id = 1 or wdwf_id = 2, dps is redundant. More in [2].

Outputs:
  metric: The value of the speech intelligibility metric that was selected to be computed. It has a resolution of 0.1%.
  For instance, if metric_value = 76.9%, then the next better value of the metric is better_metric_value = 77%.
  Normally, this value is between 0.0 and 100.0. If it is -1.0, then an unpleasant situation came to surface.

  If time_information = True:
    denoising_time: The time, in seconds, that the denoising procedure needed in order to give its result.
    evaluation_time: The time, in seconds, that the evaluation procedure needed in order to give its result.

[1] Ma, J., Hu, Y., and Loizou, P.C. (2009). Objective measures for predicting speech intelligibility in noisy conditions based on new
band-importance functions. The Journal of the Acoustical Society of America, volume 125, issue 5, pages 3387-3405, May 2009.

[2] Dimoulas, C., Kalliris, G., Papanikolaou, G., and Kalampakas, A. (2006). Novel wavelet domain Wiener filtering de-noising
techniques: Application to bowel sounds captured by means of abdominal surface vibrations.
Biomedical Signal Processing and Control, volume 1, issue 3, pages 177-218, July 2006."""

	t_denoising_start = time.time()
	noise_footprint = best_esnr_channel_selection(noisy_signal, sampling_rate, 1, db_cutoff, ensr_method)

	# Checking if the noise_footprint has samples. -----------------------------------------------------------------------------------
	if noise_footprint.shape[0] == 0:
		if time_information:
			t_denoising_end = time.time()
			return [-1.0, t_denoising_end - t_denoising_start, 0.0]

		else:
			return -1.0
	# --------------------------------------------------------------------------------------------------------------------------------

	# Checking if the noise_footprint is a zero-signal. ------------------------------------------------------------------------------
	noise_footprint_list = list(noise_footprint)

	if noise_footprint_list.count(0) == len(noise_footprint_list):
		denoised_signal = noisy_signal

	else:
		denoised_signal = denoising(noisy_signal, noise_footprint, sampling_rate, wdwf_id, in_use, is_dwt, dwt_level, a, b, c, d, dps)
	# --------------------------------------------------------------------------------------------------------------------------------

	# Checking if the denoised_signal has samples. -----------------------------------------------------------------------------------
	if denoised_signal.shape[0] == 0:
		if time_information:
			t_denoising_end = time.time()
			return [-1.0, t_denoising_end - t_denoising_start, 0.0]

		else:
			return -1.0
	# --------------------------------------------------------------------------------------------------------------------------------

	t_denoising_end = time.time()

	# Checking if the clean_signal and the denoised_signal have the same duration. ---------------------------------------------------
	if clean_signal.shape[0] != denoised_signal.shape[0]:
		min_length = min(clean_signal.shape[0], denoised_signal.shape[0])
		clean_signal = clean_signal[:min_length]
		denoised_signal = denoised_signal[:min_length]
	# --------------------------------------------------------------------------------------------------------------------------------

	metric = metric_computation(clean_signal, denoised_signal, sampling_rate, metric_id)
	t_evaluation_end = time.time()

	if time_information:
		denoising_time = t_denoising_end - t_denoising_start
		evaluation_time = t_evaluation_end - t_denoising_end

		return [metric, denoising_time, evaluation_time]

	else:
		return metric

def wdwfs_and_speech_recognition(rapp_platform_api_instance, asr_id, asr_parameters, recognition_word, noisy_signal, sampling_rate, \
								 time_information, noise_estimation_process_id, wdwf_id, in_use, is_dwt, dwt_level, \
								 db_cutoff, a, b, c, d, dps):

	"""Performs an automated denoising procedure and evaluates its result using a speech recognition system.

The automated denoising procedure consists of 2 parts.
The first part is the creation of the noise footprint from the noisy signal.
The second part is the denoising of the noisy signal and the creation of the denoised signal.

First Part (Creation of the noise footprint from the noisy signal):
  There are 2 ways to create the noise footprint from the noisy signal. The first is to select the noisy channel with the best
  Effective Signal to Noise Ratio (ESNR) value and to create a noise footprint from this noisy channel.
  The noise_footprint_creation.best_esnr_channel_selection() implements this whole procedure.

  The second way is to calculate the ESNR value of each noisy channel and to create a simplistically denoised channel for each noisy
  channel using its ESNR. Afterwards, the common zero areas from the denoised channels have to be found. Finally, a noise footprint
  for each channel can be created by matching the common zero areas to the channels of the noisy signal.
  The noise_footprint_creation.esnr_and_common_zero_areas() implements this whole procedure.

Second Part (Denoising of the noisy signal and creation of the denoised signal):
  If the noise_footprint_creation.best_esnr_channel_selection() has been selected, the noisy channel with the best ESNR value is
  denoised, with the assistance of its corresponding noise footprint, by using a Wavelet Domain Wiener Filter (WDWF).

  If the noise_footprint_creation.esnr_and_common_zero_areas() has been selected, each channel of the noisy signal is denoised
  separately, with the assistance of its corresponding noise footprint. However, all the channels are denoised using the same WDWF.
  As a result, they are created so many denoised signals as the number of channels of the noisy signal. Eventually, these denoised
  signals are merged into 1 denoised signal using Sox [1].

The performance of the speech recognition system is evaluated using the Automatic Word Recognition Precision (AWRP) metric,
described in [2]. As a consequence, the final denoised signal is evaluated by the AWRP metric.

Inputs:
  rapp_platform_api_instance: An instance of the class RappPlatformAPI of the module RappPlatformApi.py of the package RappCloud.

  asr_id: Determines the Automatic Speech Recognition (ASR) system that will be used.
  If asr_id = 1, Sphinx4 will be used [3]. Otherwise, Google Cloud Speech-to-Text will be used [4].

  asr_parameters: A tuple that contains some necessary parameters for the operation of the ASR systems.
  If asr_id = 1, these parameters are the audio_source, the language and the words as defined in [5].
  Otherwise, they are the audio_source and the language as defined in [6].

  recognition_word: A string that indicates the word that has to be recognized.
  noisy_signal: A 2-D numpy array of shape (number_of_samples, number_of_channels).

  sampling_rate: The sampling rate of the noisy signal. Usually, sampling_rate = 16000 Hz.
  A different value is going to give a result but it will most probably be wrong.

  time_information: True if it is desirable the function to return the denoising time and the evaluation time. False, otherwise.

  noise_estimation_process_id: 1 if the noise_footprint_creation.best_esnr_channel_selection() will be used.
  Different from 1 otherwise.

  wdwf_id: Determines which WDWF will be used. The WDWFs are described in [7] and they are 4.
  If wdwf_id = 1, the WDWF_0 will be used.
  If wdwf_id = 2, the WDWF_I will be used.
  If wdwf_id = 3, the WDWF_I&II will be used.
  If wdwf_id != 1, 2, 3, the WDWF_II will be used.

  in_use: True if the perceptual filter for broad-band acoustic noise, presented in [7], will be used.

  is_dwt: True if a Discrete Wavelet Tranform (DWT) will be used (preferably 6-level or 7-level).
  False if the 22-bands Wavelet Packet Analysis (WPA) will be used.

  dwt_level: Defines the level of the DWT (preferably 6 or 7) that will be used. If the WPA is selected, dwt_level = 6.

  db_cutoff: The threshold needed for the ESNRs to be computed. It's value is expressed in negative dB.
  If esnr_method = 1, usually db_cutoff = -20.0 dB.
  If esnr_method != 1, usually db_cutoff = -7.0 dB.

  a: Parameter of the WDWFs. If wdwf_id = 1, usually a = 4.0. Otherwise, usually a = 2.0. More in [7].
  b: Parameter of the WDWFs. If wdwf_id = 1, usually b = 0.5. Otherwise, usually b = 2.0. More in [7].
  c: Parameter of the WDWFs. If wdwf_id = 1, usually c = 0.25. Otherwise, usually c = 1.0. More in [7].

  d: Parameter of the WDWFs. Its value depends on the highest frequency of the noisy signal.
  For highest_frequency = 8000 Hz, usually d = 0.22. If wdwf_id = 1, d is redundant. More in [7].

  dps: Parameter of the WDWFs. Usually, dps = 0.9. If wdwf_id = 1 or wdwf_id = 2, dps is redundant. More in [7].

Outputs:
  awrp: The value of the AWRP metric. It will be either 100.0 or 0.0. If it is -1.0, then an unpleasant situation came to surface.
  error: A Unicode string that normally has 0 length. Otherwise, it represents an error message.

  If time_information = True:
    denoising_time: The time, in seconds, that the denoising procedure needed in order to give its result.
    evaluation_time: The time, in seconds, that the evaluation procedure needed in order to give its result.

[1] SoX is cross-platform command line utility for computer audio files (http://sox.sourceforge.net/Main/HomePage).

[2] Tsardoulias, E.G., Thallas, A.G., Symeonidis, A.L., and Mitkas, P.A. (2016). Improving multilingual interaction for consumer
robots through signal enhancement in multichannel speech. JAES, volume 64, issue 7/8, pages 514-524, July 2016.

[3] Sphinx-4 is a state-of-the-art, speaker-independent, continuous speech recognition system written entirely in the Java programming
language (https://github.com/cmusphinx/sphinx4).

[4] Google Cloud Speech-to-Text is a speech-to-text conversion system powered by machine learning and it is available for short-form
or long-form audio (https://cloud.google.com/speech-to-text/).

[5] RAPP Speech Detection using Sphinx4
(https://github.com/rapp-project/rapp-platform/wiki/RAPP-Speech-Detection-using-Sphinx4).

[6] RAPP Speech Detection using Google API
(https://github.com/rapp-project/rapp-platform/wiki/RAPP-Speech-Detection-using-Google-API).

[7] Dimoulas, C., Kalliris, G., Papanikolaou, G., and Kalampakas, A. (2006). Novel wavelet domain Wiener filtering de-noising
techniques: Application to bowel sounds captured by means of abdominal surface vibrations.
Biomedical Signal Processing and Control, volume 1, issue 3, pages 177-218, July 2006."""

	t_denoising_start = time.time()
	number_of_channels = noisy_signal.shape[1]

	if noise_estimation_process_id == 1:
		noisy_channel, noise_footprint = best_esnr_channel_selection(noisy_signal, sampling_rate, number_of_channels, \
																	 db_cutoff, ensr_method)

		# Checking if the noise_footprint has samples. -------------------------------------------------------------------------------
		if noise_footprint.shape[0] == 0:
			if time_information:
				t_denoising_end = time.time()
				return [-1.0, u"", t_denoising_end - t_denoising_start, 0.0]

			else:
				return [-1.0, u""]
		# ----------------------------------------------------------------------------------------------------------------------------

		# Checking if the noise_footprint is a zero-signal. --------------------------------------------------------------------------
		noise_footprint_list = list(noise_footprint)

		if noise_footprint_list.count(0) == len(noise_footprint):
			denoised_signal = noisy_channel

		else:
			denoised_signal = denoising(noisy_channel, noise_footprint, sampling_rate, \
										wdwf_id, in_use, is_dwt, dwt_level, a, b, c, d, dps)
		# ----------------------------------------------------------------------------------------------------------------------------

		# Checking if the denoised_signal has samples. -------------------------------------------------------------------------------
		if denoised_signal.shape[0] == 0:
			if time_information:
				t_denoising_end = time.time()
				return [-1.0, u"", t_denoising_end - t_denoising_start, 0.0]

			else:
				return [-1.0, u""]
		# ----------------------------------------------------------------------------------------------------------------------------

		t_denoising_end = time.time()
		wavfile.write("singlechannel_denoised_signal.wav", sampling_rate, denoised_signal)

	else:
		noise_footprints = esnr_and_common_zero_areas(noisy_signal, sampling_rate, number_of_channels, db_cutoff, ensr_method)

		# Checking if common zero areas were found. ----------------------------------------------------------------------------------
		if len(noise_footprints) == 0:
			if time_information:
				t_denoising_end = time.time()
				return [-1.0, u"", t_denoising_end - t_denoising_start, 0.0]

			else:
				return [-1.0, u""]
		# ----------------------------------------------------------------------------------------------------------------------------

		denoised_channels = []
		encoding_type = noisy_signal.dtype

		for channel in range(number_of_channels):
			noisy_channel = noisy_signal[:, channel]
			noise_footprint = noise_footprints[channel]

			# Checking if the noise_footprint is a zero-signal. ----------------------------------------------------------------------
			noise_footprint_list = list(noise_footprint)

			if noise_footprint_list.count(0) == len(noise_footprint_list):
				denoised_channel = noisy_channel

			else:
				denoised_channel = denoising(noisy_channel, noise_footprint, sampling_rate, \
											 wdwf_id, in_use, is_dwt, dwt_level, a, b, c, d, dps)
			# ------------------------------------------------------------------------------------------------------------------------

			# Checking if the denoised_channel has samples. --------------------------------------------------------------------------
			if denoised_channel.shape[0] == 0:
				if time_information:
					t_denoising_end = time.time()
					return [-1.0, u"", t_denoising_end - t_denoising_start, 0.0]

				else:
					return [-1.0, u""]
			# ------------------------------------------------------------------------------------------------------------------------

			denoised_channels.append(denoised_channel)

		t_denoising_end = time.time()
		write_multichannel_wav_file("multichannel_denoised_signal.wav", sampling_rate, encoding_type, denoised_channels)
		command = "sox multichannel_denoised_signal.wav -c 1 -r " + str(sampling_rate) + " singlechannel_denoised_signal.wav"
		os.system(command)

	awrp, error = awrp_computation(rapp_platform_api_instance, asr_id, asr_parameters, \
								   "singlechannel_denoised_signal.wav", recognition_word)

	t_evaluation_end = time.time()

	if time_information:
		denoising_time = t_denoising_end - t_denoising_start
		evaluation_time = t_evaluation_end - t_denoising_end

		return [awrp, error, denoising_time, evaluation_time]

	else:
		return [awrp, error]