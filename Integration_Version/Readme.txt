noise_footprint_creation.py:

(1) Module that is being used for computing the data of the noise footprint given the data of the noisy signal.

(2) There are 2 main functions: best_esnr_channel_selection() and esnr_and_common_zero_areas(), which are called in the automated_denoising_and_evaluation.py module.

(3) best_esnr_channel_selection() can deal both with single-channel and multi-channel signals.

(4) esnr_and_common_zero_areas() can deal only with multi-channel signals.

(5) There is detailed documentation inside the code.

(6) Example:

    >>> from scipy.io import wavfile
    >>> import noise_footprint_creation

    >>> sampling_rate, noisy_signal_array = wavfile.read("sp23_car_sn05_16kHz.wav")
    >>> print(sampling_rate, noisy_signal_array.shape) -> (16000, (42418,))

    >>> noise_footprint_array = noise_footprint_creation.best_esnr_channel_selection(noisy_signal_array, sampling_rate, 1, -7.0, 2)

    >>> sampling_rate, noisy_signal_array = wavfile.read("thirsty-med-3_manos_16kHz.wav")
    >>> print(sampling_rate, noisy_signal_array.shape) -> (16000, (46421, 4))

    >>> best_esnr_noisy_channel_array, noise_footprint_array = noise_footprint_creation.best_esnr_channel_selection(noisy_signal_array, sampling_rate, 4, -7.0, 2)

    >>> noise_footprints_list = noise_footprint_creation.esnr_and_common_zero_areas(noisy_signal_array, sampling_rate, 4, -7.0, 2)
    >>> print(len(noise_footprints_list)) -> 4

-------------------------------------------------------------------------------------------------------------------------------------
fix_wpa_bug.py:

(1) Module that is being used for extending the length of a signal so as the signal to have an appropriate length for it's 22-bands Wavelet Packet Analysis (WPA).

(2) There is 1 main function: signal_adaptation_for_22_bands_wpa(), which is called in the wdwf.py module.

(3) signal_adaptation_for_22_bands_wpa() can deal only with single-channel signals.

(4) There is detailed documentation inside the code.

(5) Example:

    >>> from scipy.io import wavfile
    >>> import fix_wpa_bug

    >>> sampling_rate, noisy_signal_array = wavfile.read("sp23_car_sn05_16kHz.wav")
    >>> print(sampling_rate, noisy_signal_array.shape) -> (16000, (42418,))

    >>> extended_noisy_signal_array = fix_wpa_bug.signal_adaptation_for_22_bands_wpa(noisy_signal_array, 6)
    >>> print(extended_noisy_signal_array.shape) -> (42432,)

-------------------------------------------------------------------------------------------------------------------------------------
wdwf.py:

(1) Module that is being used for producing the data of the denoised signal given the data of the noisy signal and noise footprint.

(2) There is 1 main function: denoising(), which is called in the automated_denoising_and_evaluation.py module.

(3) denoising() can deal only with single-channel signals.

(4) There is detailed documentation inside the code.

(5) Example:

    >>> from scipy.io import wavfile
    >>> import noise_footprint_creation
    >>> import wdwf

    >>> sampling_rate, noisy_signal_array = wavfile.read("sp23_car_sn05_16kHz.wav")

    >>> noise_footprint_array = noise_footprint_creation.best_esnr_channel_selection(noisy_signal_array, sampling_rate, 1, -7.0, 2)

    >>> print(sampling_rate, noisy_signal_array.shape, noise_footprint_array.shape) -> (16000, (42418,), (6578,))

    >>> denoised_signal_array_1 = wdwf.denoising(noisy_signal_array, noise_footprint_array, sampling_rate, 3, True, True, 6, 2.0, 2.0, 1.0, 0.22, 0.9)

    >>> denoised_signal_array_2 = wdwf.denoising(noisy_signal_array, noise_footprint_array, sampling_rate, 2, False, True, 7, 2.0, 2.0, 1.0, 0.22, 0.9)

    >>> denoised_signal_array_3 = wdwf.denoising(noisy_signal_array, noise_footprint_array, sampling_rate, 4, False, False, 6, 2.0, 2.0, 1.0, 0.22, 0.9)

    >>> print(denoised_signal_array_1.shape, denoised_signal_array_2.shape, denoised_signal_array_3.shape) -> ((42418,), (42418,), (42432,))

-------------------------------------------------------------------------------------------------------------------------------------
speech_intelligibility_metrics.py:

(1) Module that is being used for computing some speech intelligibility metrics (MSC, CSII, CSII-high, CSII-mid, CSII-low, I3) given the data of the clean and denoised signals.

(2) There is 1 main function: metric_computation(), which is called in the automated_denoising_and_evaluation.py module.

(3) metric_computation() can deal only with single-channel signals.

(4) There is detailed documentation inside the code.

(5) Example:

    >>> from scipy.io import wavfile
    >>> import noise_footprint_creation
    >>> import wdwf
    >>> import speech_intelligibility_metrics

    >>> cs_sr, clean_signal_array = wavfile.read("sp23_16kHz.wav")
    >>> ns_sr, noisy_signal_array = wavfile.read("sp23_car_sn05_16kHz.wav")
    >>> print(cs_sr, ns_sr, clean_signal_array.shape, noisy_signal_array.shape) -> (16000, 16000, (42418,), (42418,))

    >>> noise_footprint_array = noise_footprint_creation.best_esnr_channel_selection(noisy_signal_array, ns_sr, 1, -7.0, 2)

    >>> denoised_signal_array = wdwf.denoising(noisy_signal_array, noise_footprint_array, ns_sr, 4, True, False, 6, 2.0, 2.0, 1.0, 0.22, 0.9)

    >>> print(noise_footprint_array.shape, denoised_signal_array.shape) -> ((6578,), (42432,))

    >>> msc = speech_intelligibility_metrics.metric_computation(clean_signal_array, denoised_signal_array, cs_sr, 1)

    >>> csii = speech_intelligibility_metrics.metric_computation(clean_signal_array, denoised_signal_array, cs_sr, 2)

    >>> csii_high = speech_intelligibility_metrics.metric_computation(clean_signal_array, denoised_signal_array, cs_sr, 3)

    >>> csii_mid = speech_intelligibility_metrics.metric_computation(clean_signal_array, denoised_signal_array, cs_sr, 4)

    >>> csii_low = speech_intelligibility_metrics.metric_computation(clean_signal_array, denoised_signal_array, cs_sr, 5)

    >>> i3 = speech_intelligibility_metrics.metric_computation(clean_signal_array, denoised_signal_array, cs_sr, 6)

    >>> print(msc, csii, csii_high, csii_mid, csii_low, i3) -> (36.9, 60.4, 72.0, 38.7, 16.0, 66.7)

-------------------------------------------------------------------------------------------------------------------------------------
speech_recognition_metrics.py:

(1) Module that is being used for computing the Automatic Word Recognition Precision (AWRP) metric given an Automatic Speech Recognition (ASR) system, the word that has to be recognized, and the single-channel WAV file of the recorded word.

(2) The value of the AWRP metric is either 0 or 100.

(3) There is 1 main function: awrp_computation(), which is called in the automated_denoising_and_evaluation.py module.

(4) There is detailed documentation inside the code.

(5) Example:

    >>> from RappCloud import RappPlatformAPI
    >>> import speech_recognition_metrics

    >>> rapp_platform_api_instance = RappPlatformAPI()
    >>> audio_source = "headset"
    >>> language = "en"
    >>> vocabulary = ["thirsty", "thirteen"]
    >>> sphinx_parameters = (audio_source, language, vocabulary)
    >>> google_parameters = (audio_source, language)
    >>> sphinx_id = 1
    >>> google_id = 2
    >>> recognition_word = "thirsty"

    >>> sphinx_awrp, sphinx_error = speech_recognition_metrics.awrp_computation(rapp_platform_api_instance, sphinx_id, sphinx_parameters, "single-channel_thirsty-med-3_manos_16kHz.wav", recognition_word)

    >>> google_awrp, google_error = speech_recognition_metrics.awrp_computation(rapp_platform_api_instance, google_id, google_parameters, "single-channel_thirsty-med-3_manos_16kHz.wav", recognition_word)

    >>> print(sphinx_awrp, google_awrp, sphinx_error, google_error) -> (0.0, 100.0, u'', u'')

-------------------------------------------------------------------------------------------------------------------------------------
automated_denoising_and_evaluation.py:

(1) Module that is being used for performing an automated denoising procedure (creating a noise footprint and denoising) and evaluating its result using either a Speech Intelligibility Metric (SIM) or an Automatic Speech Recognition (ASR) system.

(2) There are 2 main functions: wdwfs_and_objective_metrics() and wdwfs_and_speech_recognition(), which are called in the optimization.py and tester.py modules.

(3) wdwfs_and_objective_metrics() can deal only with single-channel signals.

(4) wdwfs_and_speech_recognition() can deal only with multi-channel signals.

(5) There is detailed documentation inside the code.

(6) Example:

    >>> from scipy.io import wavfile
    >>> import automated_denoising_and_evaluation
    >>> from RappCloud import RappPlatformAPI

    >>> cs_sr, clean_signal_array = wavfile.read("sp23_16kHz.wav")
    >>> ns_sr, noisy_signal_array = wavfile.read("sp23_car_sn05_16kHz.wav")
    >>> print(cs_sr, ns_sr, clean_signal_array.shape, noisy_signal_array.shape) -> (16000, 16000, (42418,), (42418,))

    >>> msc = automated_denoising_and_evaluation.wdwfs_and_objective_metrics(clean_signal_array, noisy_signal_array, cs_sr, False, 1, 4, True, False, 6, -7.0, 2.0, 2.0, 1.0, 0.22, 0.9)

    >>> i3 = automated_denoising_and_evaluation.wdwfs_and_objective_metrics(clean_signal_array, noisy_signal_array, cs_sr, False, 6, 4, True, False, 6, -7.0, 2.0, 2.0, 1.0, 0.22, 0.9)

    >>> print(msc, i3) -> (37.0, 66.7)

    >>> msc, denoising_time_1, evaluation_time_1 = automated_denoising_and_evaluation.wdwfs_and_objective_metrics(clean_signal_array, noisy_signal_array, cs_sr, True, 1, 4, True, False, 6, -7.0, 2.0, 2.0, 1.0, 0.22, 0.9)

    >>> i3, denoising_time_2, evaluation_time_2 = automated_denoising_and_evaluation.wdwfs_and_objective_metrics(clean_signal_array, noisy_signal_array, cs_sr, True, 6, 4, True, False, 6, -7.0, 2.0, 2.0, 1.0, 0.22, 0.9)

    >>> denoising_time_1 = round(denoising_time_1, 2) # In seconds.
    >>> denoising_time_2 = round(denoising_time_2, 2) # In seconds.
    >>> evaluation_time_1 = round(evaluation_time_1, 2) # In seconds.
    >>> evaluation_time_2 = round(evaluation_time_2, 2) # In seconds.
    >>> print(denoising_time_1, denoising_time_2, evaluation_time_1, evaluation_time_2) -> (0.42, 0.44, 0.67, 12.03)

    >>> sampling_rate, noisy_signal_array = wavfile.read("thirsty-med-3_manos_16kHz.wav")
    >>> print(sampling_rate, noisy_signal_array.shape) -> (16000, (46421, 4))

    >>> rapp_platform_api_instance = RappPlatformAPI()
    >>> audio_source = "headset"
    >>> language = "en"
    >>> vocabulary = ["thirsty", "thirteen"]
    >>> sphinx_parameters = (audio_source, language, vocabulary)
    >>> google_parameters = (audio_source, language)
    >>> sphinx_id = 1
    >>> google_id = 2
    >>> recognition_word = "thirsty"

    >>> sphinx_awrp, sphinx_error = automated_denoising_and_evaluation.wdwfs_and_speech_recognition(rapp_platform_api_instance, sphinx_id, sphinx_parameters, recognition_word, noisy_signal_array, sampling_rate, False, 1, 4, True, False, 6, -7.0, 2.0, 2.0, 1.0, 0.22, 0.9)

    >>> google_awrp, google_error = automated_denoising_and_evaluation.wdwfs_and_speech_recognition(rapp_platform_api_instance, google_id, google_parameters, recognition_word, noisy_signal_array, sampling_rate, False, 1, 4, True, False, 6, -7.0, 2.0, 2.0, 1.0, 0.22, 0.9)

    >>> print(sphinx_awrp, google_awrp, sphinx_error, google_error) -> (0.0, 0.0, u'', u'')

    >>> sphinx_awrp, sphinx_error, denoising_time_1, evaluation_time_1 = automated_denoising_and_evaluation.wdwfs_and_speech_recognition(rapp_platform_api_instance, sphinx_id, sphinx_parameters, recognition_word, noisy_signal_array, sampling_rate, True, 1, 4, True, False, 6, -7.0, 2.0, 2.0, 1.0, 0.22, 0.9)

   >>> google_awrp, google_error, denoising_time_2, evaluation_time_2 = automated_denoising_and_evaluation.wdwfs_and_speech_recognition(rapp_platform_api_instance, google_id, google_parameters, recognition_word, noisy_signal_array, sampling_rate, True, 1, 4, True, False, 6, -7.0, 2.0, 2.0, 1.0, 0.22, 0.9)

   >>> denoising_time_1 = round(denoising_time_1, 2) # In seconds.
   >>> denoising_time_2 = round(denoising_time_2, 2) # In seconds.
   >>> evaluation_time_1 = round(evaluation_time_1, 2) # In seconds.
   >>> evaluation_time_2 = round(evaluation_time_2, 2) # In seconds.
   >>> print(denoising_time_1, denoising_time_2, evaluation_time_1, evaluation_time_2) -> (1.56, 1.53, 5.85, 1.81)

-------------------------------------------------------------------------------------------------------------------------------------
dataset_configuration.py:

(1) Module that is being for forming a dataset of signals in an effective way.

(2) There are 2 main functions: dataset_configuration_for_objective_metrics() and dataset_configuration_for_speech_recognition(), which are called in the optimization.py and tester.py modules.

(3) The Dataset_for_Objective_Metrics directory has the appropriate format for the dataset_configuration_for_objective_metrics().

(4) The Dataset_for_Speech_Recognition directory has the appropriate format for the dataset_configuration_for_speech_recognition().

(5) There is detailed documentation inside the code.

(6) Examples:

    >>> import dataset_configuration

    >>> strings_dataset, arrays_dataset = dataset_configuration.dataset_configuration_for_objective_metrics("Dataset_for_Objective_Metrics")

    >>> strings_dataset, arrays_dataset, transcription = dataset_configuration.dataset_configuration_for_speech_recognition("Dataset_for_Speech_Recognition")

-------------------------------------------------------------------------------------------------------------------------------------
optimization.py:

(1) Module that is being used for applying a Random Restart and Stochastic Hill Climbing (RRSHC) optimization algorithm so as to find out the optimal parameters of an automated denoising procedure.

(2) The RRSHC is applied on a whole dataset and it is based either on a Speech Intelligibility Metric (SIM) or on an Automatic Speech Recognition (ASR) system.

(3) The format of the dataset directory depends on the evaluation way (SIM or ASR system), as it is explained in the dataset_configuration.py module.

(4) There is 1 main function: optimization_algorithm()

(5) There is detailed documentation inside the code.

(6) Example:

    >>> import optimization

    >>> best_parameters_msc = optimization.optimization_algorithm("Dataset_for_Objective_Metrics", 1, -1, -1, 1, 3, True, False, 6, 10, 1)

    >>> best_parameters_sphinx = optimization.optimization_algorithm("Dataset_for_Speech_Recognition", 7, 1, 3, 1, 3, True, False, 6, 10, 1)

    >>> print(best_parameters_msc, best_parameters_sphinx) -> ([-7.0, 2.0, 2.0, 0.1, 0.22, 0.9], [-7.0, 2.0, 2.0, 0.1, 0.22, 0.9])

-------------------------------------------------------------------------------------------------------------------------------------
tester.py:

(1) Module that is being used for the evaluation of the performance of an automated denoising algorithm upon a whole dataset.

(2) The evaluation is based either on a Speech Intelligibility Metric (SIM) or on an Automatic Speech Recognition (ASR) system.

(3) The format of the dataset directory depends on the evaluation way (SIM or ASR system), as it is explained in the dataset_configuration.py module.

(4) There is 1 main function: testing().

(5) There is detailed documentation inside the code.

(6) Example:

    >>> import tester

    >>> mean_msc, total_den_time_1, total_eval_time_1 = tester.testing("Dataset_for_Objective_Metrics", 1, -1, -1, 1, 3, True, False, 6, [-7.0, 2.0, 2.0, 0.1, 0.22, 0.9])

    >>> mean_i3, total_den_time_2, total_eval_time_2 = tester.testing("Dataset_for_Objective_Metrics", 6, -1, -1, 1, 3, True, False, 6, [-7.0, 2.0, 2.0, 0.1, 0.22, 0.9])

    >>> mean_sphinx, total_den_time_3, total_eval_time_3 = tester.testing("Dataset_for_Speech_Recognition", 7, 1, 3, 1, 3, True, False, 6, [-7.0, 2.0, 2.0, 0.1, 0.22, 0.9])

    >>> mean_google, total_den_time_4, total_eval_time_4 = tester.testing("Dataset_for_Speech_Recognition", 7, 2, 3, 1, 3, True, False, 6, [-7.0, 2.0, 2.0, 0.1, 0.22, 0.9])

    >>> print(mean_msc, mean_i3, mean_sphinx, mean_google) -> (38.7, 88.5, 41.7, 50.0)
    >>> print(total_den_time_1, total_den_time_2, total_den_time_3, total_den_time_4) -> (0.5, 0.5, 1.0, 1.0) # In minutes.
    >>> print(total_eval_time_1, total_eval_time_2, total_eval_time_3, total_eval_time_4) -> (0.7, 12.7, 2.5, 1.1) # In minutes.
