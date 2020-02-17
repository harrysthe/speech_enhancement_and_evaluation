noise_footprint_creation.py:

(1) Module that is being used for creating the WAV file of the noise footprint given the WAV file of the noisy signal.

(2) There are 2 main functions: best_esnr_channel_selection() and esnr_and_common_zero_areas().

(3) best_esnr_channel_selection() can deal both with single-channel and multi-channel signals.

(4) esnr_and_common_zero_areas() can deal only with multi-channel signals.

(5) There is detailed documentation inside the code.

(6) Example:

    >>> import noise_footprint_creation

    >>> [best_esnr_channel, best_esnr] = noise_footprint_creation.best_esnr_channel_selection("sp23_car_sn05_16kHz.wav", -7.0, 2, 1)

    >>> [best_esnr_channel, best_esnr] = noise_footprint_creation.best_esnr_channel_selection("thirsty-med-3_manos_16kHz.wav", -7.0, 2, 1)

    >>> esnr_values = noise_footprint_creation.esnr_and_common_zero_areas("thirsty-med-3_manos_16kHz.wav", -7.0, 2, 2)

-----------------------------------------------------------------------------------------------------------------------------
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

-----------------------------------------------------------------------------------------------------------------------------
wdwf.py:

(1) Module that is being used for creating the WAV file of the denoised signal given the WAV files of the noisy signal and noise footprint.

(2) There is 1 main function: denoising().

(3) denoising() can deal only with single-channel signals.

(4) There is detailed documentation inside the code.

(5) Example:

    >>> import noise_footprint_creation
    >>> import wdwf

    >>> [best_esnr_channel, best_esnr] = noise_footprint_creation.best_esnr_channel_selection("sp23_car_sn05_16kHz.wav", -7.0, 2, 1)

    >>> denoised_signal = wdwf.denoising("sp23_car_sn05_16kHz.wav", "sp23_car_sn05_16kHz-noise_footprint-2.wav", 3, True, True, 6, 2.0, 2.0, 1.0, 0.22, 0.9)

    >>> denoised_signal = wdwf.denoising("sp23_car_sn05_16kHz.wav", "sp23_car_sn05_16kHz-noise_footprint-2.wav", 2, False, True, 7, 2.0, 2.0, 1.0, 0.22, 0.9)

    >>> denoised_signal = wdwf.denoising("sp23_car_sn05_16kHz.wav", "sp23_car_sn05_16kHz-noise_footprint-2.wav", 4, False, False, 6, 2.0, 2.0, 1.0, 0.22, 0.9)

-----------------------------------------------------------------------------------------------------------------------------
speech_intelligibility_metrics.py:

(1) Module that is being used for computing some speech intelligibility metrics (MSC, CSII, CSII-high, CSII-mid, CSII-low, I3) given the WAV files of the clean and denoised signals.

(2) There is 1 main function: metric_computation().

(3) metric_computation() can deal only with single-channel signals.

(4) There is detailed documentation inside the code.

(5) Example:

    >>> import noise_footprint_creation
    >>> import wdwf
    >>> import speech_intelligibility_metrics

    >>> [best_esnr_channel, best_esnr] = noise_footprint_creation.best_esnr_channel_selection("sp23_car_sn05_16kHz.wav", -7.0, 2, 1)

    >>> denoised_signal = wdwf.denoising("sp23_car_sn05_16kHz.wav", "sp23_car_sn05_16kHz-noise_footprint-2.wav", 4, True, False, 6, 2.0, 2.0, 1.0, 0.22, 0.9)

    >>> msc = speech_intelligibility_metrics.metric_computation("sp23_16kHz.wav", "sp23_car_sn05_16kHz-final_denoising.wav", 1)

    >>> csii = speech_intelligibility_metrics.metric_computation("sp23_16kHz.wav", "sp23_car_sn05_16kHz-final_denoising.wav", 2)

    >>> csii_high = speech_intelligibility_metrics.metric_computation("sp23_16kHz.wav", "sp23_car_sn05_16kHz-final_denoising.wav", 3)

    >>> csii_mid = speech_intelligibility_metrics.metric_computation("sp23_16kHz.wav", "sp23_car_sn05_16kHz-final_denoising.wav", 4)

    >>> csii_low = speech_intelligibility_metrics.metric_computation("sp23_16kHz.wav", "sp23_car_sn05_16kHz-final_denoising.wav", 5)

    >>> i3 = speech_intelligibility_metrics.metric_computation("sp23_16kHz.wav", "sp23_car_sn05_16kHz-final_denoising.wav", 6)

    >>> print(msc, csii, csii_h, csii_m, csii_l, i3) -> (37.0, 60.4, 72.0, 38.7, 16.0, 66.7)
