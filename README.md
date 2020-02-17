# speech_enhancement_and_evaluation
Implementation of Speech Enhancement Algorithms and Evaluation of their Performance

Speech enhancement/denoising algorithms, based on wavelet analysis and Wiener filter, are implemented and combined with noise footprint creation algorithms in order automated denoising processes to be developed. The performance of the automated denoising processes is evaluated using Speech Intelligibility Metrics (SIMs) and Automatic Speech Recognition (ASR) systems.

Standalone_Version contains a version of the basic modules of the project that is more suitable for experimenting with one wav file at a time and figuring out the effects of the various parameters of the automated denoising processes. Inside the directory you can find a detailed Readme.txt file, where it is shown how to use the modules of the Standalone_Version.

Integration_Version contains all the modules of the project and is more suitable for experimenting with whole datasets of wav files. You can try to optimize the automated denoising processes or to test a specific configuration of an automated denoising process. Note that Dataset_for_Objective_Metrics is used together with the SIMs, whereas Dataset_for_Speech_Recognition is combined with the ASR systems. Inside the directory you can find a detailed Readme.txt file, where it is shown how to use the modules of the Integration_Version.

The (detailed documented) code was written and tested on Python 2.7.17 and Ubuntu 18.04. Also, the signals of interest have a 16-bit PCM format and 16000 Hz sampling rate, while they may be either single-channel or multi-channel.

In order to take advantage of the ASR systems you have to download and install in your system the RAPP Cloud API from https://github.com/robotics-4-all/r4a_rapp_cloud_api_python.
