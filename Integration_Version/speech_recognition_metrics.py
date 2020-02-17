#!/usr/bin/python
# encoding: utf-8

def word_reformation(input_word, language):

	"""A utility function for converting a numeric word to its literal representation.

Inputs:
  input_word: It may be "7" or "13" or "22".
  language: It may be "en" or "el" or "gr".

Outputs:
  output_word: The literal representation of the input_word."""

	if input_word == "7":
		output_word = "επτά"

	elif input_word == "22":
		output_word = "εικοσιδύο"

	elif input_word == "13":
		if language == "en":
			output_word = "thirteen"

		else:
			# This is wrong when Sphinx4 is being used because the "el" value represents both the English and the Greek language.
			# On the other hand, this function will never be called when Sphinx4 is being used. Thus, there is no practical problem.
			output_word = "δεκατρία"

	else:
		print("")
		print("speech_recognition_metrics.word_reformation():")
		print("The numeric word " + input_word + " is not supported in its literal representation!")
		print("")

	return output_word

def awrp_computation(rapp_platform_api_instance, asr_id, asr_parameters, input_wav_file, recognition_word):

	"""Computes the Automatic Word Recognition Precision (AWRP) metric [1].

Inputs:
  rapp_platform_api_instance: An instance of the class RappPlatformAPI of the module RappPlatformApi.py of the package RappCloud.

  asr_id: Determines the Automatic Speech Recognition (ASR) system that will be used.
  If asr_id = 1, Sphinx4 will be used [2]. Otherwise, Google Cloud Speech-to-Text will be used [3].

  asr_parameters: A tuple that contains some necessary parameters for the operation of the ASR systems.
  If asr_id = 1, these parameters are the audio_source, the language and the words as defined in [4].
  Otherwise, they are the audio_source and the language as defined in [5].

  input_wav_file: The name of the wav file that is going to be recognized. It must be single-channel.
  recognition_word: A string that indicates the word that has to be recognized.

Outputs:
  awrp: The value of the AWRP metric. It will be either 100.0 or 0.0.
  error: A Unicode string that normally has 0 length. Otherwise, it represents an error message.

[1] Tsardoulias, E.G., Thallas, A.G., Symeonidis, A.L., and Mitkas, P.A. (2016). Improving multilingual interaction for consumer
robots through signal enhancement in multichannel speech. JAES, volume 64, issue 7/8, pages 514-524, July 2016.

[2] Sphinx-4 is a state-of-the-art, speaker-independent, continuous speech recognition system written entirely in the Java programming
language (https://github.com/cmusphinx/sphinx4).

[3] Google Cloud Speech-to-Text is a speech-to-text conversion system powered by machine learning and it is available for short-form
or long-form audio (https://cloud.google.com/speech-to-text/).

[4] RAPP Speech Detection using Sphinx4
(https://github.com/rapp-project/rapp-platform/wiki/RAPP-Speech-Detection-using-Sphinx4).

[5] RAPP Speech Detection using Google API
(https://github.com/rapp-project/rapp-platform/wiki/RAPP-Speech-Detection-using-Google-API)."""

	if asr_id == 1:
		response = rapp_platform_api_instance.speechRecognitionSphinx(input_wav_file, *asr_parameters)

	else:
		response = rapp_platform_api_instance.speechRecognitionGoogle(input_wav_file, *asr_parameters)

	error = response["error"]

	if error != "":
		print(error)
		awrp_value = 0.0

	else:
		if len(response["words"]) == 0:
			awrp_value = 0.0

		else:
			language = asr_parameters[1]
			unicode_word = response["words"][0].lower()
			word = unicode_word.encode("utf-8")

			if ((word == "7") or (word == "13") or (word == "22")):
				word = word_reformation(word, language)

			if word == recognition_word:
				awrp_value = 100.0

			else:
				awrp_value = 0.0

				if asr_id != 1: # Google Cloud Speech-to-Text.
					number_of_alternatives = len(response["alternatives"])

					for alt_id in range (number_of_alternatives):
						unicode_alternative = response["alternatives"][alt_id][0].lower()
						alternative = unicode_alternative.encode("utf-8")

						if ((alternative == "7") or (alternative == "13") or (alternative == "22")):
							alternative = word_reformation(alternative, language)

						if alternative == recognition_word:
							awrp_value = 100.0
							break

	return [awrp_value, error]