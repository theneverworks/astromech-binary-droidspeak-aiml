# astromech-binary-droidspeak
Entirely offline Windows based keyword spotting and binary 'droid speak' beep language translation for your Astromech droid.

A new OFFLINE version of https://github.com/theneverworks/astromech-binary-droidspeak-ibm-watson.

# Prerequisites

## Python
Known to support Python 3.6, other versions not tested but may work.

## Install Pronouncing
https://pypi.org/project/pronouncing/

## Install Sounddevice
https://pypi.org/project/sounddevice/

## Install Winsound
Built in.
https://docs.python.org/3/library/winsound.html

# Edits
## droid_speech.py

You must select which droid keyword you want. (R2, BB8, etc.)

Edit the table name to point to the included pretrained models. By default, the droid is called R4.

### Function speech_recognize_keyword_locally_from_microphone()

```
    # Creates an instance of a keyword recognition model. Update this to
    # point to the location of your keyword recognition model.
    model = speechsdk.KeywordRecognitionModel("r4.table")

    # The phrase your keyword recognition model triggers on.
    keyword = "R4"
 ```
 
You could/should adjust the filters that remove the keyword(s) from the payload before it is sent to Watson Assistant. This helps with accuracy but isn’t required. Some of these filters will emerge though reviewing the analytics in Watson Assistant.

` str = str.replace('are four ','').replace('Are Four ', '').replace('are Four ', '').replace('are 4 ', '').replace('our four ', '').replace('Our Four ', '').replace('our 4 ', '').replace('r 4 ', '').replace('R. for ', '')

# Running
`python.exe main.py`
