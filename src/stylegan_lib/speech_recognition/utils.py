from numpy.core.fromnumeric import argmax
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F

# define the transformations for training 

train_audio_transforms = nn.Sequential(
    # get the spectograms of the input audio
    torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=128),
    # apply frequency and time masking to eliminate the background noise
    torchaudio.transforms.FrequencyMasking(freq_mask_param=30),
    torchaudio.transforms.TimeMasking(time_mask_param=100)
)

# define the transformation for validation
valid_audio_transforms = torchaudio.transforms.MelSpectrogram()

class TextTransform:
    """Maps characters to integers and vice versa"""
    def __init__(self):
        alphabets = ['\'', ' ','a','b','c','d','e','f','g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
        self.char_map = {}
        self.index_map = {}
        for i in range (0,28):
          self.char_map[alphabets[i]] = i
          self.index_map[i] = alphabets[i]

    def toInt(self, text):
        """ convert a given string to the correseponding integers """
        # holds the converted integers
        conversion = []                         
        # loop over the given text 
        for letter in text:
          # used the character mapping dictionary to replace letters with their corresponding integers
          index = self.char_map[letter]
          print(index)
          conversion.append(index)
        return conversion

    def toString(self, indices):
        """ convert a given array of integers to its corresponding characters """
        # holds the converted string
        transcript = ""
        # loop over the given indices
        for index in indices:
          # use the index mapping dictionary to replace indecis with their corresponding letters
          transcript += self.index_map[index]
        return transcript

def dataPreprocessing(data, text_transform,  data_type="train"):
    # arrays to hold the training inputs batch
    spectrograms = []
    labels = []
    input_lengths = []
    label_lengths = []
    # loop over each audio in the data
    for (waveform, _, utterance, _, _, _) in data:
        # apply training preprocessing ( convert audio wave to spectograms and apply frequency and time masking )
        if data_type == 'train':
            spec = train_audio_transforms(waveform).squeeze(0).transpose(0, 1)
        # apply validation preprocessing ( convert audio wave to spectograms)
        elif data_type == 'valid':
            spec = valid_audio_transforms(waveform).squeeze(0).transpose(0, 1)
        else:
            raise Exception('data_type should be train or valid')
        # save the training data
        spectrograms.append(spec)
        # convert labels from string form to integer form ( ignoring cases)
        label = torch.Tensor(text_transform.toInt(utterance.lower()))
        labels.append(label)
        input_lengths.append(spec.shape[0]//2)
        label_lengths.append(len(label))
    # make batches of data
    spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).unsqueeze(1).transpose(2, 3)
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)

    # return the preprocessed data
    return spectrograms, labels, input_lengths, label_lengths


class IterMeter(object):
    """keeps track of total iterations"""
    def __init__(self):
        self.val = 0

    def step(self):
        self.val += 1

    def get(self):
        return self.val

# for inference we out the predicted transcript without collapses
def Inference(predictions, text_transform):
    
    max_args = torch.argmax(predictions, dim=2)
    decodes = []
    for i, args in enumerate(max_args):
        decode= []
        for j , index in enumerate(args):
            if index == 28:
                continue
            if j == 0:
                decode.append(index.item())
                continue
            if index != args[j-1]:
                decode.append(index.item())
        decodes.append(text_transform.toString(decode))
    return decodes


def GreedyDecoder(predictions, labels, label_lengths,text_transform ):
    # first select the highest probability character 
    arg_maxes = torch.argmax(predictions, dim=2)
    # arrays to hold the results
    decodes = []
    targets = []
    # loop over each data item in the batch
    for i, args in enumerate(arg_maxes):
        # first get the transcript text using texttransform mapping function
        #if not inference:
        targets.append(text_transform.toString(labels[i][:label_lengths[i]].tolist()))
        decode = []         # hold the decode of each example
        # loop over each time at each example, if new character exists add it to decode list else ignore it
        for j , index in enumerate(args):
            # if index is blanker ignore it
            if index == 28:
                continue
            # first index is added
            if j == 0:
                decode.append(index.item())
                continue
            # add any new index doesn't equal the previous one
            if index != args[j-1]:
                decode.append(index.item())

        # convert the indecies of each sample to string and save the them in decodes
        decodes.append(text_transform.toString(decode))
    # return the target strings
    return decodes, targets


def _levenshtein_distance(str1, str2):
    # solving levenshtein distance algorithm using dynamic programming approach

    # keep the 2 strings lengths
    m = len(str1)
    n = len(str2)
    # make str1 the longest and str2 the smallest
    if m < n:
      str1, str2 = str2, str1
      m, n = n, m

    # Create a table to store results of subproblems
    dp = [[0 for x in range(n + 1)] for x in range(m + 1)]
 
    # Fill the dp in bottom up approach
    for i in range(m + 1):
        for j in range(n + 1):
 
            # If first string is empty, insert all the second string characters
            if i == 0:
                dp[i][j] = j 
 
            # If second string is empty, insert all the first string characters
            elif j == 0:
                dp[i][j] = i 
 
            # If last characters are same, ignore last char
            elif str1[i-1] == str2[j-1]:
                dp[i][j] = dp[i-1][j-1]
 
            # If last character are different, select the minimum modifications requreid from ( insertion , deletion , substitution)
            # possibilities and find minimum
            else:
                dp[i][j] = 1 + min(dp[i][j-1], dp[i-1][j], dp[i-1][j-1])
 
    return dp[m][n]
 


def charErrorRate(trueValue, Prediction ):
    """
    Compute the levenshtein distance between given transcript and predicted one on character level
    """
    if len(trueValue) == 0:
        raise ValueError("Length of reference should be greater than 0.")

    # calculate the character errors using levenshtein distance measurement ( described above )
    edit_distance = _levenshtein_distance(trueValue, Prediction)

    # return the average character error rate
    return (float(edit_distance)/ len(trueValue))

def wordErrorRate(trueValue, Prediction):
    """
    Compute the levenshtein distance between given transcript and predicted one on word level
    """

    # get the text words to calculate the word errors using levenshtein distance measurement ( described above )
    real_words = trueValue.split(' ')
    prediction_words = Prediction.split(' ')

    edit_distance = _levenshtein_distance(real_words, prediction_words)

    # return the average word error rate
    return(float(edit_distance)/len(real_words))
