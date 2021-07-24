import os
import sys
import torch
import torchaudio
import torch.nn as nn
import torch.optim as optim
from scipy.io import wavfile
import torch.utils.data as data
import torch.nn.functional as F
from .model import SpeechRecognition
from .utils import IterMeter, TextTransform, dataPreprocessing , GreedyDecoder, charErrorRate , wordErrorRate, Inference


# define the hyperparameters
BatchSize = 20
epochs = 30
lr = 0.0001
numWorkers = 1

device = "cuda" if torch.cuda.is_available() else "cpu"

NoClasses = 29
NoCNNs = 3
NoRNNs = 5
Nofeatrues = 64
RNNCells = 512


# training function
def train(model, device, train_loader, criterion, optimizer, scheduler, epoch, iter_meter):

    model.train()
    data_len = len(train_loader.dataset)

    # loop over the data batches
    for batch_idx, _data in enumerate(train_loader):
            # get the data
            spectrograms, labels, input_lengths, label_lengths = _data 
            spectrograms, labels = spectrograms.to(device), labels.to(device)

            optimizer.zero_grad()
            # input data to model
            output = model(spectrograms)  # (batch, time, n_class)
            # apply log softmax to get the probability of each character at each time in the input 
            output = F.log_softmax(output, dim=2)
            output = output.transpose(0, 1) # (time, batch, n_class)

            # calculate losses and apply backprogation to update the model weights
            loss = criterion(output, labels, input_lengths, label_lengths)
            loss.backward()
            optimizer.step()
            scheduler.step()        # update lr
            iter_meter.step()       # move to the next data item

            # save the model checkpoints
            torch.save(
            {'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss}, 
            "./home/monda/Documents/ForthYear/gp/pytorchSpeech")

            if batch_idx % 100 == 0 or batch_idx == data_len:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(spectrograms), data_len,
                    100. * batch_idx / len(train_loader), loss.item()))



def Validate(model, device, validation_loader, criterion, text_transform):
    print('\n ----------------------------------- evaluating -------------------------------------------------')
    model.eval()
    # hold the losses
    test_loss = 0
    test_cer, test_wer = [], []
    
    with torch.no_grad():
        # loop over the validation dataset
        for i, _data in enumerate(validation_loader):
            # get the preprocessed data
            spectrograms, labels, input_lengths, label_lengths = _data 
            spectrograms, labels = spectrograms.to(device), labels.to(device)
            # enter the model
            output = model(spectrograms)  # (batch, time, n_class)
            # get the output probabilities
            output = F.log_softmax(output, dim=2)
            output = output.transpose(0, 1) # (time, batch, n_class)
            #  calculate validation losses
            loss = criterion(output, labels, input_lengths, label_lengths)
            test_loss += loss.item() / len(validation_loader)

            # get the predicted transcript
            decoded_preds, decoded_targets = GreedyDecoder(output.transpose(0, 1), labels, label_lengths, text_transform)
            # calculate the character error rate and word error rate for each batch
            for j in range(len(decoded_preds)):
                test_cer.append(charErrorRate(decoded_targets[j], decoded_preds[j]))
                test_wer.append(wordErrorRate(decoded_targets[j], decoded_preds[j]))

    # calculate the avg word error rate and character error rate over the validation data
    avg_cer = sum(test_cer)/len(test_cer)
    avg_wer = sum(test_wer)/len(test_wer)

    print('Test set: Average loss: {:.4f}, Average CER: {:4f} Average WER: {:.4f}\n'.format(test_loss, avg_cer, avg_wer))


def inference(model, device, inputPath, text_transform):
    print("------------------------------------ inference ---------------------------------------- \n")
    
    # read the input and convert it to spectogram
    model.eval()

    with torch.no_grad():
        # read the audio
        data, sample_rate = torchaudio.load(inputPath, normalization=True)
        # convert it to spectograms
        spectrograms = torchaudio.transforms.MelSpectrogram(sample_rate)(data).squeeze(0).transpose(0, 1) 
        spectrograms = nn.utils.rnn.pad_sequence([spectrograms], batch_first=True).unsqueeze(1).transpose(2, 3)
        spectrograms = spectrograms.to(device)
        # feed the model with the audio
        output = model(spectrograms)
        # get the output probabilities
        output = F.log_softmax(output, dim=2)
        output = output.transpose(0, 1)
        # get the predicted tanscript 
        decoded_preds = Inference(output.transpose(0, 1), text_transform)
        # print the predicted transcript
        print (" the transcript = " , decoded_preds , "\n\n")

def loadSavedModel(PATH):

    torch.manual_seed(7)
    text_transform = TextTransform()
    if not os.path.isdir("./data"):
        os.makedirs("./data")

    # load dataset
    traing_dataset  = torchaudio.datasets.LIBRISPEECH("./data", url='train-clean-100', download=True)
    testing_dataset = torchaudio.datasets.LIBRISPEECH("./data", url='test-clean', download=True)

    # data loaders
    training_loader = data.DataLoader(dataset=traing_dataset, batch_size=BatchSize, shuffle=True, collate_fn=lambda x: dataPreprocessing(x,text_transform , 'train'), num_workers = numWorkers, pin_memory = True)

    testing_loader = data.DataLoader(dataset=testing_dataset, batch_size=BatchSize, shuffle=False, collate_fn=lambda x: dataPreprocessing(x, text_transform, 'valid'), num_workers = numWorkers, pin_memory = True )

    # create the model
    model = SpeechRecognition( CNN_number=NoCNNs, RNN_number=NoRNNs, RNNCells=RNNCells, NoClasses=NoClasses, features=Nofeatrues, dropOut=0.1).to(device)

    print(model)
    print('Num Model Parameters', sum([param.nelement() for param in model.parameters()]))

    # create the loss and optimizer
    optimizer = optim.AdamW(model.parameters(), lr)
    criterion = nn.CTCLoss(blank=28).to(device)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr= lr,  steps_per_epoch=int(len(training_loader)), epochs=epochs, anneal_strategy='linear')

    iter_meter = IterMeter()
    checkpoint = torch.load(PATH)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    for e in range (epoch, epochs+1):
        Validate(model, device, testing_loader, criterion,text_transform , e, iter_meter)
        train(model, device, training_loader, criterion, optimizer, scheduler, e, iter_meter)

def startTrain():
    torch.manual_seed(7)

    text_transform = TextTransform()

    # create path for the dataset
    if not os.path.isdir("./data"):
        os.makedirs("./data")

    # load dataset
    traing_dataset  = torchaudio.datasets.LIBRISPEECH("./data", url='train-clean-100', download=True)
    testing_dataset = torchaudio.datasets.LIBRISPEECH("./data", url='test-clean', download=True)

    # data loaders
    training_loader = data.DataLoader(dataset=traing_dataset, batch_size=BatchSize, shuffle=True, collate_fn=lambda x: dataPreprocessing(x,text_transform , 'train'), num_workers = numWorkers, pin_memory = True)

    testing_loader = data.DataLoader(dataset=testing_dataset, batch_size=BatchSize, shuffle=False, collate_fn=lambda x: dataPreprocessing(x, text_transform, 'valid'), num_workers = numWorkers, pin_memory = True )

    # create the model
    model = SpeechRecognition( CNN_number=NoCNNs, RNN_number=NoRNNs, RNNCells=RNNCells, NoClasses=NoClasses, features=Nofeatrues, dropOut=0.1).to(device)

    print(model)
    print('Num Model Parameters', sum([param.nelement() for param in model.parameters()]))

    # create the loss and optimizer
    optimizer = optim.AdamW(model.parameters(), lr)
    criterion = nn.CTCLoss(blank=28).to(device)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr= lr,  steps_per_epoch=int(len(training_loader)), epochs=epochs, anneal_strategy='linear')

    iter_meter = IterMeter()

    # start training and validating
    for epoch in range(1, epochs + 1):
        train(model, device, training_loader, criterion, optimizer, scheduler, epoch, iter_meter)
        Validate(model, device, testing_loader, criterion,text_transform )
        


def main():
    if (len(sys.argv) < 2) :
        print("select the operation first")
        return
    operation = sys.argv[1]
    if operation == 'i' :
        text_transform = TextTransform()
        model = SpeechRecognition(CNN_number=NoCNNs, RNN_number=NoRNNs, RNNCells=RNNCells, NoClasses=NoClasses, features=Nofeatrues, dropOut=0.1).to(device)
        checkpoint = torch.load(sys.argv[2])
        model.load_state_dict(checkpoint['model_state_dict'])
        inference(model= model, device=device, inputPath=sys.argv[3], text_transform=text_transform)
    elif operation == 'l':
        loadSavedModel(sys.argv[2])
    else:
        startTrain()

main()
