def run(train_data, test_data):
    trainSamples = len(train_data)
    totalSamples = len(test_data)
    libData = 0
    conData = 0
    for eachSample in train_data:
        if eachSample["label"] == 0:
            libData = libData+1
        else:
            conData = conData+1
    if libData >= conData:
        output = 0
    else:
        output = 1

    correct = 0.0
    incorrect = 0.0
    for eachSample in test_data:
        if eachSample["label"] == 0:
            correct += 1
        else:
            incorrect += 1

    accuracy = correct/totalSamples
    return accuracy
