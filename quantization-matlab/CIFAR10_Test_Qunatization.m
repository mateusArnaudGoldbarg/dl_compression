
load('DataImagesCIFAR10.mat')


inputSize = [32 32 3];


numClasses = length(categories(YTrain));



pTrain = .75; %Porcent of train
cvObj = cvpartition(YTrain,'HoldOut',1-pTrain);

VectorTrainingData = XTrain(:,:,:,cvObj.training);
VectorTrainingLabel =  YTrain(cvObj.training);

VectorValidationData = XTrain(:,:,:,cvObj.test);
VectorValidationLabel = YTrain(cvObj.test);  


layers = [
    imageInputLayer(inputSize,'Mean', mean(VectorTrainingData,4),'Name', 'Input1')
    
    convolution2dLayer(3,8,'Padding','same','Name', 'Conv1')
    batchNormalizationLayer('Name','BatchNorm1');
    reluLayer('Name','ReLu1');
    maxPooling2dLayer(2,'Stride',2,'Name', 'MaxPooling1')
    
    convolution2dLayer(3,16,'Padding','same','Name', 'Conv2')
    batchNormalizationLayer('Name','BatchNorm2');
    reluLayer('Name','ReLu2');
    maxPooling2dLayer(2,'Stride',2,'Name', 'MaxPooling2')
    
    convolution2dLayer(3,32,'Padding','same','Name', 'Conv3')
    batchNormalizationLayer('Name','BatchNorm3');
    reluLayer('Name','ReLu3');
    maxPooling2dLayer(2,'Stride',2,'Name', 'MaxPooling3')
    
    fullyConnectedLayer(32, 'Name', 'FC1');
    batchNormalizationLayer('Name','BatchNorm4');
   dropoutLayer(0.4,'Name','Drop1');

   fullyConnectedLayer(16, 'Name', 'FC2');
   batchNormalizationLayer('Name','BatchNorm5');
   dropoutLayer(0.4,'Name','Drop2');
    
    fullyConnectedLayer(numClasses,'Name', 'FC3')
    softmaxLayer('Name','SoftMax1')];
          
          


lgraph = layerGraph(layers);
dlnet = dlnetwork(lgraph);
dlnet2 = dlnet;

dlnetQAnt = dlnet;
dlnetAnt = dlnet2;

numEpochs =10;

miniBatchSize = 64;
%miniBatchSize = 128;

learnRate = 0.01;
gradDecay = 0.75;
sqGradDecay = 0.95;

l2Regularization = 0.0001;


averageGrad = [];
averageSqGrad = [];

numObservations = numel(VectorTrainingLabel);
numIterationsPerEpoch = floor(numObservations./miniBatchSize);

iteration = 0;

%Loop over epochs.
bestdlnet  = dlnet;
bestdlnet2  = dlnet2;
bestAV = 0;
bestA = 0;
 c1 = [2 6 10 14];
 c2 = [18 20 22 24];
 
 cs1 = {'Conv1';'Conv2';'Conv3'};
 cs2 = {'FC1';'FC2';'FC3'};


 N = 10;
 
VAB1 = zeros(1,numEpochs);
VLB1 = zeros(1,numEpochs);

VAT1 = zeros(1,numEpochs*numIterationsPerEpoch);
VLT1 = zeros(1,numEpochs*numIterationsPerEpoch);

classes = categories(VectorTrainingLabel);
VAVB = zeros(1,numEpochs);

hk=0;
for epoch = 1:numEpochs
    
    %disp(epoch)
    
    % Shuffle data.
    idx = randperm(numel(VectorTrainingLabel));
    VectorTrainingData = VectorTrainingData(:,:,:,idx);
    VectorTrainingLabel = VectorTrainingLabel (idx);
    
     % Loop over mini-batches.
     lossV = zeros(1,numIterationsPerEpoch);
     accuracyV = zeros(1,numIterationsPerEpoch);
     
    for i = 1:numIterationsPerEpoch
          hk = hk+1;
        iteration = iteration + 1;
        %disp(iteration)
        
        % Read mini-batch of data and convert the labels to dummy
        % variables.
        idx = (i-1)*miniBatchSize+1:i*miniBatchSize;
        X = VectorTrainingData(:,:,:,idx);
        
        Y = zeros(numClasses, miniBatchSize, 'single');
        for c = 1:numClasses
            Y(c,VectorTrainingLabel(idx)==classes(c)) = 1;
        end
        
         % Convert mini-batch of data to dlarray.
        dlX = dlarray(single(X),'SSCB');
        

        
          dlnet = dlnet2;
         
       
        for j=1:length(cs1)
            idxW = dlnet.Learnables.Layer == cs1{j} & dlnet.Learnables.Parameter == "Weights"; 
            idxB = dlnet.Learnables.Layer == cs1{j} & dlnet.Learnables.Parameter == "Bias"; 
            
             W = extractdata(dlnet.Learnables.Value{idxW});
             B = extractdata(dlnet.Learnables.Value{idxB});

             [WQS,BQS,WQ,BQ] = WeightsQuantizer(W,B,N);   
             
             dlnet.Learnables.Value{idxW}=dlarray(single(WQS));
             dlnet.Learnables.Value{idxB}=dlarray(single(BQS));

        end
        
        for j=1:length(cs2)
            idxW = dlnet.Learnables.Layer == cs2{j} & dlnet.Learnables.Parameter == "Weights"; 
            idxB = dlnet.Learnables.Layer == cs2{j} & dlnet.Learnables.Parameter == "Bias"; 
            
             W = extractdata(dlnet.Learnables.Value{idxW});
             B = extractdata(dlnet.Learnables.Value{idxB});

             [WQS,BQS,WQ,BQ] = WeightsQuantizer(W,B,N);   
             
             dlnet.Learnables.Value{idxW}=dlarray(single(WQS));
             dlnet.Learnables.Value{idxB}=dlarray(single(BQS));

        end
        
        
       
        % Evaluate the model gradients, state, and loss using dlfeval and the
        % modelGradients function and update the network state.
         [gradients,state,loss] = dlfeval(@modelGradients,dlnet,dlX,Y);
         
       idx = dlnet.Learnables.Parameter == "Weights";
       gradients(idx,:) = dlupdate(@(g,w) g + l2Regularization*w, gradients(idx,:), dlnet.Learnables(idx,:));
         
         dlnet2.State = state;
         
         dlYPred = predict(dlnet,dlX);
         [~,idx1] = max(extractdata(dlYPred),[],1);
         [~,idx2] = max(Y);
         YPred = categorical(classes(idx1));
         Y = categorical(classes(idx2));
         accuracy = mean(YPred==Y);

        
       % Update the network parameters using the Adam optimizer.
        [dlnet2,averageGrad,averageSqGrad] = adamupdate(dlnet2,gradients,averageGrad,averageSqGrad,iteration,learnRate,gradDecay,sqGradDecay);
        
        accuracyV(i) = accuracy;
        lossV(i) = double(gather(extractdata(loss)));
        
         VAT1(hk) = accuracyV(i);
         VLT1(hk) = lossV(i);
        
        
    end
    
    VAB1(epoch) = mean(accuracyV);
    VLB1(epoch) = mean(lossV);

    
    dlXTest = dlarray(single(VectorValidationData),'SSCB');
    dlYPred = predict(dlnet,dlXTest);
    [~,idx] = max(extractdata(dlYPred),[],1);
    YPred = categorical(classes(idx));
    accuracyValidation = mean(YPred==VectorValidationLabel);
    
    VAVB(epoch)  = accuracyValidation;
    
   RoundAccuracyValidation =  round(accuracyValidation*1000)/1000;
    MeanAccuracyV = round(mean(accuracyV)*1000)/1000;
    
     
    %if (bestA < mean(accuracyV) && bestAV < accuracyValidation)
    %if (mean([bestA bestAV]) < mean([mean(accuracyV) accuracyValidation]))
   %if (bestAV <accuracyValidation && bestA < mean(accuracyV))
    %if (accuracyValidation >= .99 && mean(accuracyV) >= .99)
      if (RoundAccuracyValidation >= .99 && MeanAccuracyV >= .98)
        bestA = mean(accuracyV);
        bestAV = accuracyValidation;
        bestdlnet = dlnet;
        bestdlnet2 = dlnet2;
        epochBest = epoch;
        disp('best');
    end
    
     disp([epoch iteration]);
    disp(mean(lossV));
    disp([MeanAccuracyV  RoundAccuracyValidation]);
end

%Validation
dlXTest = dlarray(single(VectorValidationData),'SSCB');
dlYPred = predict(bestdlnet,dlXTest);
[~,idx] = max(extractdata(dlYPred),[],1);
YPred = categorical(classes(idx));
accuracyValidation = mean(YPred==VectorValidationLabel)


%Test
dlXTest = dlarray(single(XValidation),'SSCB');
dlYPred = predict(bestdlnet,dlXTest);
[~,idx] = max(extractdata(dlYPred),[],1);
YPred = categorical(classes(idx));
accuracyValidation = mean(YPred==YValidation)
          

