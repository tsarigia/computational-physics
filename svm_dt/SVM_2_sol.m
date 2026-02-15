% MAGIC telescope: Διαχωρίζουμε σήματα που προέρχονται από ακτίνες γάμμα (g - gamma) και απο κοσμική ακτινοβολία των αδρονίων (h - hadron).
%Εχουμε οτι οι 10 πρώτες στήλες είναι τα αριθμητικά χαρακτηριστικά (fLength, fWidth, κλπ.) και η 11η στήλη είναι η κλάση.
clc,clearvars
% Φορτώνουμε το αρχείο data
data = readtable('magic04.data', 'FileType', 'text', 'ReadVariableNames', false);

% Ορίζουμε τα δεδομένα εισόδου (X) και τις ετικέτες (Y)
X = table2array(data(:, 1:10)); % Οι 10 πρώτες στήλες είναι τα χαρακτηριστικά
rawLabels = table2array(data(:, 11)); % Η 11η στήλη είναι η κλάση (g/h)

% 2. Μετατροπή Ετικέτας σε Δυαδική (Gamma=1, Hadron=0)
Y = double(strcmp(rawLabels, 'g'));

% 3. Διαχωρισμός σε Train (70%) και Test (30%)
rng(1); % Για επαναληψιμότητα (Reproducibility)
cv = cvpartition(length(Y), 'HoldOut', 0.3); % Κρατάμε 30% για Test

XTrain = X(training(cv), :);
YTrain = Y(training(cv), :);

XTest = X(test(cv), :);
YTest = Y(test(cv), :);

%Υπολογίζουμε μέσο όρο (mu) και τυπική απόκλιση (sigma)
mu = mean(XTrain);
sigma = std(XTrain);

XTrained_scaled = (XTrain - mu) ./ sigma;
XTest_scaled = (XTest - mu) ./ sigma;

% Βήμα 2: Πειραματισμός με το C (BoxConstraint)
C_values = [0.1, 1, 10, 100]; % Οι τιμές που θα δοκιμάσουμε
results = []; % Εδώ θα αποθηκεύουμε τα αποτελέσματα

fprintf('Ξεκινάει ο πειραματισμός...\n');
fprintf('%-10s %-5s %-10s %-10s %-10s\n', 'Kernel', 'C', 'Accuracy', 'Precision', 'Recall');

for i = 1:length(C_values)
    C = C_values(i);
    
    % --- Μοντέλο 1: Linear Kernel ---
    % 'Standardize', true: ΑΠΑΡΑΙΤΗΤΟ στα SVM
    svmLin = fitcsvm(XTrain, YTrain, 'KernelFunction', 'linear', ...
        'BoxConstraint', C, 'Standardize', true);
    
    predLin = predict(svmLin, XTest);
    
    % Υπολογισμός Metrics (Linear)
    confLin = confusionmat(YTest, predLin); % [TN FP; FN TP]
    TP = confLin(2,2); TN = confLin(1,1); FP = confLin(1,2); FN = confLin(2,1);
    
    accLin = (TP + TN) / sum(confLin(:));
    precLin = TP / (TP + FP);
    recLin = TP / (TP + FN);
    
    fprintf('%-10s %-5.1f %-10.4f %-10.4f %-10.4f\n', 'Linear', C, accLin, precLin, recLin);
    
    % Μοντέλο 2: RBF (Gaussian) Kernel
    % KernelScale 'auto': για να βρεί μόνο του το σωστό sigma
    svmRBF = fitcsvm(XTrain, YTrain, 'KernelFunction', 'gaussian', ...
        'BoxConstraint', C, 'Standardize', true, 'KernelScale', 'auto');
    
    predRBF = predict(svmRBF, XTest);
    
    % Υπολογισμός Metrics (RBF)
    confRBF = confusionmat(YTest, predRBF);
    TP = confRBF(2,2); TN = confRBF(1,1); FP = confRBF(1,2); FN = confRBF(2,1);
    
    accRBF = (TP + TN) / sum(confRBF(:));
    precRBF = TP / (TP + FP);
    recRBF = TP / (TP + FN);
    
    fprintf('%-10s %-5.1f %-10.4f %-10.4f %-10.4f\n', 'RBF', C, accRBF, precRBF, recRBF);
    fprintf('----------------------------------------------------------\n');
end

% Βήμα 3: Σύγκριση & Συμπέρασμα
% Διαλέγουμε ενδεικτικά το RBF με C=10 για να δούμε τον πίνακα σύγχυσης
finalSVM = fitcsvm(XTrain, YTrain, 'KernelFunction', 'gaussian', ...
    'BoxConstraint', 10, 'Standardize', true, 'KernelScale', 'auto');
finalPred = predict(finalSVM, XTest);

figure;
cm = confusionchart(YTest, finalPred);
cm.Title = 'Confusion Matrix - Best SVM (RBF, C=10)';
cm.XLabel = 'Predicted (0=Hadron, 1=Gamma)';
cm.YLabel = 'Actual Class';

%Σχόλια
%Linear Kernel: Η απόδοση "κολλάει" στο ~78.7% ανεξάρτητα από το C. Αυτό δείχνει ότι μια ευθεία γραμμή (υπερεπίπεδο) δεν μπορεί να ξεχωρίσει αποτελεσματικά τα σήματα Gamma από τα Hadrons.
%RBF Kernel: Πετυχαίνει σημαντικά υψηλότερη ακρίβεια (~85.3%). Αυτό αποδεικνύει ότι η σχέση των δεδομένων είναι μη-γραμμική
%C είναι πολύ μεγάλο (100), τιμωρούμε αυστηρά κάθε λάθος στο Trainingset(overfitting)