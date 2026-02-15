% MAGIC telescope: Διαχωρίζουμε σήματα που προέρχονται από ακτίνες γάμμα (g - gamma) και απο κοσμική ακτινοβολία των αδρονίων (h - hadron).
%Oι 10 πρώτες στήλες είναι τα αριθμητικά χαρακτηριστικά (fLength, fWidth, κλπ.) και η 11η στήλη είναι η κλάση.
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

% B)% Λίγοι διαχωρισμοί (20), Μεγάλα φύλλα (100)
tree1 = fitctree(XTrain, YTrain, 'MaxNumSplits', 20, 'MinLeafSize', 100);
pred1 = predict(tree1, XTest);
acc1 = sum(pred1 == YTest) / length(YTest);
fprintf('1. Απλό Δέντρο (Splits=20, Leaf=100): Acc = %.2f%%\n', acc1*100);

% Μέτριοι διαχωρισμοί (100), Μέτρια φύλλα (10)
tree2 = fitctree(XTrain, YTrain, 'MaxNumSplits', 100, 'MinLeafSize', 10);
pred2 = predict(tree2, XTest);
acc2 = sum(pred2 == YTest) / length(YTest);
fprintf('2. Ισορροπημένο (Splits=100, Leaf=10): Acc = %.2f%%\n', acc2*100);

% Πολλοί διαχωρισμοί (500), Μικρά φύλλα (1) - οχι καλό διοτι Μαθαίνει θόρυβο
tree3 = fitctree(XTrain, YTrain, 'MaxNumSplits', 500, 'MinLeafSize', 1);
pred3 = predict(tree3, XTest);
acc3 = sum(pred3 == YTest) / length(YTest);
fprintf('3. Πολύπλοκο  (Splits=500, Leaf=1):   Acc = %.2f%%\n', acc3*100);

% Βήμα 3: Επιλογή & Τελικά Αποτελέσματα
% Βρίσκουμε ποιο είχε τη μεγαλύτερη ακρίβεια
[maxAcc, idx] = max([acc1, acc2, acc3]);
trees = {tree1, tree2, tree3};
bestTree = trees{idx};

fprintf('\n--> Επιλέχθηκε το Σενάριο %d με ακρίβεια %.2f%%\n', idx, maxAcc*100);

%Confusion Matrix (Πίνακας Σύγχυσης)
finalPred = predict(bestTree, XTest);
figure('Name', 'Confusion Matrix');
cm = confusionchart(YTest, finalPred);
cm.Title = ['Confusion Matrix - Best Model (Scenario ' num2str(idx) ')'];
cm.XLabel = 'Predicted (0=Hadron, 1=Gamma)';
cm.YLabel = 'Actual Class';

%  Σχεδίαση Δέντρου (Plot)
% Σχεδιάζουμε το δέντρο (χρησιμοποιούμε 'graph' για να είναι διαδραστικό)
view(bestTree, 'Mode', 'graph');