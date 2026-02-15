%% kNN Classification Example Using Real Physics Data (Ionosphere Radar Dataset)clear;
clc,clearvars; 
rng(0);

%% 1. Φόρτωση και Προετοιμασία
data = readtable('ionosphere.data','FileType','text');

% Μετατροπή της τελευταίας στήλης σε Categorical
Y = categorical(data.Var35);
X = data{:, 1:34};

%% 2. Διαχωρισμός Train - Test (Stratified)

cv = cvpartition(Y, 'HoldOut', 0.3);
Xtrain = X(training(cv), :);
Ytrain = Y(training(cv));

Xtest  = X(test(cv), :);
Ytest  = Y(test(cv));

%Αποθηκεύει τα ονόματα των κλάσεων (π.χ. 'b', 'g') για να τα χρησιμοποιήσουμε αργότερα στα γραφήματα.
classes = categories(Y);

% Ορίζουμε:
% 1. Τον καλύτερο αριθμό γειτόνων (NumNeighbors)
% 2. Την καλύτερη μετρική απόστασης (Distance)
% 3. Το αν θα χρησιμοποιήσει βάρη (DistanceWeight)
opts = struct('ShowPlots', false, 'Verbose', 0);

Mdl = fitcknn(Xtrain,Ytrain, ...
    'OptimizeHyperparameters', {'NumNeighbors', 'Distance', 'DistanceWeight'}, ...
    'HyperparameterOptimizationOptions', opts, ...
    'Standardize', true);

% Εμφάνιση των βέλτιστων παραμέτρων που βρέθηκαν
best_k = Mdl.NumNeighbors;
best_dist = Mdl.Distance;
fprintf('Neighbors (k): %d\n', best_k);
fprintf('Distance Metric: %s\n', best_dist);
fprintf('Weights: %s\n', Mdl.DistanceWeight);

%% 4. Πρόβλεψη
[Ypred, score] = predict(Mdl, Xtest);

%% 5. Αξιολόγηση (Accuracy)

accuracy = mean(Ypred == Ytest)*100;
fprintf('\n-> Optimizedd Test Accuracy: %.2f%% \n', accuracy);

%% Confusion Matrix

figure;
confusionchart(Ytest, Ypred);%Τα κουτάκια στη διαγώνιο (πάνω αριστερά & κάτω δεξιά) δείχνουν τις σωστές προβλέψεις.
title("kNN Classification — Ionosphere Physics Data");
%% 6. Metrics (Precision, Recall, F1)

cm = confusionmat(Ytest, Ypred, 'Order', classes);
TP = cm(1,1); FP = cm(2,1); FN = cm(1,2); TN = cm(2,2);
% Note: confusionmat rows = true, cols = predicted with 'Order' above.
% Adjust indexing if classes order differs. Here classes(1) is treated as positive below.

posIdx = 1;
negIdx = 2;

% Compute using counts aggregated per-class
TP_pos = cm(posIdx,posIdx);
FP_pos = sum(cm(:,posIdx)) - TP_pos;
FN_pos = sum(cm(posIdx,:)) - TP_pos;
TN_pos = sum(cm(:)) - TP_pos - FP_pos - FN_pos;

precision_pos = TP_pos / (TP_pos + FP_pos);%Όταν το μοντέλο μας λεει "Είναι Bad!", πόσο συχνά έχει δίκιο
recall_pos    = TP_pos / (TP_pos + FN_pos);%Από όλα τα πραγματικά "Bad" που υπάρχουν στα δεδομένα, πόσα καταφέραμε να βρούμε;
f1_pos        = 2 * (precision_pos * recall_pos) / (precision_pos + recall_pos);%Είναι ο συνδυασμός των δύο παραπάνω. Αν ένα μοντέλο έχει τέλειο Precision αλλά χάλια Recall, το F1 θα είναι χαμηλό

% For other class.Θεωρεί το 'g' (Good) ως το "Θετικό" γεγονός που ψάχνουμε.
TP_neg = cm(negIdx,negIdx);
FP_neg = sum(cm(:,negIdx)) - TP_neg;
FN_neg = sum(cm(negIdx,:)) - TP_neg;
precision_neg = TP_neg / (TP_neg + FP_neg);
recall_neg    = TP_neg / (TP_neg + FN_neg);
f1_neg        = 2 * (precision_neg * recall_neg) / (precision_neg + recall_neg);

% Macro averages
precision_macro = mean([precision_pos, precision_neg]);%Επειδή βγάλαμε δύο Precisions (ένα για τα 'b' και ένα για τα 'g'), εδώ βγάζουμε τον μέσο όρο τους.
recall_macro    = mean([recall_pos, recall_neg]);
f1_macro        = mean([f1_pos, f1_neg]);

% Display
fprintf('Confusion matrix (rows=true, cols=predicted):\n'); disp(cm);
fprintf('%s positive metrics:\n', string(classes(posIdx)));
fprintf('  Precision: %.4f   Recall: %.4f   F1: %.4f\n', precision_pos, recall_pos, f1_pos);
fprintf('%s positive metrics:\n', string(classes(negIdx)));
fprintf('  Precision: %.4f   Recall: %.4f   F1: %.4f\n', precision_neg, recall_neg, f1_neg);
fprintf('Macro averages -> Precision: %.4f  Recall: %.4f  F1: %.4f\n', precision_macro, recall_macro, f1_macro);

%% ROC, AUC
% ROC and AUC for positive class (classes(posIdx)), perfcurve expects scores for the positive class; find the matching column

[~,colPos] = ismember(classes(posIdx), Mdl.ClassNames);%classes(posIdx): Είναι η κλάση που ψάχνουμε (π.χ. 'b').Mdl.ClassNames: Είναι η λίστα με τα ονόματα των κλάσεων όπως τα έχει αποθηκεύσει
scoresPos = score(:, colPos);%Οι γραμμές αυτές βρίσκουν ποια στήλη του score έχει τις πιθανότητες για την κλάση 'b' (την 1η κλάση).

%Compute ROC curve and AUC.Προσομοίωση για να δει πόσο καλό είναι το μοντέλο σου σε όλα τα πιθανά σενάρια αυστηρότητας.
[Xroc, Yroc, T, AUC] = perfcurve(Ytest, scoresPos, classes(posIdx));%perfcurve: Υπολογίζει τα σημεία της καμπύλης.
fprintf('AUC (class %s): %.4f\n', string(classes(posIdx)), AUC);

% Optional: plot ROC
figure;
plot(Xroc, Yroc, 'LineWidth', 1.5);
xlabel('False positive rate'); ylabel('True positive rate');
title(sprintf('ROC (AUC = %.4f) for class %s', AUC, string(classes(posIdx))));
grid on;