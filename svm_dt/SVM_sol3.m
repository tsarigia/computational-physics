%Εδω έχουμε πρόβλεψη συνεχούς τιμής (Path Loss),γι' αυτο θα χρησιμοποιήσουμε Regression Trees (fitrtree)
% Βήμα 1: Φόρτωση Δεδομένων
clear; clc; close all;

% Διαβάζουμε το αρχείο 
T = readtable('paper.xlsx');
data = table2array(T);

% Χωρίζουμε σε X (Features) και Y (Target - Path Loss)
X = data(:, 1:end-1); % Όλες οι στήλες εκτός από την τελευταία
Y = data(:, end);     % Η τελευταία στήλη (Path Loss in dB)

% Διαχωρισμός Train (70%) - Test (30%)
rng(42); % Για επαναληψιμότητα
cv = cvpartition(length(Y), 'HoldOut', 0.3);
XTrain = X(training(cv), :);
YTrain = Y(training(cv), :);
XTest = X(test(cv), :);
YTest = Y(test(cv), :);

fprintf('Δεδομένα Train: %d, Test: %d\n', length(YTrain), length(YTest));

% Α: Decision Tree Regressor (fitrtree)
fprintf('\n PART A: Decision Tree Regressor\n');
fprintf('%-10s %-10s %-10s %-10s\n', 'MaxDepth', 'MAE', 'RMSE', 'MAPE');


values = [20, 50, 70]; % Τιμές για MaxNumSplits

% Αρχικοποίηση μεταβλητών για να βρούμε το καλύτερο δεντρο αργοτερα.
minRMSE = inf; 
bestTree = [];
bestYPred = [];
bestErr = [];
bestR2 = 0;

for i = 1:length(values)
    currentSplit = values(i); 
    
    % Εκπαίδευση
    tree = fitrtree(XTrain, YTrain, 'MaxNumSplits', currentSplit);
    
    % Πρόβλεψη
    YPred = predict(tree, XTest);
    
    % Metrics
    err = YTest - YPred;
    MAE = mean(abs(err));
    RMSE = sqrt(mean(err.^2));
    MAPE = mean(abs(err ./ YTest));
    
    fprintf('%-10d %-10.4f %-10.4f %-10.4f\n', currentSplit, MAE, RMSE, MAPE);
    
    % Αποθήκευση του καλύτερου μοντέλου 
    if RMSE < minRMSE
        minRMSE = RMSE;
        bestTree = tree;
        bestYPred = YPred;
        bestErr = err;
        % Υπολογισμός του R-squared για το γράφημα
        bestR2 = 1 - sum(err.^2) / sum((YTest - mean(YTest)).^2);
    end
end

% ΟΠΤΙΚΟΠΟΙΗΣΗ
fprintf('\n-> Δημιουργία Γραφημάτων για το Βέλτιστο Δέντρο (RMSE=%.4f)...\n', minRMSE);
figure('Position', [100, 100, 1300, 400], 'Name', 'Decision Tree Analysis');

% Γράφημα 1: Feature Importance
subplot(1,2,1);
imp = predictorImportance(bestTree);
bar(imp, 'FaceColor', [0.2, 0.6, 0.8]);
xlabel('Features');
ylabel('Importance');
title('Ποια είσοδος επηρεάζει πιο πολύ;');
grid on;

% Γράφημα 2: Actual vs Predicted
subplot(1,2,2);
scatter(YTest, bestYPred, 10, 'b', 'filled', 'MarkerFaceAlpha', 0.4);
hold on;
plot([min(YTest) max(YTest)], [min(YTest) max(YTest)], 'r--', 'LineWidth', 2);
xlabel('Πραγματική Τιμή (Actual)');
ylabel('Πρόβλεψη (Predicted)');
title(['Actual vs Predicted (R^2 = ' num2str(bestR2, '%.3f') ')']);
grid on; axis equal;

%% --- ΜΕΡΟΣ Β: SVR (Linear vs RBF) ---
fprintf('\n PART B: SVR (Linear vs RBF)\n');
fprintf('%-10s %-5s %-10s %-10s %-10s\n', 'Kernel', 'C', 'MAE', 'RMSE', 'MAPE');

kernels = {'linear', 'gaussian'};
C_values = [1, 10, 100];

for i = 1:length(kernels)
    kernelName = kernels{i};
    
    for j = 1:length(C_values)
        currentC = C_values(j);
        
        % Εκπαίδευση SVR
        svmModel = fitrsvm(XTrain, YTrain, ...
            'KernelFunction', kernelName, ...
            'BoxConstraint', currentC, ...
            'Standardize', true);
        
        YPred = predict(svmModel, XTest);
        
        % Metrics
        err = YTest - YPred;
        MAE = mean(abs(err));
        RMSE = sqrt(mean(err.^2));
        MAPE = mean(abs(err ./ YTest));
        
        fprintf('%-10s %-5d %-10.4f %-10.4f %-10.4f\n', kernelName, currentC, MAE, RMSE, MAPE);
    end
end

%Comments

%Παρατηρούμε ότι καθώς αυξάνουμε την πολυπλοκότητα του δέντρου (MaxNumSplits), το σφάλμα μειώνεται σταθερά.Από τα 20 στα 70 splits, το RMSE πέφτει από 9.29 σε 8.61.
%Το δέντρο με τα 70 splits είναι το βέλτιστο από αυτά που δοκιμάστηκαν, πετυχαίνοντας την καλύτερη πρόβλεψη.
%Το RMSE "κολλάει" στο 12.44 ανεξάρτητα από το C. Αυτό σημαίνει ότι η σχέση των χαρακτηριστικών με το Path Loss δεν είναι γραμμική.Gaussian (RBF) Kernel: Το RMSE πέφτει στο 8.79 (με C=100).
%Η αύξηση του C (από 1 σε 100) βελτιώνει το μοντέλο RBF (το RMSE πέφτει από 9.02 σε 8.79), δείχνοντας ότι χρειαζόμαστε ένα πιο "αυστηρό" όριο απόφασης για να πιάσουμε τις λεπτομέρειες των δεδομένων.
%Aπο το γραφημα το 4ο χαρακτηριστικό (Feature 4) είναι το πιο σημαντικό (κυρίαρχο).
%Ενα ενδιαφέρον εύρημα είναι ότι το Decision Tree (RMSE 8.61) απέδωσε ελαφρώς καλύτερα από το βέλτιστο SVR (RMSE 8.79) στο συγκεκριμένο Test set