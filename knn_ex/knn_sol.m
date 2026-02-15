%% Example: kNN Classification on Physics Data
% We create a simple dataset based on orbital mechanics:
% If total energy E = (1/2)mv^2 + U is negative → bound orbit (class 0)
% If E is positive → unbound orbit (class 1)

clc, clearvars
rng(1); % for reproducibility

% 1. Δημιουργία Δεδομένων
N = 300;
mass = 1; 
velocity = 10*rand(N,1);          % Ταχύτητα 0-10 m/s
potential = -32 + 100*rand(N,1);  % Δυναμική Ενέργεια -32 έως 68 J

% Υπολογισμός Ολικής Ενέργειας
E = 0.5*mass.*velocity.^2 + potential;

% Labels: 0 = bound (Δέσμια), 1 = unbound (Αδέσμευτη)
labels = double(E > 0);

% Πίνακας Χαρακτηριστικών
X = [velocity potential];

%% --- 2. Διαχωρισμός σε Train και Test ---
cv = cvpartition(labels,'HoldOut',0.3); % 70% Train, 30% Test

Xtrain = X(training(cv),:);
Ytrain = labels(training(cv));
Xtest  = X(test(cv),:);
Ytest  = labels(test(cv));

%% 3. Εκπαίδευση Μοντέλου (ΒΕΛΤΙΩΣΗ)
% Αλλαγή 1: Αυξάνουμε το k σε 5 για λιγότερο overfitting
% Αλλαγή 2: Ενεργοποιούμε το 'Standardize' για να έχουν ίδια βαρύτητα τα χαρακτηριστικά
k = 5;
Mdl = fitcknn(Xtrain, Ytrain, ...
    'NumNeighbors', k, ...
    'Standardize', true);

%%  4. Πρόβλεψη 
Ypred = predict(Mdl, Xtest);

%% 5. Υπολογισμός Μετρικών (Confusion Matrix & Scores)
% Order: [0, 1] -> Η 1η γραμμή/στήλη είναι η κλάση 0, η 2η είναι η κλάση 1
C = confusionmat(Ytest, Ypred, 'Order', [0 1]);

% Δομή Πίνακα:
% [TN  FP]
% [FN  TP]
TN = C(1,1); % True Negatives (Σωστά Bound)
FP = C(1,2); % False Positives (Λάθος Unbound)
FN = C(2,1); % False Negatives (Λάθος Bound)
TP = C(2,2); % True Positives (Σωστά Unbound)

% Υπολογισμός Scores
Accuracy = (TP + TN) / sum(C(:));
Precision = TP / (TP + FP);
Recall = TP / (TP + FN); %Από όλα τα σωματίδια που όντως διεφυγαν (1), πόσα καταφέραμε να βρούμε;"
F1_Score = 2 * (Precision * Recall) / (Precision + Recall);

% Εκτύπωση Αποτελεσμάτων
fprintf('\n--- Model Evaluation ---\n');
fprintf('Accuracy:  %.2f%%\n', Accuracy * 100);
fprintf('Precision: %.4f\n', Precision);
fprintf('Recall:    %.4f\n', Recall);
fprintf('F1 Score:  %.4f\n', F1_Score);

%% --- 6. Οπτικοποίηση ---

% Α. Confusion Matrix Chart
figure('Name', 'Confusion Matrix');
confusionchart(Ytest, Ypred, ...
    'RowSummary','row-normalized', ...
    'ColumnSummary','column-normalized');
title('Confusion Matrix Results');

% Β. Decision Boundaries
figure('Name', 'Decision Boundary'); hold on;

% Σχεδίαση σημείων εκπαίδευσης
gscatter(Xtrain(:,1), Xtrain(:,2), Ytrain, 'rb', 'ox');
xlabel('Velocity (m/s)');
ylabel('Potential Energy (J)');
title(sprintf('k-NN Boundary (k=%d, Standardized)', k));

% Δημιουργία πλέγματος για το φόντο
[vGrid, uGrid] = meshgrid(linspace(min(velocity), max(velocity), 200), ...
                          linspace(min(potential), max(potential), 200));
gridPoints = [vGrid(:) uGrid(:)];

% Πρόβλεψη στο πλέγμα
gridPred = predict(Mdl, gridPoints);

% Σχεδίαση περιοχών
contourf(vGrid, uGrid, reshape(gridPred, size(vGrid)), ...
         'LineColor', 'none', 'FaceAlpha', 0.2);
legend('Bound (0)','Unbound (1)','Location','best');
hold off;