%% 1. Προετοιμασία Δεδομένων
N = 2000; % Αριθμός δειγμάτων

% Δημιουργία τυχαίων εισόδων εντός των ορίων
L = 0.2 + (2.0 - 0.2) * rand(N, 1);% [0.2, 2.0] m
theta0_deg = 5 + (80 - 5) * rand(N, 1);% [5, 80] μοίρες
g = 9.6 + (9.9 - 9.6) * rand(N, 1);% [9.6, 9.9] m/s^2

% Μετατροπή γωνίας σε ακτίνια (radians)
theta0_rad = deg2rad(theta0_deg);

% Υπολογισμός της θεωρητικής περιόδου T (με την προσέγγιση μεγάλων γωνιών)
% T ≈ 2π√(L/g) * [1 + (1/16)θ² + (11/3072)θ⁴]
T_theory = 2 * pi * sqrt(L ./ g) .* (1 + (1/16)*theta0_rad.^2 + (11/3072)*theta0_rad.^4);

% Προσθήκη Γκαουσιανού θορύβου (π.χ. 2% standard deviation)
noise_level = 0.02; 
T = T_theory + noise_level * randn(N, 1);

% Συγκέντρωση εισόδων σε πίνακα (κάθε στήλη είναι ένα χαρακτηριστικό)
inputs = [L, theta0_rad, g]'; % μετατροπη για να είναι 3xN
targets = T';         % Μετατροπη για να είναι 1xN

%Κανονικοποίηση (Normalization) στο [-1, 1]
[inputs_norm, settings_in] = mapminmax(inputs, -1, 1);
[targets_norm, settings_out] = mapminmax(targets, -1, 1);

%Με την χρήση της mapminmax για κανονικοποίηση στο εύρος [-1, 1] βοηθαμε το ANN να συγκλίνει ταχύτερα, καθώς όλες οι είσοδοι έχουν την ίδια κλίμακα.


% 4. Δημιουργία feedforward neural network με 10 νευρώνες
net = feedforwardnet(10);

% 5. Διαχωρισμός δεδομένων (70% Train, 15% Val, 15% Test)
net.divideParam.trainRatio = 0.70;
net.divideParam.valRatio = 0.15;
net.divideParam.testRatio = 0.15;

% 6. Εκπαίδευση του δικτύου
[net, tr] = train(net, inputs_norm, targets_norm);


% Γ) Αξιολόγηση
% Ανάκτηση δεδομένων ελέγχου (test set)
testIdx = tr.testInd;
testInputsNorm = inputs_norm(:, testIdx);
testTargetsActual = targets(:, testIdx);

% Πρόβλεψη και απο-κανονικοποίηση
outputsNorm = net(testInputsNorm);
outputsPredicted = mapminmax('reverse', outputsNorm, settings_out);

% 7. Υπολογισμός δεικτών σφάλματος
errors = testTargetsActual - outputsPredicted;

MSE  = mean(errors.^2);
MAE  = mean(abs(errors));
RMSE = sqrt(MSE);
MAPE = mean(abs(errors ./ testTargetsActual)) * 100;

% Εμφάνιση αποτελεσμάτων
fprintf('Στατιστικά Σφάλματος\n');
fprintf('MSE:  %.6f\n', MSE);
fprintf('MAE:  %.6f\n', MAE);
fprintf('RMSE: %.6f\n', RMSE);
fprintf('MAPE: %.2f%%\n', MAPE);

% 8. Γράφημα πραγματικής έναντι προβλεπόμενης περιόδου
figure;
plot(testTargetsActual, outputsPredicted, 'b.');
hold on;
line_vals = [min(testTargetsActual) max(testTargetsActual)];
plot(line_vals, line_vals, 'r', 'LineWidth', 2); % Γραμμή y=x
xlabel('Actual T (s)'); ylabel('Predicted T (s)');
title('ANN Regression: Actual vs Predicted');
legend('Predictions', 'Ideal (y=x)');
grid on;

% Παρατηρουμε οτι στις μικρές γωνίες, η περίοδος είναι ανεξάρτητη της γωνίας, ενω για μεγάλες γωνίες, η μη γραμμικότητα γίνεται έντονη.
% Με το ANN είναι ιδανική εδώ, καθώς το δίκτυο προσεγγίζει αυτές τις μη γραμμικές συναρτήσεις χωρίς να δώσουμε  τον τύπο της Taylor.
%Ένα κρυφό στρώμα είναι ηταν αρκετό για να προσεγγίστει η συνεχή συνάρτηση.
%$R = 0.99: Σημαίνει ότι το δίκτυο έμαθε τη σχέση σχεδόν τέλεια.
%Το RMSE δίνει το σφάλμα στις μονάδες της περιόδου (δευτερόλεπτα),το MAPE δείχνει το ποσοστιαίο σφάλμα.

