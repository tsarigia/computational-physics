%% ΑΣΚΗΣΗ 1: ΠΡΟΕΠΕΞΕΡΓΑΣΙΑ ΔΕΔΟΜΕΝΩΝ
clc,clearvars

% Φόρτωση του αρχείου δεδομενων
load('sensor_data.mat');

% Εμφάνιση των πρώτων 10 τιμών!
disp('---(A) Πρωτες 10 τιμες ---');
disp('t (first 10):'); disp(t(1:10));
disp('x (first 10):'); disp(x(1:10));

% Β) Αντιμετώπιση Ελλιπών Τιμών (NaN)
nan_mask = ismissing(x);
nan_count = sum(nan_mask);
fprintf('\n Πλήθος NaN που βρέθηκαν: %d\n', nan_count);

% Αντικατάσταση με ΓΡΑΜΜΙΚΗ ΠΑΡΕΜΒΟΛΗ
x_filled = fillmissing(x, 'linear');

% Γ) Αναγνώριση και Αντικατάσταση Outliers
tf_outliers = isoutlier(x_filled);
num_outliers = sum(tf_outliers);
fprintf('Πλήθος Outliers που βρέθηκαν: %d\n', num_outliers);

% Αντικατάσταση outliers με γραμμική παρεμβολή
x_clean = filloutliers(x_filled, 'linear');

% Δ) Κανονικοποίηση
x_minmax = normalize(x_clean, 'range');
%Κανονικοποίηση ως z-scores
x_zscore = normalize(x_clean);
fprintf('Min-Max Range: [%.2f, %.2f]\n', min(x_minmax), max(x_minmax));
fprintf('Z-score Range: [%.2f, %.2f]\n', min(x_zscore), max(x_zscore));

% Ε) Γραφήματα
figure('Name', 'Επεξεργασία Sensor_data', 'Position', [100 100 900 400]);

% 1. Αρχικά Δεδομένα
subplot(4,1,1);
plot(t, x, 'b'); 
title('1. Αρχικά Δεδομένα');
xlabel('t (s)'); ylabel('x'); grid on;

% 2. Χωρίς NaN
subplot(4,1,2);
plot(t, x_filled, 'm');
title('2. Χωρίς NaN (fillmissing linear)');
xlabel('t (s)'); ylabel('x'); grid on;

% 3. Χωρίς Outliers (Τελικό Καθαρό Σήμα)
subplot(4,1,3);
plot(t, x_clean, 'g');
title('3. Χωρίς Outliers');
xlabel('t (s)'); ylabel('x'); grid on;

% 4. Εξομαλυμένα
subplot(4,1,4);
plot(t, x_minmax, 'r', 'LineWidth', 1.5); 
title('4. Κανονικοποιημένα Δεδομένα [0, 1]');
xlabel('t (s)'); ylabel('x'); grid on;

% ΣΤ) Αποθήκευση
writematrix([t(:), x_clean(:), x_minmax(:), x_zscore(:)], ...
            'Sol_Sensor_Data.csv');
save('Sol_Sensor_Data.mat', 't', 'x_clean', 'x_minmax', 'x_zscore');