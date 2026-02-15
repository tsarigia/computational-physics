%% ΑΣΚΗΣΗ 2: ΠΡΟΕΠΕΞΕΡΓΑΣΙΑ ΔΕΔΟΜΕΝΩΝ
clc,clearvars

T = readtable('physics_data_sample.csv');
data = table2array(T);

varNames = T.Properties.VariableNames;

fprintf('Αρχικές διαστάσεις δεδομένων: %d γραμμές x %d στήλες\n', size(data));

m_val = mean(data, 'omitnan');      % Μέσος όρος (για κάθε στήλη)
med_val = median(data, 'omitnan');  % Διάμεσος
s_val = std(data, 'omitnan');       % Τυπική απόκλιση

% 4. Εμφάνιση Αποτελεσμάτων
disp('Mean values per column:');
disp(m_val);

disp('Standard Deviation per column:');
disp(s_val);
%%
% X : Οι 6 στήλες με τις μετρήσεις (Energy...Time) ->επεξεργασία
X = data(:, 1:end-1);
y = data(:, end);

fprintf('Δεδομένα προς επεξεργασία (X): %d στήλες.\n', size(X,2));
fprintf('Στήλη ετικετών (y): %d.\n', size(y,2));

% α) Missing Values (Αντικατάσταση κενών)
nan_count = sum(ismissing(X), 'all');
X_filled = fillmissing(X, 'linear');
fprintf('Τιμες που αντικαθηστανται: %d\n', nan_count);

% β) Outliers (Εντοπισμός και αντικατάσταση)
num_outliers = sum(isoutlier(X_filled, 'median'), 'all');
X_clean = filloutliers(X_filled, 'linear', 'movmedian', 5);
fprintf('Outliers: %d\n', num_outliers);

% γ) Normalization (Κανονικοποίηση 0-1)
X_norm = normalize(X_clean, 'range');

% Ενώνουμε τις κανονικοποιημένες μετρήσεις με την αρχική στήλη Particle
Final_Matrix = [X_norm, y];
T_Final = array2table(Final_Matrix, 'VariableNames', varNames);

% Αποθήκευση σε CSV
writetable(T_Final, 'physics_data_final.csv');

% Αποθήκευση σε MAT
save('physics_data_final.mat', 'T_Final', 'Final_Matrix');