function ypred = knn_regress_sol(Xtrain,Ytrain,Xtest,k,metric)
%KNN_REGRESS k-NN Regression with multiple distance metrics
%   ypred = knn_regress(Xtrain, Ytrain, Xtest, k, metric)
%   metric options: 'euclidean', 'manhattan', 'chebyshev'
%   Default metric: 'euclidean' (αν παραλειφθεί)

% Αν δεν δώσουμε metric, 'euclidean'
    if nargin < 5
        metric = 'euclidean';
    end

    % Αρχικοποίηση πίνακα προβλέψεων (N x 1)
    ypred = zeros(size(Xtest, 1), 1);

    % Κύριος βρόχος για κάθε σημείο του Test set
    for i = 1:size(Xtest, 1)
        
        % Παίρνουμε το τρέχον σημείο ελέγχου (γραμμή i)
        testPoint = Xtest(i, :);
        
        % Υπολογισμός διαφοράς από όλα τα σημεία εκπαίδευσης
        % (Broadcasting: Το MATLAB αφαιρεί το testPoint από κάθε γραμμή του Xtrain)
        diffs = Xtrain - testPoint;

        % Επιλογή Μετρικής Απόστασης
        switch lower(metric)
            case 'euclidean'
                % Ευκλείδεια: sqrt(sum((x1-x2)^2))
                % Κατάλληλη για γενική χρήση
                d = sqrt(sum(diffs.^2, 2));
                
            case 'manhattan'
                % Manhattan (City Block): sum(|x1-x2|)
                d = sum(abs(diffs), 2);
                
            case 'chebyshev'
                % Chebyshev: max(|x1-x2|)
                d = max(abs(diffs), [], 2);
                
            otherwise
                error('Unknown metric. Use euclidean, manhattan, or chebyshev.');
        end

        % Ταξινόμηση και εύρεση των k κοντινότερων.
        [~, idx] = sort(d);              
        
        % Υπολογισμός μέσου όρου των τιμών των γειτόνων.(δείκτες) των κ πρώτων στην λίστα
        ypred(i) = mean(Ytrain(idx(1:k))); 
    end
end