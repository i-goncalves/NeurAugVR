function [EEG_patterns] = coeff_estimates(M_model_subjects)

% INPUT 
% > M_model_subjects - Structure sx1 with s = number of subjects
% for each s, structure obtained from elastic net regression model

% OUTPUT
% > EEG_patterns - distribution of coefficient estimates in function of
% frequency and channels (averaged over delays) across subjects and sessions

    for s = 1:14
        features = M_model_subjects{s,2}(1).efp;
        features = features(2:end);
        for d = 1:6               % delay dimension
            for ch = 1:23         % channel dimension
                M_features_ses1(ch,:,d,s) = features(1:10);
                features = features(11:end);
            end
        end
    end

    M_mean_features_ses1 = squeeze(mean(M_features_ses1,[3 4]));

    for s = 1:14
        features = M_model_subjects{s,2}(2).efp;
        features = features(2:end);
        for d = 1:6
            for ch = 1:23
                M_features_ses2(ch,:,d,s) = features(1:10);
                features = features(11:end);
            end
        end
    end

    M_mean_features_ses2 = squeeze(mean(M_features_ses2,[3 4]));

    M_features(:,:,1) = M_mean_features_ses1;
    M_features(:,:,2) = M_mean_features_ses2;
    EEG_patterns = mean(M_features,3);
end
