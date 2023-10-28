% Elastic Net Linear Regression Model

% In collaboration with Marta Xavier (https://github.com/martaxavier)

function [M_subject_model] = elastic_reg_blocked_k_fold(BOLD_ses1, BOLD_ses2, EEG_ses1, EEG_ses2)

% INPUT 
% > BOLD_ses1 and BOLD_ses2 - nx1 BOLD signal from each session with n
% samples (zero mean and one std)
% > EEG_ses1 and EEG_ses2 - nxp design matrix from each session with n
% samples and p EEG features (zero mean and one std)

% OUTPUT
% > M_subject_model - cell array that contains test, learn, and validation
% related measures

    M_subject_model = struct([]);

    % Two outer folds -> one session is the learning set, the other the
    % testing set

    BOLD_sessions = [BOLD_ses1 BOLD_ses2];
    EEG_sessions = [EEG_ses1 EEG_ses2];

    K = 2;
    
    for k = 1:K

        EEG_learn = EEG_sessions(:,k);
        EEG_test = EEG_sessions(:,1:end ~= k);

        BOLD_learn = BOLD_sessions(:,k);
        BOLD_test = BOLD_sessions(:,1:end ~= k);

        siz_test = size(EEG_test,1);
        siz_learn = size(EEG_learn,1);

        % Number of samples (time-points)

        n_pnts = length(BOLD_learn);

        % Number of regressors in the model

        n_features = size(EEG_train, 2);

        % Define fixed value of alpha hyperparameter
        alpha = 0.5;

        % Get list of lambdas (regularization hyperparameter)

        lambda = get_lambdas(EEG_learn,BOLD_learn,alpha);
        % or define broad list
        % lambda = logspace(-4,0,20);

        M = 15;

        % Number of lambda values

        n_lambda = length(lambda);

        % Validation Measures

        bic_val = zeros(n_lambda, M); nmse_val = bic_val;
        mse_val = bic_val;
        corr_val = zeros(n_lambda,M); corr_train = corr_val;
        pval_train = corr_val; pval_val = corr_val;
        aic_val = zeros(n_lambda, M);
        lambda_val = zeros(1,M);

        df_inner = zeros(n_lambda, M);

        %-------------- Begin Inner CV loop ---------------%

        % Divide time-series into M total consecutive folds
        % blocks of samples

        indices_out = sort(crossvalind('Kfold', n_pnts, M));

        for m = 1 : M

            % Assign test set indices
            ind_val = (indices_out == m);
            ind_val = find(ind_val);

            ind_train = (indices_out ~= m);
            ind_train = find(ind_train);

            siz_val = length(ind_val);

            % Assign train and validation X (EEG) and Y (BOLD) variables

            X_train = EEG_learn(ind_train, :); y_train = BOLD_learn(ind_train);
            X_val = EEG_learn(ind_val, :); y_val = BOLD_learn(ind_val);

            X_train = zscore(X_train);
            y_train = zscore(y_train);

            X_val = zscore(X_val);
            y_val = zscore(y_val);

            % Elastic-net regression fit

            [betas_CV, stats_CV] = lasso(X_train, y_train, ...
                'Alpha', rho, 'Lambda', lambda);

            df = stats_CV.DF;
            df = flip(df)';
            betas_CV = flip(betas_CV, 2);

            % Save intercept values for current rho-lambda

            intercept = stats_CV.Intercept;
            intercept=flip(intercept);
            y_hat_val = intercept + X_val*betas_CV;
            y_hat_train = intercept + X_train*betas_CV;

            % Compute validation measures for all rho-lambda
            % pairs in the val and train set

            mse_val(:,m) = sum((y_hat_val - y_val).^2)';
            nmse_val(:, m) = mse_val(:,m)/sum((y_val - mean(y_val)).^2);

            [corr_val(:,m),pval_val(:,m)] = corr(y_hat_val,y_val,'Tail','right');
            [corr_train(:,m),pval_train(:,m)] = corr(y_hat_train,y_train,'Tail','right');

            bic_val(:, m) = (log(siz_val).*df) + siz_val.*log(mse_val(:,m) ./ siz_val);
            aic_val(:, m) = (2*df) + siz_val.*log(mse_val(:,m) ./ siz_val);
            df_inner(:,m) = df;

            % Lmabda chosen for each fold (minimum NMSE)

            [~, ind_opt_val] = min(nmse_val(:,m));

            lambda_val(m) = lambda(ind_opt_val);

        end

        [~, ind_opt] = min(sum(nmse_val, 2));

        % Save rho-lambda pair that minimizes NMSE criterion in CV set

        opt_lambda = lambda(ind_opt);

        [betas, stats] = lasso(EEG_learn, ...
            BOLD_learn, 'Alpha', rho, 'Lambda', opt_lambda);

        opt_df = stats.DF;

        learn.efp =     [stats.Intercept; betas];
        learn.df =      length(find(learn.efp))-1;
        learn.yhat =    learn.efp(1) + EEG_learn*learn.efp(2:end);
        y_hat_test = stats.Intercept + EEG_test*betas;

        % ------------------------------------------------------------
        % Prepare output data
        % ------------------------------------------------------------

        % Estimated hyperparameters of the model

        learn.lambda =  lam;
        learn.rho =     rho;

        % Compute accuracy on the learning set

        learn.mse =     sum((learn.yhat - BOLD_learn).^2);
        learn.bic =     log(n_pnts).* learn.df + n_pnts.* log(learn.mse ./ n_pnts);
        learn.nmse =    learn.mse / sum((BOLD_learn - mean(BOLD_learn)).^2);
        [learn.corr,learn.pval] =    corr(learn.yhat, BOLD_learn,'Tail','right');

        % Prediction performance (accuracy on the test set)

        opt_mse_test = sum((y_hat_test - BOLD_test).^2)';

        opt_bic_test = log(siz_test) * opt_df + siz_test.* log(opt_mse_test / siz_test);

        opt_nmse_test = opt_mse_test / sum((BOLD_test - mean(BOLD_test)).^2);

        [opt_corr_test,opt_pval_test] = corr(y_hat_test,BOLD_test,'Tail','right');

        opt_mse_learn = sum((learn.yhat - BOLD_learn).^2);

        opt_bic_learn = log(siz_learn) * opt_df + siz_learn.* log(opt_mse_learn / siz_learn);

        opt_nmse_learn = opt_mse_learn / sum((BOLD_learn - mean(BOLD_learn)).^2);

        [opt_corr_learn,opt_pval_learn] = corr(learn.yhat,BOLD_learn,'Tail','right');

        % Save prediction performance

        optimal.bic_test = opt_bic_test;
        optimal.mse_test = opt_mse_test;
        optimal.nmse_test = opt_nmse_test;
        optimal.corr_test = opt_corr_test;
        optimal.pval_test = opt_pval_test;
        optimal.df = opt_df;

        % Save learning performance

        optimal.bic_learn = opt_bic_learn;
        optimal.mse_learn = opt_mse_learn;
        optimal.nmse_learn = opt_nmse_learn;
        optimal.corr_learn = opt_corr_learn;
        optimal.pval_learn = opt_pval_learn;

        % Save real and estimated BOLD

        optimal.yhat_test = y_hat_test;
        optimal.yhat_learn = learn.yhat;
        optimal.BOLD_test = BOLD_test;
        optimal.BOLD_learn = BOLD_learn;

        % Validation 

        validation_measures.bic = bic_val;
        validation_measures.nmse = nmse_val;
        validation_measures.mse = mse_val;
        validation_measures.aic = aic_val;
        validation_measures.corr_val = corr_val;
        validation_measures.pval_val = pval_val;
        validation_measures.corr_train = corr_train;
        validation_measures.pval_train = pval_train;
        validation_measures.df = df_inner;
        validation_measures.lambda = lambda;
        validation_measures.y_hat_val = y_hat_val;
        validation_measures.y_val = y_val;
        validation_measures.y_hat_train = y_hat_train;
        validation_measures.y_train = y_train;
        validation_measures.lambda_val = lambda_val;

        test_cycles(1,k) = optimal;
        validation_cycles(1,k) = validation_measures;
        learn_cycles(1,k) = learn;

    end

    M_subject_model{1,1} = test_cycles;
    M_subject_model{1,2} = learn_cycles;
    M_subject_model{1,3} = validation_cycles;

end
