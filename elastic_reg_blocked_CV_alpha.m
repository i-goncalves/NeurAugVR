% Elastic Net Linear Regression Model
% Alpha Optimiation

% In collaboration with Marta Xavier (https://github.com/martaxavier)

function [M_subject_model] = elastic_reg_blocked_CV_alpha(BOLD_ses1, BOLD_ses2, EEG_ses1, EEG_ses2)

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

    % Define list of alpha values

    alpha = 10.^linspace(-2,0,30);

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

        % Get list of lambdas (regularization hyperparameter)

        lambdas = zeros(length(alpha),20);
        for j = 1:length(alpha)
            lambdas(j,:) = get_lambdas(EEG_learn,BOLD_learn,alpha(j));
        end

        % (get_lambdas function:
        % https://github.com/martaxavier/Mig_N2Treat/blob/main/get_lambdas.m)
        % or define broad list
        % lambda = logspace(-4,0,20);

        M = 15;

        [m,p] = ndgrid(alpha,1:M);
        grid = [m(:),p(:)];

        val_curves = zeros(size(alpha,2)*M,size(lambdas,2));
        val_size = floor(size(BOLD_learn,1) ./ M);

        %-------------- Begin Inner CV loop ---------------%

        for m = 1:size(grid,1)

                alpha_ind = grid(m,1);
                fold = grid(m,2);
                val_ind = (fold-1)*val_size+1:(fold-1)*val_size + val_size;
                train_ind = linspace(1,size(EEG_learn,1),size(EEG_learn,1));
                train_ind(val_ind) = [];

                EEG_train = EEG_learn(train_ind,:);
                BOLD_train = BOLD_learn(train_ind,:);
                EEG_val = EEG_learn(val_ind,:);
                BOLD_val = BOLD_learn(val_ind,:);

                EEG_train = zscore(EEG_train);
                BOLD_train = zscore(BOLD_train);
                EEG_val = zscore(EEG_val);
                BOLD_val = zscore(BOLD_val);

                alpha_k = find(alpha == alpha_ind);

                % L2+1 elastic-net regression fit
                [B,FitInfo] = lasso(EEG_train,BOLD_train,'Alpha',alpha_ind,'Lambda',lambdas(alpha_k,:));

                df(m,:) = FitInfo.DF;
                df(m,:) = flip(df(m,:),2);
                intercept = FitInfo.Intercept;
                intercept = flip(intercept);
                B = flip(B, 2);

                valpred = [ones(size(EEG_val,1),1) , EEG_val ] * [intercept; B];
                siz_CV = size(valpred,1);

                % NMSE

                valscore = sum((repmat(BOLD_val,1,size(lambdas,2)) - valpred).^2,1)/sum((BOLD_val - mean(BOLD_val)).^2);
                val_curves(m,:) = valscore;
        end

        val_plane = zeros(size(alpha,2),M,size(lambdas,2));
        val_plane = reshape(val_curves,[size(alpha,2),M,size(lambdas,2)]);
        mean_plane = squeeze(mean(val_plane,2));

        % Select Optimal Hyperparameters

        [~,index_best_lambda]=min(min(mean_plane,[],1)); 
        [~,index_best_alpha]=min(min(mean_plane,[],2)); 

        [B_fin,FitInfo_fin]=lasso(EEG_learn,BOLD_learn,'Alpha',alpha(index_best_alpha),'Lambda',lambdas(index_best_alpha,index_best_lambda));

        opt_df = FitInfo_fin.DF;

        learn.efp =     [FitInfo_fin.Intercept; B_fin];
        learn.df =      length(find(learn.efp))-1;
        learn.yhat =    learn.efp(1) + EEG_learn*learn.efp(2:end);
        y_hat_test = FitInfo_fin.Intercept + EEG_test*B_fin;

        % ------------------------------------------------------------
        % Prepare output data
        % ------------------------------------------------------------

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

        validation_measures.mean_plane = mean_plane;
        validation_measures.val_curves = val_curves;
        validation_measures.df = df;
        validation_measures.lambda_list = lambdas;
        validation_measures.alpha_list = alpha;

        test_cycles(1,k) = optimal;
        validation_cycles(1,k) = validation_measures;
        learn_cycles(1,k) = learn;

    end

    M_subject_model{1,1} = test_cycles;
    M_subject_model{1,2} = learn_cycles;
    M_subject_model{1,3} = validation_cycles;
    
end