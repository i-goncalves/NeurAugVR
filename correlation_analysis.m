%% EEG-fMRI Correlation Analysis
% Example for Resting State Data and EEG Absolute Power (All frequency
% bins)

% INPUT 
% > BOLD_ses1 and BOLD_ses2 - cell strcuture sx1 (s = nb of subjects)
% with nx1 BOLD signal from with n samples (zero mean and one std)

% > EEG_ses1 and EEG_ses2 - cell strcuture sx1 (s = nb of subjects)
% with ch x 1 cell structure (ch = nb of channels). For each ch, 
% PSD (fxn, with f = frequency bins)

% > freq - list of frequency for each bin

% > delay - delay for HRF convolution

% OUTPUT
% > T_corr - EEG-fMRI Correlation T-statistics (across subjects and
% sessions)

function [T_corr] = correlation_analysis(BOLD_ses1, BOLD_ses2, EEG_ses1, EEG_ses2, freq, delay)

    subs = [1,2,3,4,5,6,7,8,9,11,12,13,14,16,17];
    TR = 1.26;

    for s = 1:length(subs)

       % Session 1

       EEG_ses1_sub = EEG_ses1{s,1};
       BOLD_ses1_sub = BOLD_ses1{s,1};

       for ch = 1:31

           PSD_ch = EEG_ses1_sub{ch};

           for f = 1:numel(freq)
               PSD_ch_freq = PSD_ch(f,:);
               PSD_ch_freq = zscore(convolve_features_fast(PSD_ch_freq',1/TR,delay,32))';

               [rho,pval] = corr(BOLD_ses1_sub,PSD_ch_freq');
               M_corr(ch,f) = rho;

           end
       end

       M_corr_ses1{s,1} = M_corr(:,1:61);

       % Session 2

       EEG_ses2_sub = EEG_ses2{s,1};
       BOLD_ses2_sub = BOLD_ses2{s,1};

       for ch = 1:31

           PSD_ch = EEG_ses2_sub{ch};

           for f = 1:numel(freq)
               PSD_ch_freq = PSD_ch(f,:);
               PSD_ch_freq = zscore(convolve_features_fast(PSD_ch_freq',1/TR,delay,32))';

               [rho,pval] = corr(BOLD_ses2_sub,PSD_ch_freq');
               M_corr(ch,f) = rho;

           end
       end

       M_corr_ses2{s,1} = M_corr(:,1:61);

       % Organize correlation matrices

       for i = 1:length(subs)

           M_corr_sub_ses1(:,:,i) = M_corr_ses1{i,1};
           M_corr_sub_ses2(:,:,i) = M_corr_ses2{i,1};

       end

       % T-test across subjects and sessions

       M_corr_both_sessions(:,:,1:15) = M_corr_sub_ses1;
       M_corr_both_sessions(:,:,16:30) = M_corr_sub_ses2;
       T_corr = zeros(31,61);

       for i = 1:31 % Channels
           for j = 1:61 % Freq bins
               [h,p,ci,stats] = ttest(M_corr_both_sessions(i,j,:));
               T_corr(i,j) = stats.tstat;
           end
       end

    end
end
