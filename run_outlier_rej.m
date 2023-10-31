function [EEGCAP] = run_outlier_rej(EEG,method,win)

% In collaboration with Inês Esteves and Alexandre Perdigão

    nbchan = EEG.nbchan;
    TR = 1.26; % s

    nrchan2outlier = 2; 
    TR_outlier_percent = 2;

    channels = 1:nbchan;
    [newsig_capmean,  mavec, mivec] = outlier_rejection(EEG.data(1:nbchan,:),method,win);
    outliers_chan = zeros(nbchan, size(EEG.data, 2));
    
    for chan = 1:nbchan
        outliers_chan(chan,:) = (EEG.data(chan,:) > mavec(chan)) |  (EEG.data(chan,:) < mivec(chan));
    end
    
    outliers_allchan = sum(outliers_chan); 
    outliers_timecourse = outliers_allchan>0;
    outliers_total = sum(outliers_allchan>0);
    outliers_percentage = 100*(outliers_total/size(EEG.data, 2));
    outliers_totalperchan = sum(outliers_chan');
    
    EEGCAP = EEG;

    EEGCAP.data_raw = EEG.data;
    EEGCAP.data = newsig_capmean;

    EEGCAP.outlier.chanmax = mavec;
    EEGCAP.outlier.chanmin = mivec;
    EEGCAP.outlier.perchan = outliers_chan;
    EEGCAP.outlier.nbchan = outliers_allchan;
    EEGCAP.outlier.timecourse = outliers_timecourse;
    EEGCAP.outlier.total_perchan = outliers_totalperchan;
    EEGCAP.outlier.total_global = outliers_total;
    EEGCAP.outlier.percentage_global = outliers_percentage;  
    
    EEG_vol_ind = strcmp({EEG.event(:).type}, 'R128');
    EEG_vol_lat = [EEG.event(EEG_vol_ind).latency];
    EEGCAP.TR.nb = length(EEG_vol_lat);
    EEGCAP.TR.latencies = EEG_vol_lat;
    
    outliers_allchan_TR = reshape(outliers_allchan, [TR*EEGCAP.srate, EEGCAP.TR.nb]);    
    outliers_chan_criterion = outliers_allchan_TR >= nrchan2outlier;    
    outliers_perTR = sum(outliers_chan_criterion);
    outliers_perTR_percentage = 100*outliers_perTR/(TR*EEGCAP.srate);
    bad_TR = outliers_perTR_percentage>=TR_outlier_percent;
    
    EEGCAP.outlier.nbchan_perTR = outliers_allchan_TR;
    EEGCAP.outlier.nbOutliers_perTR = outliers_perTR;
    EEGCAP.outlier.outlierPercentage_perTR = outliers_perTR_percentage;
    EEGCAP.outlier.bad_TRs = bad_TR;
    
end
