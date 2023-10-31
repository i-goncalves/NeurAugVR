function [newdata, mavec, mivec] = outlier_rejection(data, method, win)

% In collaboration with Inês Esteves and Alexandre Perdigão

% [newdata, mavec, mivec] = outlier_rejection(data, method)

% INPUT
% > data - nxm EEG signal, with n channels and m timepoints
% > method - interpolation method for outlier points exceeding mean+4*std
% or mean-4*std of the signal in each EEG channel
% -- mean - computes the mean of the point itself and neighbouring points
% (default window = 2)
% -- capmean - computes the mean of the point itself and neighbouring
% points (window = 1), but all the values contributing to this mean are capped 
% at mean+4*std or mean-4*std, if they are beyond these limits (default window = 1)
% > win - scalar indicating the size of the neighbourhood window, e.g. if
% win = 1, 1 point on the left and 1 point on the right of the outlier will
% be considered. In case there are not enough points on either side, the outlier
% is replaced by mean+4*std or mean-4*std, depending on its original value

% OUTPUT
% > newdata - nxm EEG signal, with n channels and m timepoints. For each
% channel, points exceeding mean+4*std or mean-4*std of the original signal
% have been interpolated using the chosen method
% > mavec - mx1 vector with the outlier upper limit for each channel
% > mivec - mx1 vector with the outlier lower limit for each channel

    newdata = zeros(size(data));
    mavec = zeros(size(data,1), 1);
    mivec = zeros(size(data,1), 1);
    switch method
        case 'capmean'
            for c = 1:size(data,1)
                chan = c;
                signal = data(chan, :);
                msig = mean(signal);
                stdsig = std(signal);
                ma = msig + 4*stdsig;
                mi = msig - 4*stdsig;

                out_ind = find((signal > ma) | (signal < mi));
                newsig = signal;
                for k = 1:length(out_ind)
                    ind = out_ind(k);
                    if (ind > win) && (ind < length(signal)-win)
                        sig_win = newsig(ind-win:ind+win);
                        sig_win(sig_win > ma) = ma;
                        sig_win(sig_win < mi) = mi;
                        val_ind = mean(sig_win);
                        newsig(ind) = val_ind;
                    elseif newsig(ind) > ma
                        newsig(ind) = ma;
                    elseif newsig(ind) < mi
                        newsig(ind) = mi;
                   end
                end
                newdata(c, :) = newsig;
                mavec(c,:) = ma;
                mivec(c,:) = mi;
            end
            
            
        case 'mean'
            
            for c = 1:size(data,1)
                chan = c;
                signal = data(chan, :);
                msig = mean(signal);
                stdsig = std(signal);
                ma = msig + 4*stdsig;
                mi = msig - 4*stdsig;

                out_ind = find((signal > ma) | (signal < mi));

                newsig = signal;
                for k = 1:length(out_ind)
                    ind = out_ind(k);
                    if (ind > win) && (ind < length(signal)-win)
                        ind = out_ind(k);
                        sig_win = newsig(ind-win:ind+win);
                        val_ind = mean(sig_win);
                        newsig(ind) = val_ind;
                    elseif newsig(ind) > ma
                        newsig(ind) = ma;
                    elseif newsig(ind) < mi
                        newsig(ind) = mi;
                   end
                 end
                 newdata(c, :) = newsig;
                 mavec(c,:) = ma;
                 mivec(c,:) = mi;
            end
        otherwise
            disp('Unrecognized method')
    end
end
