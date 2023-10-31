% File paths

path_design = '\\Thesis\\DATA\\design_matrices\\sub-%02d\\ses-%s\\task-%s'; % GLM Design matrices
path_pe = '\\Thesis\\DATA\\PEs\\sub-%02d\\ses-%s\\task-%s'; % PE files
path_data = '\\Thesis\\DATA\\filtered_func\\sub-%02d\\ses-%s\\task-%s'; % fMRI data
path_final = '\\Thesis\\DATA\\reconstruct_data\\sub-%02d\\ses-%s\\task-%s'; % path for final files (clean fMRI signal)

% Example Task NeurowMIMO, Session 2, Run 3

% Regress out motion regressors from the data (y_clean = y - (sum(MPs) +
% sum (MOs)))

task = 'neurowMIMO_run-03';
ses= 'inside2';

sub = [1,2,3,4,5,6,7,8,9,11,12,13,14,16,17];

for s = 1:length(sub)
    l = sub(s);

    design_mat = dlmread(strcat('D:',sprintf(path_design,l,ses,task), '\design.mat'));
    num_volumes = length(design_mat);

    dir_PE=dir([strcat('D:',sprintf(path_pe,l,ses,task)) '/*.nii.gz']);
    num_PEs=size(dir_PE,1);

    PEs_grouped = cell(1,num_PEs);
    for p = 1:num_PEs
        add_PE = niftiread(strcat('D:',sprintf(path_pe,l,ses,task),'\pe',int2str(p),'.nii.gz'));
        PEs_grouped{1,p} = add_PE;
    end

    filtered_func_data = niftiread(strcat('D:',sprintf(path_data,l,ses,task),'\filtered_func_data.nii.gz'));
    header = niftiinfo(strcat('D:',sprintf(path_data,l,ses,task),'\filtered_func_data.nii.gz'));

    x = size(PEs_grouped{1},1);
    y = size(PEs_grouped{1},2);
    z = size(PEs_grouped{1},3);
    
    PE_motion = num_PEs - 4;
    M = cell(1,PE_motion);
    
    for m = 1:PE_motion

        M_element = zeros(x,y,z,num_volumes);
        for i = 1:x
            for j = 1:y
                for k = 1:z
                    M_element(i,j,k,:) = PEs_grouped{1,m+4}(i,j,k)*design_mat(:,m+4);
                end
            end
        end
        M{1,m} = M_element;
    end
    for n = 1:length(M)
        final_data = filtered_func_data - M{1,n};
        filtered_func_data = final_data;
    end
    niftiwrite(final_data,strcat('D:',sprintf(path_final,l,ses,task),'\final_data'),header);
    clear
end