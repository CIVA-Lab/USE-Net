% count correctly classified nuclei and missing nuclei in each predicted mask
function count = countNuclei(correct_mask,predicted_map)
    correct_list = unique(correct_mask); % set of unique gt nuclei
    correct_list = correct_list(2:end); % exclude 0
    ncorrect = numel(correct_list);

    predicted_indices = nonzeros(unique(predicted_map));

    % Count correctly classified nuclei and missing nuclei
    miss_nuc = 0; % missing nuclei
    correct_nuc = 0; % correctly classified nuclei

    for c = 1:ncorrect
        % fprintf('Processing object # %d \n',c);
        temp_mask = (correct_mask==correct_list(c));
         pred = temp_mask.*predicted_map;%Has intersecting unique labels 
         matched_indices = nonzeros(unique(pred));

         if ~nnz(matched_indices) == 0%If non-zero, find intersecting pixels
                intersection_pixels = [];
                for i=1:numel(matched_indices)
                   temp = temp_mask.* (pred==matched_indices(i));
                   intersection_pixels(i) = sum(temp(:));      
                end

            [n idx]= max(intersection_pixels);
            matched_idx = matched_indices(idx);

            predicted_map(predicted_map == matched_idx) = 0;
            predicted_indices(predicted_indices == matched_idx) = [];

            correct_nuc = correct_nuc+1;
         else
             miss_nuc = miss_nuc+1;
         end

    count = [correct_nuc miss_nuc];

    end
end
